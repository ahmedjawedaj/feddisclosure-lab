"""Wire-level capture and analysis helpers for the deployed Flower SecAgg+ stack (v0.15, WP3).

Two observers are modelled.

* The server view is a recording proxy around the ServerApp grid. It sees every SecAgg+ stage
  message, the masked vectors, the clear sample counts and metrics, and the final dequantised
  aggregate. `WireRecorder` stores that view on disk.
* The network view is a plain TCP relay in front of every SuperNode (docker/wire_relay/relay.py).
  It logs byte counts, timestamps and rolling hashes and dumps the raw streams. This module only
  reads those dumps back. It parses HTTP/2 framing and gRPC message framing in the analysis, never
  in the relay.

The two views are linked by content hash. A proxied object (a Flower array or a bytes value) is
identified by the SHA-256 of its exact bytes. The same bytes appear inside the relay stream, so the
hash can be searched for there.

Nothing here implements any cryptography. This module has no import of Flower or PyTorch at load
time so that the relay parsing and the check logic run anywhere. Synthetic stand-ins for Flower
messages are provided for tests and for the dry-run demo.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import threading
import time
import zlib
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
STAGES = ("setup", "share_keys", "collect_masked_vectors", "unmask")
SECAGG_CONFIG_RECORD = "secaggplus_configs"
STAGE_KEY = "stage"
MASKED_KEY = "masked_params"
MIN_OBJECT_BYTES = 16
H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
NPY_MAGIC = b"\x93NUMPY"
CLIP_RANGE = 8.0
QUANT_RANGE = 2 ** 22
MODEL_SHAPES = ((32, 64), (32,), (10, 32), (10,))

# Pre-registered thresholds (docs/preregistration/v0.15.md rows W1 to W6). Never change them.
W1_MARGIN = 0.05
W3_RATE_TOLERANCE = 0.10
W3_MSE_TOLERANCE = 0.01
W3_MIN_SEEDS = 5
W4_MIN_RATE = 0.85


class WireCaptureError(RuntimeError):
    """A capture or analysis step failed. The caller must treat the run as unusable."""


# ---------------------------------------------------------------------------------------------
# Hashing and message description
# ---------------------------------------------------------------------------------------------

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def ndarray_to_npy_bytes(arr: np.ndarray) -> bytes:
    """The same serialisation Flower uses for one array (numpy save format, no pickle)."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(arr), allow_pickle=False)
    return buffer.getvalue()


def npy_bytes_to_ndarray(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def flower_chunk_object_id(blob: bytes) -> str:
    """Object id Flower gives to an array stored as a single chunk.

    The object content is a short text header followed by the array bytes and the id is the
    SHA-256 of that content. Only valid when the whole array fits in one chunk, which holds for the
    model sizes used here. It is an extra cross-check and no pass rule depends on it.
    """
    return sha256_hex(b"ArrayChunk  " + str(len(blob)).encode() + b"\x00" + blob)


def _plain(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"bytes_sha256": sha256_hex(raw), "nbytes": len(raw)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def describe_content(content) -> dict:
    """Canonical JSON-safe description of a record dictionary. Arrays and bytes are reduced to hashes."""
    arrays = {}
    for rname, rec in content.array_records.items():
        arrays[rname] = {key: {"dtype": str(a.dtype), "shape": list(a.shape), "stype": str(a.stype),
                               "nbytes": len(a.data), "sha256": sha256_hex(a.data)}
                         for key, a in rec.items()}
    metrics = {rname: {k: _plain(v) for k, v in rec.items()} for rname, rec in content.metric_records.items()}
    configs = {rname: {k: _plain(v) for k, v in rec.items()} for rname, rec in content.config_records.items()}
    return {"array_records": arrays, "metric_records": metrics, "config_records": configs}


def content_sha256(content) -> str:
    text = json.dumps(describe_content(content), sort_keys=True, separators=(",", ":"), default=repr)
    return sha256_hex(text.encode("utf-8"))


def iter_objects(content):
    """Every array and every bytes value in a record dictionary, with the SHA-256 of its exact bytes."""
    for rname, rec in content.array_records.items():
        for key, arr in rec.items():
            data = bytes(arr.data)
            yield {"kind": "array", "record": rname, "key": key, "index": None, "dtype": str(arr.dtype),
                   "shape": list(arr.shape), "nbytes": len(data), "sha256": sha256_hex(data)}
    for rname, rec in content.config_records.items():
        for key, value in rec.items():
            items = [value] if isinstance(value, (bytes, bytearray)) else (
                list(value) if isinstance(value, (list, tuple)) else [])
            for index, item in enumerate(items):
                if isinstance(item, (bytes, bytearray)):
                    raw = bytes(item)
                    yield {"kind": "bytes", "record": rname, "key": key, "index": index, "dtype": None,
                           "shape": None, "nbytes": len(raw), "sha256": sha256_hex(raw)}


def _has_content(msg) -> bool:
    has = getattr(msg, "has_content", None)
    return bool(has()) if callable(has) else True


def infer_stage(messages) -> str:
    """SecAgg+ stage name of an outgoing batch, or plain_<message type> for a run without SecAgg+."""
    message_type = "message"
    for msg in messages:
        if not _has_content(msg):
            continue
        message_type = str(getattr(msg.metadata, "message_type", message_type))
        configs = msg.content.config_records
        if SECAGG_CONFIG_RECORD in configs and STAGE_KEY in configs[SECAGG_CONFIG_RECORD]:
            return str(configs[SECAGG_CONFIG_RECORD][STAGE_KEY])
    return "plain_" + message_type


def _decode_masked(content) -> list[np.ndarray] | None:
    for rec in content.config_records.values():
        if MASKED_KEY in rec:
            return [npy_bytes_to_ndarray(bytes(b)) for b in rec[MASKED_KEY]]
    return None


def _reply_facts(content) -> dict:
    """Sample count and metrics as the server sees them in clear text."""
    num_examples, found = None, False
    for rec in content.metric_records.values():
        if "num_examples" in rec:
            num_examples, found = _plain(rec["num_examples"]), True
    metrics, metrics_visible = None, False
    for rname, rec in content.config_records.items():
        if rname.endswith(".metrics"):
            metrics, metrics_visible = {k: _plain(v) for k, v in rec.items()}, True
    return {"num_examples": num_examples, "num_examples_visible": found,
            "metrics": metrics, "metrics_visible": metrics_visible}


# ---------------------------------------------------------------------------------------------
# Server-side recorder
# ---------------------------------------------------------------------------------------------

class WireRecorder:
    """Append-only record of everything the ServerApp sends and receives.

    One JSON line per message in `messages.jsonl`. Arrays that matter for the analysis are saved
    as npz files under `arrays/`: the global model that goes out (public to every participant),
    plaintext replies in a run without SecAgg+, and the masked vectors of the SecAgg+ collect stage.
    No client dataset is ever read here.
    """

    def __init__(self, directory: str | Path, *, mode: str, meta: dict | None = None, clock=time.time):
        if mode not in ("secagg", "plaintext"):
            raise ValueError("mode must be secagg or plaintext")
        self.dir = Path(directory)
        self.mode, self.meta, self.clock = mode, dict(meta or {}), clock
        if (self.dir / "messages.jsonl").exists():
            raise FileExistsError("Refusing to append to an existing capture")
        (self.dir / "arrays").mkdir(parents=True, exist_ok=True)
        self._fh = (self.dir / "messages.jsonl").open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0
        self._call = 0
        self._pending: dict[str, tuple[int, str]] = {}
        self._stage_counts: dict[str, dict[str, int]] = {}
        self._global_files: dict[str, str] = {}
        self._aggregates: dict[str, dict] = {}
        self._started = clock()
        self._finished = False

    # -- helpers --------------------------------------------------------------------------
    def _save_npz(self, relpath: str, arrays: list[np.ndarray], prefix: str) -> str:
        np.savez(self.dir / relpath, **{f"{prefix}{i}": np.asarray(a) for i, a in enumerate(arrays)})
        return relpath

    def _write(self, entry: dict) -> None:
        self._fh.write(json.dumps(entry, sort_keys=True, default=repr) + "\n")
        self._fh.flush()

    def _record_one(self, msg, direction: str, call: int, stage: str) -> None:
        self._seq += 1
        seq = self._seq
        md = msg.metadata
        node = int(md.dst_node_id if direction == "out" else md.src_node_id)
        entry = {"seq": seq, "call": call, "direction": direction, "stage": stage, "mode": self.mode,
                 "node_id": node, "message_type": str(getattr(md, "message_type", "")),
                 "group_id": str(getattr(md, "group_id", "")), "t": self.clock(), "error": None,
                 "arrays_file": None, "masked_file": None}
        if not _has_content(msg) or (hasattr(msg, "has_error") and msg.has_error() and not _has_content(msg)):
            err = getattr(msg, "error", None)
            entry.update(error=str(getattr(err, "reason", err)), payload_bytes=0, content_sha256=None,
                         objects=[], content=None)
            self._count(stage, direction)
            self._write(entry)
            return
        content = msg.content
        objects = list(iter_objects(content))
        count_bytes = getattr(content, "count_bytes", None)
        entry.update(payload_bytes=int(count_bytes()) if callable(count_bytes) else sum(o["nbytes"] for o in objects),
                     content_sha256=content_sha256(content), objects=objects, content=describe_content(content))
        entry.update(_reply_facts(content) if direction == "in" else {})
        array_list = [a.numpy() if hasattr(a, "numpy") else np.asarray(a)
                      for rec in content.array_records.values() for a in rec.values()]
        if array_list:
            digest = sha256_hex(b"".join(o["sha256"].encode() for o in objects if o["kind"] == "array"))[:16]
            if direction == "out":
                rel = self._global_files.get(digest)
                if rel is None:
                    rel = self._save_npz(f"arrays/global_{digest}.npz", array_list, "a")
                    self._global_files[digest] = rel
            else:
                rel = self._save_npz(f"arrays/update_{seq:04d}_node{node}.npz", array_list, "a")
            entry["arrays_file"] = rel
        if direction == "in":
            masked = _decode_masked(content)
            if masked is not None:
                entry["masked_file"] = self._save_npz(f"arrays/masked_{seq:04d}_node{node}.npz", masked, "m")
        self._count(stage, direction)
        self._write(entry)

    def _count(self, stage: str, direction: str) -> None:
        row = self._stage_counts.setdefault(stage, {"out": 0, "in": 0})
        row[direction] += 1

    # -- public recording API --------------------------------------------------------------
    def record_out(self, messages) -> tuple[int, str]:
        with self._lock:
            self._call += 1
            call, stage = self._call, infer_stage(messages)
            for msg in messages:
                self._record_one(msg, "out", call, stage)
            return call, stage

    def record_in(self, call: int, stage: str, messages) -> None:
        with self._lock:
            for msg in messages:
                self._record_one(msg, "in", call, stage)

    def remember_pushed(self, messages, message_ids, call: int, stage: str) -> None:
        with self._lock:
            for msg, mid in zip(messages, message_ids):
                self._pending[str(mid)] = (call, stage)

    def lookup_pending(self, message) -> tuple[int, str]:
        with self._lock:
            return self._pending.get(str(getattr(message.metadata, "reply_to_message_id", "")), (-1, "unknown"))

    def record_aggregate(self, name: str, arrays: list[np.ndarray]) -> None:
        """Save an aggregate (for example the dequantised SecAgg+ result) with a content hash."""
        with self._lock:
            arrs = [np.asarray(a) for a in arrays]
            rel = self._save_npz(f"{name}.npz", arrs, "a")
            self._aggregates[name] = {"file": rel, "dtype": [str(a.dtype) for a in arrs],
                                      "shapes": [list(a.shape) for a in arrs],
                                      "sha256": sha256_hex(b"".join(a.tobytes() for a in arrs))}

    def finish(self, *, completed: bool, error: str | None = None, extra: dict | None = None) -> dict:
        with self._lock:
            summary = {"schema_version": SCHEMA_VERSION, "mode": self.mode, "completed": bool(completed),
                       "error": error, "messages": self._seq, "calls": self._call,
                       "stages": self._stage_counts, "aggregates": self._aggregates, "meta": self.meta,
                       "started": self._started, "finished": self.clock(), **(extra or {})}
            if not self._finished:
                self._fh.close()
                self._finished = True
            (self.dir / "capture_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            return summary


# ---------------------------------------------------------------------------------------------
# Synthetic stand-ins for Flower messages (tests and the dry-run demo only)
# ---------------------------------------------------------------------------------------------

class SimpleArray:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)
        self.data = ndarray_to_npy_bytes(self._arr)
        self.dtype, self.shape, self.stype = str(self._arr.dtype), tuple(self._arr.shape), "numpy.ndarray"

    def numpy(self):
        return self._arr


class SimpleContent:
    def __init__(self, arrays=None, metrics=None, configs=None):
        self.array_records = {k: {n: SimpleArray(a) for n, a in v.items()} for k, v in (arrays or {}).items()}
        self.metric_records = {k: dict(v) for k, v in (metrics or {}).items()}
        self.config_records = {k: dict(v) for k, v in (configs or {}).items()}


class _Meta:
    def __init__(self, src, dst, message_type, group_id, message_id="", reply_to=""):
        self.src_node_id, self.dst_node_id = src, dst
        self.message_type, self.group_id = message_type, group_id
        self.message_id, self.reply_to_message_id = message_id, reply_to


class SimpleMessage:
    def __init__(self, content, *, src=0, dst=0, message_type="train", group_id="1", message_id="", reply_to="",
                 error=None):
        self.metadata = _Meta(src, dst, message_type, group_id, message_id, reply_to)
        self._content, self.error = content, error

    @property
    def content(self):
        return self._content

    def has_content(self):
        return self._content is not None

    def has_error(self):
        return self.error is not None


# ---------------------------------------------------------------------------------------------
# Stream parsing (analysis side only)
# ---------------------------------------------------------------------------------------------

def http2_data_by_stream(stream: bytes) -> dict[int, bytes]:
    """Concatenated DATA payloads per HTTP/2 stream id. Stops quietly at anything that is not valid framing."""
    pos = len(H2_PREFACE) if stream.startswith(H2_PREFACE) else 0
    total, out = len(stream), {}
    while pos + 9 <= total:
        length = int.from_bytes(stream[pos:pos + 3], "big")
        ftype, flags = stream[pos + 3], stream[pos + 4]
        sid = int.from_bytes(stream[pos + 5:pos + 9], "big") & 0x7FFFFFFF
        end = pos + 9 + length
        if ftype > 0x0A or end > total:
            break
        if ftype == 0:
            payload = stream[pos + 9:end]
            if flags & 0x08:
                if not payload or payload[0] >= len(payload):
                    break
                payload = payload[1:len(payload) - payload[0]]
            out.setdefault(sid, bytearray()).extend(payload)
        pos = end
    return {sid: bytes(buf) for sid, buf in out.items()}


def grpc_messages(payload: bytes) -> list[bytes]:
    """Split a gRPC stream payload into messages. A compressed message is inflated when possible."""
    pos, out = 0, []
    while pos + 5 <= len(payload):
        flag, length = payload[pos], int.from_bytes(payload[pos + 1:pos + 5], "big")
        if flag not in (0, 1) or pos + 5 + length > len(payload):
            break
        body = payload[pos + 5:pos + 5 + length]
        if flag == 1:
            try:
                body = zlib.decompress(body, 47)
            except zlib.error:
                body = b""
        out.append(body)
        pos += 5 + length
    return out


def _npy_blob_at(view: bytes, i: int) -> bytes | None:
    if i + 10 > len(view):
        return None
    major = view[i + 6]
    if major == 1:
        hlen, hstart = int.from_bytes(view[i + 8:i + 10], "little"), i + 10
    elif major in (2, 3):
        if i + 12 > len(view):
            return None
        hlen, hstart = int.from_bytes(view[i + 8:i + 12], "little"), i + 12
    else:
        return None
    if hlen > 4096 or hstart + hlen > len(view):
        return None
    try:
        header = ast.literal_eval(view[hstart:hstart + hlen].decode("latin1"))
        dtype = np.dtype(header["descr"])
        shape = tuple(int(s) for s in header["shape"])
    except (ValueError, SyntaxError, KeyError, TypeError, UnicodeDecodeError):
        return None
    if dtype.hasobject or any(s < 0 for s in shape):
        return None
    end = hstart + hlen + math.prod(shape) * dtype.itemsize
    return view[i:end] if end <= len(view) else None


def extract_npy_blobs(view: bytes) -> list[bytes]:
    """Every complete numpy-format array found in a byte string, in order of appearance."""
    out, pos = [], 0
    while True:
        i = view.find(NPY_MAGIC, pos)
        if i < 0:
            return out
        blob = _npy_blob_at(view, i)
        if blob is None:
            pos = i + 1
        else:
            out.append(bytes(blob))
            pos = i + len(blob)


class StreamIndex:
    """One captured byte stream with lazily built views, from the raw bytes to gRPC messages.

    `find` reports the lowest layer where a byte string was found. A value that a large frame split
    in two is still found at the HTTP/2 layer, and a compressed message at the gRPC layer.
    """

    LAYERS = ("raw", "http2", "grpc")

    def __init__(self, data: bytes, *, label: str = ""):
        self.data, self.label = bytes(data), label
        self._views: dict[str, list[bytes]] = {"raw": [self.data]}

    def views(self, layer: str) -> list[bytes]:
        if layer not in self._views:
            if layer == "http2":
                self._views[layer] = list(http2_data_by_stream(self.data).values())
            elif layer == "grpc":
                self._views[layer] = [m for p in self.views("http2") for m in grpc_messages(p)]
            else:
                raise ValueError(layer)
        return self._views[layer]

    def find(self, needle: bytes) -> str | None:
        if len(needle) < MIN_OBJECT_BYTES:
            return None
        for layer in self.LAYERS:
            if any(needle in v for v in self.views(layer)):
                return layer
        return None

    def blobs(self) -> dict[str, tuple[str, bytes]]:
        """sha256 of every numpy-format blob in any layer mapped to (layer, blob)."""
        found: dict[str, tuple[str, bytes]] = {}
        # When the dump parses as HTTP/2 the raw layer is skipped, because a frame boundary inside an
        # array would make a blob read from the raw bytes include the frame header.
        layers = ("http2", "grpc") if self.views("http2") else ("raw",)
        for layer in layers:
            for v in self.views(layer):
                for blob in extract_npy_blobs(v):
                    found.setdefault(sha256_hex(blob), (layer, blob))
        return found


def find_array_bytes(stream: StreamIndex, arr: np.ndarray) -> str | None:
    """Search for the raw element bytes of an array (not only its numpy-format blob)."""
    return stream.find(np.ascontiguousarray(arr).tobytes())


# ---------------------------------------------------------------------------------------------
# Relay logs
# ---------------------------------------------------------------------------------------------

def verify_stream_log(data: bytes, log: dict) -> dict:
    """Replay a relay direction log against its dump. Returns which checks held."""
    result = {"bytes_match": len(data) == int(log.get("dump_bytes", len(data))),
              "truncated": bool(log.get("truncated", False)), "chunks_ok": True, "rolling_ok": None}
    events = log.get("chunks") or []
    if not log.get("events_dropped") and not result["truncated"]:
        offset, rolling = 0, hashlib.sha256()
        for _, nbytes, chunk_hash, cumulative in events:
            piece = data[offset:offset + nbytes]
            rolling.update(piece)
            if sha256_hex(piece) != chunk_hash or rolling.hexdigest() != cumulative:
                result["chunks_ok"] = False
                break
            offset += nbytes
        result["chunks_ok"] = result["chunks_ok"] and offset == len(data)
    if not result["truncated"]:
        result["rolling_ok"] = sha256_hex(data) == log.get("sha256")
    return result


def load_relay_dir(directory: str | Path) -> dict[str, list[dict]]:
    """Read every relay log in a directory. Returns relay name to a list of connections.

    A connection is {conn, open, close, c2s, s2c, checks} where c2s is the byte stream from the
    SuperNode to the SuperLink and s2c the other direction.
    """
    base, out = Path(directory), {}
    for logfile in sorted(base.glob("*.connections.jsonl")):
        name = logfile.name[: -len(".connections.jsonl")]
        conns: dict[str, dict] = {}
        for line in logfile.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            conns.setdefault(rec["conn"], {"conn": rec["conn"], "open": None, "close": None})[rec["event"]] = rec
        rows = []
        for conn, row in conns.items():
            close = row["close"]
            if close is None:
                raise WireCaptureError(f"relay {name} connection {conn} has no close record, stop the relay first")
            row["c2s"], row["s2c"], row["checks"] = b"", b"", {}
            for direction in ("c2s", "s2c"):
                file = base / close[direction]["dump_file"]
                data = file.read_bytes() if file.is_file() else b""
                row[direction] = data
                row["checks"][direction] = verify_stream_log(data, close[direction])
            rows.append(row)
        out[name] = rows
    return out


def relay_streams(connections: list[dict], direction: str) -> list[StreamIndex]:
    return [StreamIndex(c[direction], label=f"{c['conn']}:{direction}") for c in connections]


def relay_integrity(relays: dict[str, list[dict]]) -> dict:
    bad, truncated, total = [], [], 0
    for name, conns in relays.items():
        for c in conns:
            for direction, chk in c["checks"].items():
                total += 1
                if not chk["bytes_match"] or not chk["chunks_ok"] or chk["rolling_ok"] is False:
                    bad.append(f"{name}:{c['conn']}:{direction}")
                if chk["truncated"]:
                    truncated.append(f"{name}:{c['conn']}:{direction}")
    return {"streams": total, "inconsistent": bad, "truncated": truncated, "ok": not bad}


def find_in_relays(relays: dict[str, list[dict]], needle: bytes, *, direction: str | None = None,
                   _cache: dict | None = None) -> list[tuple[str, str, str]]:
    """All (relay, direction, layer) where a byte string appears."""
    hits = []
    cache = _cache if _cache is not None else {}
    for name, conns in relays.items():
        for c in conns:
            for d in ((direction,) if direction else ("c2s", "s2c")):
                key = (name, c["conn"], d)
                if key not in cache:
                    cache[key] = StreamIndex(c[d])
                layer = cache[key].find(needle)
                if layer:
                    hits.append((name, d, layer))
    return hits


def blobs_from_relay(conns: list[dict], direction: str) -> dict[str, tuple[str, bytes]]:
    found: dict[str, tuple[str, bytes]] = {}
    for stream in relay_streams(conns, direction):
        for h, v in stream.blobs().items():
            found.setdefault(h, v)
    return found


def model_arrays_from_blobs(blobs: dict[str, tuple[str, bytes]], shapes=MODEL_SHAPES) -> list[np.ndarray]:
    """Pick the float32 model tensors out of a set of numpy blobs by shape. Ambiguity is an error."""
    by_shape: dict[tuple, list[np.ndarray]] = {s: [] for s in shapes}
    for _, blob in blobs.values():
        try:
            arr = npy_bytes_to_ndarray(blob)
        except (ValueError, OSError):
            continue
        if arr.dtype == np.float32 and tuple(arr.shape) in by_shape:
            by_shape[tuple(arr.shape)].append(arr)
    out = []
    for shape in shapes:
        cands = by_shape[shape]
        if len(cands) != 1:
            raise WireCaptureError(f"expected exactly one float32 array of shape {shape} in the stream, found {len(cands)}")
        out.append(cands[0])
    return out


# ---------------------------------------------------------------------------------------------
# Quantisation bound (docs/proofs/one-round-secagg-error.md)
# ---------------------------------------------------------------------------------------------

def quantised_weights(sizes, *, max_weight: float = 1000.0, quant: int = QUANT_RANGE) -> list[int]:
    """The integer weight each client sends, computed the way the SecAgg+ client modifier does."""
    return [round(n / max_weight * quant) for n in sizes]


def secagg_error_bound(sizes, *, max_weight: float = 1000.0, clip: float = CLIP_RANGE, quant: int = QUANT_RANGE,
                       value_range: np.ndarray | None = None) -> dict:
    """Per coordinate bound of the one-round note.

    quantisation = 2 c n / T, float64 server stage = 2^-47 n Q / T, and the weight rounding term
    n (1 + 2^-30) R_j / (2 T) against the exact FedAvg mean (zero when all sizes are equal).
    """
    q = quantised_weights(sizes, max_weight=max_weight, quant=quant)
    n, total = len(sizes), sum(q)
    if n < 2 or total <= 0:
        raise ValueError("need at least two clients with positive weight")
    base = 2.0 * clip * n / total
    server = 2.0 ** -47 * n * quant / total
    if len(set(sizes)) == 1 or value_range is None:
        weight_term = np.zeros(1) if len(set(sizes)) == 1 else None
    else:
        weight_term = n * (1 + 2.0 ** -30) * np.asarray(value_range, float) / (2.0 * total)
    return {"n": n, "T": int(total), "quantisation": base, "server_float64": server,
            "weight_rounding": weight_term, "integer_weights": q}


def weighted_means(updates: list[np.ndarray], sizes) -> tuple[np.ndarray, np.ndarray]:
    """(exact FedAvg mean with the true sample counts, mean with the integer weights SecAgg+ uses)."""
    u = np.stack([np.asarray(v, dtype=np.float64) for v in updates])
    sizes = np.asarray(sizes, dtype=np.float64)
    q = np.asarray(quantised_weights(sizes), dtype=np.float64)
    return (u * sizes[:, None]).sum(0) / sizes.sum(), (u * q[:, None]).sum(0) / q.sum()


def w2_seed_stats(aggregate: np.ndarray, updates: list[np.ndarray], sizes, *, max_weight: float = 1000.0) -> dict:
    """Error of the captured dequantised aggregate against the exact mean, and the bound that applies."""
    mean_true, mean_int = weighted_means(updates, sizes)
    u = np.stack([np.asarray(v, dtype=np.float64) for v in updates])
    rng_j = u.max(0) - u.min(0)
    b = secagg_error_bound(sizes, max_weight=max_weight, value_range=rng_j)
    extra = 0.0 if b["weight_rounding"] is None or not np.any(b["weight_rounding"]) else b["weight_rounding"]
    bound = b["quantisation"] + b["server_float64"] + extra
    err_true = np.abs(np.asarray(aggregate, dtype=np.float64) - mean_true)
    err_int = np.abs(np.asarray(aggregate, dtype=np.float64) - mean_int)
    bound_arr = np.broadcast_to(np.asarray(bound, dtype=np.float64), err_true.shape)
    clip_free = bool(np.all(np.abs(u.astype(np.float32) * np.float32(1.0)) <= CLIP_RANGE))
    return {"max_error": float(err_true.max()), "max_error_vs_integer_weight_mean": float(err_int.max()),
            "bound": float(bound_arr.min()), "bound_max": float(bound_arr.max()),
            "all_coordinates_below_bound": bool(np.all(err_true < bound_arr)),
            "ratio_max": float((err_true / bound_arr).max()), "coordinates": int(err_true.size),
            "T": b["T"], "n": b["n"], "no_clipping": clip_free,
            "quantisation_bound": b["quantisation"], "equal_sizes": len(set(sizes)) == 1}


# ---------------------------------------------------------------------------------------------
# Pre-registered checks (pure functions of numbers, tested with known answers)
# ---------------------------------------------------------------------------------------------

def _wilson(k: int, n: int):
    from feddisclosure.evaluation.intervals import wilson_interval
    return list(wilson_interval(int(k), int(n)))


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in values)


def _failed(check_id: str, prediction: str, rule: str, reason: str) -> dict:
    return {"id": check_id, "prediction": prediction, "pass_rule": rule, "observed": {"error": reason}, "pass": False}


def pooled_rate(rates, images_per_seed: int) -> tuple[int, int, float]:
    r = np.asarray(rates, dtype=float)
    k, n = int(round(float(r.sum()) * images_per_seed)), int(len(r) * images_per_seed)
    return k, n, (k / n if n else float("nan"))


W_TEXT = {
    "W1": ("Inversion on one captured masked vector is at the label floor",
           "Recognisable rate at most floor plus 5 points"),
    "W2": ("The dequantised captured aggregate matches the exact weighted mean inside the one-round bound",
           "Maximum coordinate error below the proven bound for all coordinates in all runs"),
    "W3": ("Attack results on the captured aggregate agree with the simulated results",
           "Difference in recognisable rate within 10 points and in mean MSE within 0.01, over at least 5 seeds"),
    "W4": ("In plaintext mode a passive observer on the relay recovers individual updates",
           "Inversion recognisable rate at least 85%"),
    "W5": ("Per-client sample counts and metrics are visible to the server in clear",
           "Observed in 100% of recorded SecAgg+ replies"),
    "W6": ("The relay stream and the proxy record agree",
           "Every proxied object hash appears in the relay stream in plaintext mode"),
}


def check_w1(attack_rates, floor_rates, attack_mses=None, images_per_seed: int = 8) -> dict:
    pred, rule = W_TEXT["W1"]
    if not len(attack_rates) or len(attack_rates) != len(floor_rates) or not _finite(*attack_rates, *floor_rates):
        return _failed("W1", pred, rule, "no usable attack and floor results")
    k, n, rate = pooled_rate(attack_rates, images_per_seed)
    fk, fn, floor = pooled_rate(floor_rates, images_per_seed)
    return {"id": "W1", "prediction": pred, "pass_rule": rule,
            "observed": {"seeds": len(attack_rates), "attack_rate": rate, "attack_wilson95": _wilson(k, n),
                         "floor_rate": floor, "floor_wilson95": _wilson(fk, fn), "margin": W1_MARGIN,
                         "attack_mean_mse": None if attack_mses is None else float(np.mean(attack_mses))},
            "pass": bool(rate <= floor + W1_MARGIN)}


def check_w2(seed_stats: list[dict]) -> dict:
    pred, rule = W_TEXT["W2"]
    if not seed_stats:
        return _failed("W2", pred, rule, "no runs")
    ok = all(s["all_coordinates_below_bound"] for s in seed_stats)
    worst = max(seed_stats, key=lambda s: s["ratio_max"])
    return {"id": "W2", "prediction": pred, "pass_rule": rule,
            "observed": {"runs": len(seed_stats), "worst_max_error": worst["max_error"], "bound": worst["bound"],
                         "worst_error_over_bound": worst["ratio_max"],
                         "runs_all_below_bound": sum(bool(s["all_coordinates_below_bound"]) for s in seed_stats),
                         "coordinates_per_run": worst["coordinates"], "T": worst["T"], "n": worst["n"],
                         "no_clipping_in_all_runs": all(s["no_clipping"] for s in seed_stats)},
            "pass": bool(ok)}


def check_w3(captured: list[dict], simulated: list[dict], images_per_seed: int = 8) -> dict:
    """Each entry holds {"rate": ..., "mse": ...} for one seed, in the same seed order."""
    pred, rule = W_TEXT["W3"]
    if len(captured) != len(simulated) or not captured:
        return _failed("W3", pred, rule, "captured and simulated runs do not pair up")
    if not _finite(*[d["rate"] for d in captured + simulated], *[d["mse"] for d in captured + simulated]):
        return _failed("W3", pred, rule, "non finite attack result")
    ck, cn, crate = pooled_rate([d["rate"] for d in captured], images_per_seed)
    sk, sn, srate = pooled_rate([d["rate"] for d in simulated], images_per_seed)
    cmse, smse = float(np.mean([d["mse"] for d in captured])), float(np.mean([d["mse"] for d in simulated]))
    drate, dmse = abs(crate - srate), abs(cmse - smse)
    enough = len(captured) >= W3_MIN_SEEDS
    return {"id": "W3", "prediction": pred, "pass_rule": rule,
            "observed": {"seeds": len(captured), "min_seeds": W3_MIN_SEEDS,
                         "captured_rate": crate, "captured_wilson95": _wilson(ck, cn),
                         "simulated_rate": srate, "simulated_wilson95": _wilson(sk, sn),
                         "captured_mean_mse": cmse, "simulated_mean_mse": smse,
                         "abs_rate_difference": drate, "abs_mse_difference": dmse},
            "pass": bool(enough and drate <= W3_RATE_TOLERANCE and dmse <= W3_MSE_TOLERANCE)}


def check_w4(rates, relay_matches_proxy: int | None = None, clients_checked: int | None = None,
             images_per_seed: int = 8) -> dict:
    pred, rule = W_TEXT["W4"]
    if not len(rates) or not _finite(*rates):
        return _failed("W4", pred, rule, "no usable plaintext attack results")
    k, n, rate = pooled_rate(rates, images_per_seed)
    return {"id": "W4", "prediction": pred, "pass_rule": rule,
            "observed": {"seeds": len(rates), "recognisable_rate": rate, "wilson95": _wilson(k, n),
                         "threshold": W4_MIN_RATE, "relay_updates_equal_proxy": relay_matches_proxy,
                         "relay_updates_checked": clients_checked},
            "pass": bool(rate >= W4_MIN_RATE)}


def check_w5(replies: list[dict], expected_counts: dict | None = None) -> dict:
    """replies: one dict per fit-stage SecAgg+ reply with num_examples, num_examples_visible, metrics_visible."""
    pred, rule = W_TEXT["W5"]
    if not replies:
        return _failed("W5", pred, rule, "no SecAgg+ replies were recorded")
    good = 0
    for r in replies:
        ok = bool(r.get("num_examples_visible")) and bool(r.get("metrics_visible"))
        want = None if expected_counts is None else expected_counts.get(r.get("node_id"))
        if want is not None:
            ok = ok and r.get("num_examples") == want
        good += int(ok)
    return {"id": "W5", "prediction": pred, "pass_rule": rule,
            "observed": {"replies": len(replies), "visible_in_clear": good, "fraction": good / len(replies),
                         "metric_keys_seen": sorted({k for r in replies for k in (r.get("metrics") or {})})},
            "pass": bool(good == len(replies))}


def check_w6(found_flags: list[bool], methods: dict | None = None, integrity_ok: bool = True) -> dict:
    pred, rule = W_TEXT["W6"]
    if not found_flags:
        return _failed("W6", pred, rule, "no proxied objects were recorded")
    n_found = int(sum(bool(f) for f in found_flags))
    return {"id": "W6", "prediction": pred, "pass_rule": rule,
            "observed": {"objects": len(found_flags), "found_in_relay": n_found, "methods": methods or {},
                         "relay_logs_consistent": bool(integrity_ok)},
            "pass": bool(n_found == len(found_flags) and integrity_ok)}


# ---------------------------------------------------------------------------------------------
# Data selection for a capture run
# ---------------------------------------------------------------------------------------------

def prepare_view_partitions(output_dir: str | Path, seed: int, *, clients: int = 4, records_per_client: int = 8,
                            val_per_client: int = 1, test_records: int = 16, loader=None) -> dict:
    """Write the simulation's data selection (view_seed.select_view_records) in the Flower deployment layout.

    Client i holds records_per_client public digits. The public pool supplies a tiny validation file per
    client and an evaluator-only test file, because the deployment helpers expect both. The manifest
    has the same schema as prepare_partitions so scripts/prepare_flower_deployment.py accepts it.
    """
    from feddisclosure.attacks.view_seed import select_view_records
    if loader is None:
        from feddisclosure.training.datasets import _source
        loader = lambda root: _source("sklearn_digits", root)  # noqa: E731
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing partition dataset: {out}")
    out.mkdir(parents=True, exist_ok=True)
    x, y, provenance = loader(out)
    sel = select_view_records(len(y), clients, records_per_client, seed)
    pool = sel.pool_idx
    if len(pool) < clients * val_per_client + test_records:
        raise ValueError("public pool too small")
    file_hash = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    manifest = {"schema_version": 2, "source": "sklearn_digits", "provenance": provenance, "seed": int(seed),
                "partition": "iid", "clients": clients, "source_limit": None, "num_features": int(x.shape[1]),
                "num_classes": 10, "dirichlet_alpha": None, "allocation_method": "view_seed_selection",
                "client_partitions": [], "evaluator_test": {},
                "raw_data_policy": "only client-owned NPZ files in training. Evaluator files are separate"}
    for i in range(clients):
        tr = sel.idx[i * records_per_client:(i + 1) * records_per_client]
        va = pool[i * val_per_client:(i + 1) * val_per_client]
        site = out / f"institution-{i:02d}"
        site.mkdir()
        np.savez_compressed(site / "train.npz", x=x[tr], y=y[tr], source_idx=tr)
        np.savez_compressed(site / "val.npz", x=x[va], y=y[va], source_idx=va)
        manifest["client_partitions"].append({
            "client_id": i, "file": f"institution-{i:02d}/train.npz", "val_file": f"institution-{i:02d}/val.npz",
            "samples": len(tr), "val_samples": len(va),
            "sha256": file_hash(site / "train.npz"), "val_sha256": file_hash(site / "val.npz")})
    test_idx = pool[clients * val_per_client:clients * val_per_client + test_records]
    (out / "evaluator_only").mkdir()
    np.savez_compressed(out / "evaluator_only" / "test.npz", x=x[test_idx], y=y[test_idx], source_idx=test_idx)
    manifest["evaluator_test"] = {"file": "evaluator_only/test.npz", "samples": len(test_idx),
                                  "sha256": file_hash(out / "evaluator_only" / "test.npz")}
    (out / "evaluator_only" / "view_selection.json").write_text(json.dumps(
        {"seed": int(seed), "records_per_client": records_per_client, "clients": clients,
         "selection_function": "feddisclosure.attacks.view_seed.select_view_records",
         "client_source_idx": [sel.idx[i * records_per_client:(i + 1) * records_per_client].tolist()
                               for i in range(clients)]}, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_config_dict(seed: int, *, clients: int = 4, learning_rate: float = 0.15, batch_size: int = 8) -> dict:
    """Training configuration of a capture run. One round, one local epoch over one full batch."""
    return {"id": f"wire-seed{seed}", "seed": int(seed), "dataset": "sklearn_digits", "partition": "iid",
            "clients": clients, "rounds": 1, "local_epochs": 1, "batch_size": batch_size,
            "learning_rate": learning_rate, "source_limit": None, "observer": "wire_capture",
            "allowed_observations": ["masked_vectors", "sample_counts", "relay_byte_streams"]}
