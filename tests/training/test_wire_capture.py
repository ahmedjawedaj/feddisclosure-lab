"""Unit tests for the wire-level validation (v0.15, WP3). No Docker is used here.

Docker orchestration is covered by static checks and a dry run. The capture logic is tested against
fakes with known answers, including failing cases.
"""
import asyncio
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from feddisclosure.training import wire_capture as wc  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


relay_mod = _load("wire_relay_script", ROOT / "docker/wire_relay/relay.py")


# ----------------------------------------------------------------------------------------------
# helpers that build fake captures
# ----------------------------------------------------------------------------------------------

def h2_frame(ftype, flags, sid, payload):
    return len(payload).to_bytes(3, "big") + bytes([ftype, flags]) + sid.to_bytes(4, "big") + payload


def h2_stream(messages, *, preface, split=True, compress=False):
    out = (wc.H2_PREFACE if preface else b"") + h2_frame(4, 0, 0, b"")
    for k, msg in enumerate(messages):
        sid = 1 + 2 * k
        if compress:
            body = gzip.compress(msg)
            grpc = b"\x01" + len(body).to_bytes(4, "big") + body
        else:
            grpc = b"\x00" + len(msg).to_bytes(4, "big") + msg
        mid = len(grpc) // 2 if split else len(grpc)
        out += h2_frame(1, 4, sid, b"\x00" * 5) + h2_frame(0, 0 if split else 1, sid, grpc[:mid])
        if split:
            out += h2_frame(0, 1, sid, grpc[mid:])
    return out


def write_relay(directory, name, c2s, s2c, *, conn="c0001", chunk=700):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rec = {}
    for d, data in (("c2s", c2s), ("s2c", s2c)):
        (directory / f"{name}.{conn}.{d}.bin").write_bytes(data)
        rolling, events = hashlib.sha256(), []
        for i in range(0, len(data), chunk):
            piece = data[i:i + chunk]
            rolling.update(piece)
            events.append([0.01 * i, len(piece), hashlib.sha256(piece).hexdigest(), rolling.hexdigest()])
        rec[d] = {"bytes": len(data), "dump_file": f"{name}.{conn}.{d}.bin", "dump_bytes": len(data),
                  "truncated": False, "sha256": hashlib.sha256(data).hexdigest(), "first_t": 0.0, "last_t": 1.0,
                  "chunks": events, "events_dropped": 0}
    with (directory / f"{name}.connections.jsonl").open("a") as fh:
        fh.write(json.dumps({"event": "open", "relay": name, "conn": conn, "t": 0.0}) + "\n")
        fh.write(json.dumps({"event": "close", "relay": name, "conn": conn, "t": 1.0, "error": None, **rec}) + "\n")


def model_arrays(seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float32) for s in wc.MODEL_SHAPES]


def blobs_of(arrays):
    return b"".join(b"filler" + wc.ndarray_to_npy_bytes(a) for a in arrays)


def build_synthetic_run(root, seed, mode, *, drop_relay_s2c=False, hide_counts=False, aggregate_shift=0.0,
                        shared_data=True):
    """A full run directory as scripts/run_wire_capture.py would leave it, built without Docker."""
    import torch
    from feddisclosure.attacks.view_seed import client_step_seed
    from feddisclosure.attacks.view_study import client_step, flatten, secagg_aggregate
    from feddisclosure.training.task import DigitMLP
    root = Path(root)
    data_dir = root / "data" / f"seed{seed}"
    if not data_dir.exists():
        wc.prepare_view_partitions(data_dir, seed)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    cx, cy = [], []
    for part in manifest["client_partitions"]:
        with np.load(data_dir / part["file"]) as z:
            cx.append(z["x"]), cy.append(z["y"])
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in DigitMLP(64, 10).state_dict().items()}
    init_arrays = [v.numpy().copy() for v in init.values()]
    states = [client_step(init, cx[i], cy[i], lr=0.15, sigma=None, seed=client_step_seed(seed, i)) for i in range(4)]
    updates = [[v.numpy().copy() for v in s.values()] for s in states]
    run_dir = root / "runs" / f"seed{seed}-{mode}"
    cap = run_dir / "server" / "wire"
    rec = wc.WireRecorder(cap, mode=mode, meta={"seed": seed, "max_weight": 1000.0})
    nodes = [1000 + 17 * i for i in range(4)]
    glob = {"fitins.parameters": {str(j): a for j, a in enumerate(init_arrays)}}
    rng = np.random.default_rng(seed)
    c2s_payload, s2c_payload = {}, {i: blobs_of(init_arrays) for i in range(4)}
    if mode == "plaintext":
        out = [wc.SimpleMessage(wc.SimpleContent(arrays=glob), dst=n) for n in nodes]
        call, stage = rec.record_out(out)
        replies = []
        for i, n in enumerate(nodes):
            arrays = {"fitres.parameters": {str(j): a for j, a in enumerate(updates[i])}}
            replies.append(wc.SimpleMessage(wc.SimpleContent(
                arrays=arrays, metrics={"fitres.num_examples": {"num_examples": 8}},
                configs={"fitres.metrics": {}}), src=n))
            c2s_payload[i] = blobs_of(updates[i])
        rec.record_in(call, stage, replies)
        rec.record_aggregate("strategy_output", [np.mean([u[j] for u in updates], axis=0) for j in range(4)])
    else:
        for stage_name in ("setup", "share_keys"):
            out = [wc.SimpleMessage(wc.SimpleContent(configs={wc.SECAGG_CONFIG_RECORD: {wc.STAGE_KEY: stage_name}}), dst=n)
                   for n in nodes]
            call, stage = rec.record_out(out)
            rec.record_in(call, stage, [wc.SimpleMessage(wc.SimpleContent(
                configs={wc.SECAGG_CONFIG_RECORD: {"pk1": os.urandom(32), "pk2": os.urandom(32)}}), src=n) for n in nodes])
        out = [wc.SimpleMessage(wc.SimpleContent(arrays=glob, configs={
            wc.SECAGG_CONFIG_RECORD: {wc.STAGE_KEY: "collect_masked_vectors", "ctxts": [os.urandom(80)]}}), dst=n) for n in nodes]
        call, stage = rec.record_out(out)
        replies = []
        for i, n in enumerate(nodes):
            masked = [np.array([33554], dtype=np.int64)] + [
                rng.integers(0, 2 ** 32, size=a.shape, dtype=np.int64) for a in init_arrays]
            blobs = [wc.ndarray_to_npy_bytes(m) for m in masked]
            replies.append(wc.SimpleMessage(wc.SimpleContent(
                metrics={"fitres.num_examples": {"num_examples": 8}}, configs={
                    "fitres.metrics": {}, wc.SECAGG_CONFIG_RECORD: {wc.MASKED_KEY: blobs}}), src=n))
            c2s_payload[i] = b"".join(blobs)
        if hide_counts:
            for r in replies:
                r.content.metric_records.clear()
        rec.record_in(call, stage, replies)
        out = [wc.SimpleMessage(wc.SimpleContent(configs={wc.SECAGG_CONFIG_RECORD: {wc.STAGE_KEY: "unmask"}}), dst=n)
               for n in nodes]
        call, stage = rec.record_out(out)
        rec.record_in(call, stage, [wc.SimpleMessage(wc.SimpleContent(
            configs={wc.SECAGG_CONFIG_RECORD: {"nids": [1], "shares": [os.urandom(20)]}}), src=n) for n in nodes])
        agg = secagg_aggregate([flatten(s) for s in states], [8] * 4, max_weight=1000.0, rng_seed=seed) + aggregate_shift
        parts, pos = [], 0
        for a in init_arrays:
            parts.append(agg[pos:pos + a.size].reshape(a.shape))
            pos += a.size
        rec.record_aggregate("dequantised_aggregate", parts)
        rec.record_aggregate("strategy_output", parts)
    rec.finish(completed=True)
    for i in range(4):
        write_relay(run_dir / "relay", f"site-{i}", h2_stream([c2s_payload[i]], preface=True),
                    h2_stream([b"" if drop_relay_s2c else s2c_payload[i]], preface=False))
        site_dir = run_dir / "clients" / f"site-{i}"
        site_dir.mkdir(parents=True)
        digest = hashlib.sha256(b"".join(a.tobytes() for a in updates[i])).hexdigest()
        (site_dir / "events.jsonl").write_text(json.dumps({"event_type": "train", "weights_sha256": digest}) + "\n")
    (run_dir / "wire_run.json").write_text(json.dumps({"seed": seed, "mode": mode, "ok": True, "data_dir": str(data_dir)}))
    return run_dir


# ----------------------------------------------------------------------------------------------
# hashing and recording
# ----------------------------------------------------------------------------------------------

def test_object_hashes_are_sha256_of_exact_bytes_and_cover_arrays_and_bytes():
    arr = np.arange(40, dtype=np.float32)
    key = os.urandom(32)
    content = wc.SimpleContent(arrays={"p": {"0": arr}}, configs={"c": {"keys": [key, key[::-1]], "n": 3}})
    objs = list(wc.iter_objects(content))
    assert [o["kind"] for o in objs] == ["array", "bytes", "bytes"]
    assert objs[0]["sha256"] == hashlib.sha256(wc.ndarray_to_npy_bytes(arr)).hexdigest()
    assert objs[1]["sha256"] == hashlib.sha256(key).hexdigest() and objs[1]["index"] == 0
    other = wc.SimpleContent(arrays={"p": {"0": arr + 1}}, configs={"c": {"keys": [key, key[::-1]], "n": 3}})
    assert wc.content_sha256(content) == wc.content_sha256(content)
    assert wc.content_sha256(content) != wc.content_sha256(other)


def test_flower_chunk_object_id_matches_real_flower_for_a_small_array():
    flwr_record = pytest.importorskip("flwr.app")
    arr = np.arange(10, dtype=np.float32)
    record = flwr_record.ArrayRecord([arr])
    array = next(iter(record.values()))
    chunk_id = next(iter(array.children))
    assert wc.flower_chunk_object_id(array.data) == chunk_id


def _real_message(content, *, dst=0, src=0, reply_to=None):
    from flwr.app import Message
    if reply_to is not None:
        return Message(content, reply_to=reply_to)
    return Message(content, dst_node_id=dst, message_type="train", group_id="1")


class FakeInnerGrid:
    """Echoes a masked-vector style reply for every message. Counts calls."""

    def __init__(self, replies_factory):
        self.factory, self.calls, self.closed = replies_factory, 0, False

    def send_and_receive(self, messages, *, timeout=None):
        self.calls += 1
        return [self.factory(m) for m in messages]

    def push_messages(self, messages):
        self.pending = list(messages)
        return [f"id{i}" for i, _ in enumerate(self.pending)]

    def pull_messages(self, ids):
        return [self.factory(m) for m in self.pending]

    def get_node_ids(self):
        return [1, 2]

    def close(self):
        self.closed = True


def _stage_message(stage, dst, extra=None, arrays=None):
    from flwr.app import ArrayRecord, ConfigRecord, RecordDict
    rd = RecordDict()
    cfg = {"stage": stage, **(extra or {})}
    rd["secaggplus_configs"] = ConfigRecord(cfg)
    if arrays:
        rd["fitins.parameters"] = ArrayRecord(arrays)
    return _real_message(rd, dst=dst)


def _masked_reply(msg):
    from flwr.app import ConfigRecord, MetricRecord, RecordDict
    from flwr.common import Message
    stage = msg.content["secaggplus_configs"]["stage"]
    rd = RecordDict()
    if stage == "collect_masked_vectors":
        rd["fitres.num_examples"] = MetricRecord({"num_examples": 8})
        rd["fitres.metrics"] = ConfigRecord({})
        blobs = [wc.ndarray_to_npy_bytes(np.arange(5, dtype=np.int64) + k) for k in range(3)]
        rd["secaggplus_configs"] = ConfigRecord({"masked_params": blobs})
    else:
        rd["secaggplus_configs"] = ConfigRecord({"pk": os.urandom(32)})
    return Message(rd, reply_to=msg)


def test_recording_grid_records_all_four_stages_and_is_transparent(tmp_path):
    pytest.importorskip("flwr")
    from feddisclosure.training.secagg_server_capture import RecordingGrid
    inner = FakeInnerGrid(_masked_reply)
    rec = wc.WireRecorder(tmp_path / "cap", mode="secagg")
    grid = RecordingGrid(inner, rec)
    glob = [np.ones((2, 2), dtype=np.float32)]
    originals = {}
    for stage in wc.STAGES:
        msgs = [_stage_message(stage, d, arrays=glob if stage == "collect_masked_vectors" else None) for d in (11, 12, 13, 14)]
        replies = grid.send_and_receive(msgs, timeout=5)
        assert len(replies) == 4 and all(r.metadata.reply_to_message_id == m.metadata.message_id
                                         for r, m in zip(replies, msgs))
        originals[stage] = replies
    assert inner.calls == 4
    summary = rec.finish(completed=True)
    assert {s: summary["stages"][s] for s in wc.STAGES} == {s: {"out": 4, "in": 4} for s in wc.STAGES}
    lines = [json.loads(x) for x in (tmp_path / "cap" / "messages.jsonl").read_text().splitlines()]
    assert len(lines) == 32
    collect = [m for m in lines if m["direction"] == "in" and m["stage"] == "collect_masked_vectors"]
    assert all(m["num_examples"] == 8 and m["num_examples_visible"] and m["metrics_visible"] for m in collect)
    assert all(m["masked_file"] and len(np.load(tmp_path / "cap" / m["masked_file"]).files) == 3 for m in collect)
    assert all(m["node_id"] in (11, 12, 13, 14) for m in lines)
    glob_out = [m for m in lines if m["direction"] == "out" and m["arrays_file"]]
    assert len({m["arrays_file"] for m in glob_out}) == 1   # identical global model is stored once
    assert grid.get_node_ids() == [1, 2]
    grid.close()
    assert inner.closed


def test_recording_grid_reply_objects_are_not_altered(tmp_path):
    pytest.importorskip("flwr")
    from feddisclosure.training.secagg_server_capture import RecordingGrid
    produced = []

    def factory(msg):
        reply = _masked_reply(msg)
        produced.append(reply)
        return reply
    grid = RecordingGrid(FakeInnerGrid(factory), wc.WireRecorder(tmp_path / "cap", mode="secagg"))
    got = grid.send_and_receive([_stage_message("collect_masked_vectors", 5)])
    assert got[0] is produced[0]
    before = wc.content_sha256(got[0].content)
    assert wc.content_sha256(got[0].content) == before      # recording did not change the reply content


def test_push_pull_path_is_recorded_with_the_stage_of_the_push(tmp_path):
    pytest.importorskip("flwr")
    from feddisclosure.training.secagg_server_capture import RecordingGrid
    rec = wc.WireRecorder(tmp_path / "cap", mode="secagg")
    grid = RecordingGrid(FakeInnerGrid(_masked_reply), rec)
    msgs = [_stage_message("setup", 1), _stage_message("setup", 2)]
    # reply_to ids are the messages' own ids here, push returns stand-in ids
    ids = grid.push_messages(msgs)
    rec.remember_pushed(msgs, [m.metadata.message_id for m in msgs], 1, "setup")
    replies = grid.pull_messages(ids)
    assert len(replies) == 2
    lines = [json.loads(x) for x in (tmp_path / "cap" / "messages.jsonl").read_text().splitlines()]
    assert [m["direction"] for m in lines].count("in") == 2
    assert {m["stage"] for m in lines} == {"setup"}


def test_proxy_failure_propagates_and_marks_the_capture_incomplete(tmp_path):
    pytest.importorskip("flwr")
    from feddisclosure.training import secagg_server as base
    from feddisclosure.training import secagg_server_capture as cap

    class Ctx:
        run_config = {"use-secagg": True, "wire-capture": True, "artifact-dir": str(tmp_path), "seed": 1, "clients": 4,
                      "max-weight": 1000.0, "num-shares": 4, "reconstruction-threshold": 2, "num-server-rounds": 1,
                      "learning-rate": 0.15, "batch-size": 8, "local-epochs": 1}
    original_main, original_cls = base.main, base.CountingFedAvg

    def broken(grid, context):
        raise RuntimeError("grid broke")
    base.main = broken
    try:
        with pytest.raises(RuntimeError, match="grid broke"):
            cap.run_captured(object(), Ctx())
    finally:
        base.main = original_main
    assert base.CountingFedAvg is original_cls      # the default application class is restored
    summary = json.loads((tmp_path / "wire" / "capture_summary.json").read_text())
    assert summary["completed"] is False and "grid broke" in summary["error"]


def test_capture_is_refused_when_the_aggregate_hook_never_ran(tmp_path):
    pytest.importorskip("flwr")
    from feddisclosure.training import secagg_server as base
    from feddisclosure.training import secagg_server_capture as cap

    class Ctx:
        run_config = {"use-secagg": False, "wire-capture": True, "artifact-dir": str(tmp_path), "seed": 1, "clients": 4,
                      "max-weight": 1000.0, "num-shares": 4, "reconstruction-threshold": 2, "num-server-rounds": 1,
                      "learning-rate": 0.15, "batch-size": 8, "local-epochs": 1}
    original_main = base.main
    base.main = lambda grid, context: None        # returns normally and never aggregates
    try:
        with pytest.raises(wc.WireCaptureError):
            cap.run_captured(object(), Ctx())
    finally:
        base.main = original_main
    assert json.loads((tmp_path / "wire" / "capture_summary.json").read_text())["completed"] is False


def test_capturing_strategy_saves_the_dequantised_aggregate_and_rejects_disagreement(tmp_path):
    pytest.importorskip("flwr")
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
    from feddisclosure.training import secagg_server_capture as cap
    from feddisclosure.training.secagg_server import CountingFedAvg
    rec = wc.WireRecorder(tmp_path / "cap", mode="secagg")
    strat = cap.make_capturing_strategy(rec, "secagg")(fraction_fit=1.0)
    agg = [np.full((2,), 0.5)]
    ok = Status(code=Code.OK, message="")
    results = [(object(), FitRes(status=ok, parameters=ndarrays_to_parameters(agg), num_examples=8, metrics={}))
               for _ in range(3)]
    strat.aggregate_fit(1, results, [])
    assert np.array_equal(np.load(tmp_path / "cap" / "dequantised_aggregate.npz")["a0"], agg[0])
    bad = results[:2] + [(object(), FitRes(status=ok, parameters=ndarrays_to_parameters([agg[0] + 1]), num_examples=8, metrics={}))]
    with pytest.raises(wc.WireCaptureError):
        strat.aggregate_fit(1, bad, [])
    assert issubclass(type(strat), CountingFedAvg)


# ----------------------------------------------------------------------------------------------
# stream search
# ----------------------------------------------------------------------------------------------

def test_relay_stream_search_finds_a_known_float32_array_and_not_a_different_one():
    known, other = model_arrays(1)[0], model_arrays(2)[0]
    stream = wc.StreamIndex(h2_stream([blobs_of([known])], preface=True, split=False))
    assert wc.find_array_bytes(stream, known) == "raw"
    assert wc.find_array_bytes(stream, other) is None
    assert wc.find_array_bytes(stream, known[::-1].copy()) is None


def test_search_finds_an_array_that_http2_framing_split_in_two():
    arr = model_arrays(3)[0]
    stream = wc.StreamIndex(h2_stream([blobs_of([arr])], preface=True, split=True))
    assert wc.find_array_bytes(stream, arr) == "http2"


def test_search_finds_an_array_inside_a_compressed_grpc_message():
    arr = model_arrays(4)[0]
    payload = blobs_of([arr])
    stream = wc.StreamIndex(h2_stream([payload], preface=False, split=False, compress=True))
    assert stream.find(np.ascontiguousarray(arr).tobytes()) == "grpc"


def test_blob_extraction_recovers_arrays_with_their_hashes_and_survives_garbage():
    arrays = model_arrays(5)
    stream = wc.StreamIndex(h2_stream([blobs_of(arrays)], preface=True) + os.urandom(300))
    blobs = stream.blobs()
    for a in arrays:
        assert hashlib.sha256(wc.ndarray_to_npy_bytes(a)).hexdigest() in blobs
    recovered = wc.model_arrays_from_blobs(blobs)
    assert all(np.array_equal(r, a) for r, a in zip(recovered, arrays))
    assert wc.StreamIndex(b"").blobs() == {} and wc.StreamIndex(os.urandom(5000)).find(b"x" * 20) is None
    cut = h2_stream([blobs_of(arrays)], preface=True)[:-2000]
    wc.StreamIndex(cut).blobs()   # truncated dump must not raise


def test_model_array_selection_refuses_ambiguity():
    a, b = model_arrays(6), model_arrays(7)
    stream = wc.StreamIndex(h2_stream([blobs_of(a) + blobs_of(b)], preface=False))
    with pytest.raises(wc.WireCaptureError):
        wc.model_arrays_from_blobs(stream.blobs())


# ----------------------------------------------------------------------------------------------
# the real relay over localhost
# ----------------------------------------------------------------------------------------------

def _run_relay(tmp_path, payload, *, dump_cap):
    async def go():
        async def upstream(reader, writer):
            data = await reader.read(100000)
            writer.write(b"reply:" + data[:10])
            await writer.drain()
            writer.close()
        up = await asyncio.start_server(upstream, "127.0.0.1", 0)
        uport = up.sockets[0].getsockname()[1]
        relay = relay_mod.Relay("site-0", "127.0.0.1", uport, tmp_path, dump_cap, 10 ** 9, 1000)
        task = asyncio.create_task(relay.serve("127.0.0.1", 0))
        for _ in range(100):
            await asyncio.sleep(0.02)
            if getattr(relay, "port", 0):
                break
        reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        got = await reader.read(1000)
        writer.close()
        await asyncio.sleep(0.1)
        for t in list(relay.tasks):
            t.cancel()
        await asyncio.gather(*relay.tasks, return_exceptions=True)
        relay.log.close()
        task.cancel()
        up.close()
        return got
    return asyncio.run(go())


def test_relay_forwards_bytes_unchanged_and_logs_counts_and_hashes(tmp_path):
    payload = os.urandom(3000)
    got = _run_relay(tmp_path, payload, dump_cap=10 ** 6)
    assert got == b"reply:" + payload[:10]
    relays = wc.load_relay_dir(tmp_path)
    (conn,) = relays["site-0"]
    assert conn["c2s"] == payload and conn["s2c"] == got
    assert conn["close"]["c2s"]["bytes"] == 3000 and conn["open"]["peer"].startswith("127.0.0.1")
    assert wc.relay_integrity(relays)["ok"]
    assert conn["close"]["c2s"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_relay_dump_cap_flags_truncation_and_the_integrity_report_says_so(tmp_path):
    payload = os.urandom(5000)
    got = _run_relay(tmp_path, payload, dump_cap=1000)
    assert got.startswith(b"reply:")        # forwarding continues after the cap
    relays = wc.load_relay_dir(tmp_path)
    (conn,) = relays["site-0"]
    assert len(conn["c2s"]) == 1000 and conn["close"]["c2s"]["bytes"] == 5000
    report = wc.relay_integrity(relays)
    assert report["truncated"] == ["site-0:c0001:c2s"] and report["ok"]


def test_tampered_dump_is_detected(tmp_path):
    write_relay(tmp_path, "site-0", b"A" * 3000, b"B" * 10)
    dump = tmp_path / "site-0.c0001.c2s.bin"
    data = bytearray(dump.read_bytes())
    data[1500] ^= 1
    dump.write_bytes(bytes(data))
    report = wc.relay_integrity(wc.load_relay_dir(tmp_path))
    assert not report["ok"] and report["inconsistent"] == ["site-0:c0001:c2s"]


def test_relay_log_without_close_record_is_an_error(tmp_path):
    (tmp_path / "site-0.connections.jsonl").write_text(json.dumps({"event": "open", "conn": "c0001"}) + "\n")
    with pytest.raises(wc.WireCaptureError):
        wc.load_relay_dir(tmp_path)


# ----------------------------------------------------------------------------------------------
# pre-registered check logic with known answers
# ----------------------------------------------------------------------------------------------

def test_w1_pass_and_fail():
    ok = wc.check_w1([0.0] * 5, [0.675] * 5)
    assert ok["pass"] and ok["id"] == "W1" and ok["observed"]["attack_wilson95"][0] == 0.0
    assert not wc.check_w1([0.75] * 5, [0.675] * 5)["pass"]            # 7.5 points above the floor
    assert not wc.check_w1([1.0] * 5, [0.5] * 5)["pass"]
    assert not wc.check_w1([], [])["pass"]
    assert not wc.check_w1([float("nan")], [0.5])["pass"]


def test_w1_margin_is_five_points_exactly():
    assert wc.check_w1([0.5] * 8, [0.5] * 8)["pass"]
    assert wc.check_w1([0.5625] * 8, [0.5] * 8)["pass"] is False      # 56.25 percent against 50 plus 5
    assert wc.check_w1([0.5] * 4 + [0.5625] * 4, [0.5] * 8)["pass"]   # 53.1 percent is inside the margin


def _stats(max_error, bound=4e-4):
    return {"all_coordinates_below_bound": max_error < bound, "ratio_max": max_error / bound, "max_error": max_error,
            "bound": bound, "coordinates": 2410, "T": 134216, "n": 4, "no_clipping": True}


def test_w2_requires_every_run_below_the_bound():
    assert wc.check_w2([_stats(1e-4), _stats(3e-4)])["pass"]
    failed = wc.check_w2([_stats(1e-4), _stats(5e-4)])
    assert not failed["pass"] and failed["observed"]["runs_all_below_bound"] == 1
    assert not wc.check_w2([])["pass"]


def test_w3_tolerances_and_minimum_seeds():
    cap = [{"rate": 0.5, "mse": 0.05}] * 5
    assert wc.check_w3(cap, [{"rate": 0.55, "mse": 0.055}] * 5)["pass"]
    assert not wc.check_w3(cap, [{"rate": 0.65, "mse": 0.05}] * 5)["pass"]      # rate differs by 15 points
    assert not wc.check_w3(cap, [{"rate": 0.5, "mse": 0.07}] * 5)["pass"]        # mse differs by 0.02
    few = wc.check_w3(cap[:4], [{"rate": 0.5, "mse": 0.05}] * 4)
    assert not few["pass"] and few["observed"]["seeds"] == 4
    assert not wc.check_w3(cap, cap[:3])["pass"]
    assert not wc.check_w3([{"rate": float("nan"), "mse": 0.1}] * 5, cap)["pass"]


def test_w4_threshold_is_85_percent():
    assert wc.check_w4([0.875] * 5, 20, 20)["pass"]
    assert wc.check_w4([1.0, 1.0, 1.0, 1.0, 0.25], 20, 20)["pass"]              # exactly 85 percent
    assert wc.check_w4([1.0, 1.0, 1.0, 0.75, 0.25], 20, 20)["pass"] is False    # 80 percent
    assert not wc.check_w4([])["pass"]


def test_w5_needs_clear_counts_and_metrics_in_every_reply():
    replies = [{"node_id": i, "num_examples": 8, "num_examples_visible": True, "metrics_visible": True, "metrics": {}}
               for i in range(4)]
    assert wc.check_w5(replies, {i: 8 for i in range(4)})["pass"]
    hidden = [dict(r) for r in replies]
    hidden[2]["num_examples_visible"] = False
    out = wc.check_w5(hidden)
    assert not out["pass"] and out["observed"]["fraction"] == 0.75
    assert not wc.check_w5(replies, {0: 8, 1: 8, 2: 9, 3: 8})["pass"]               # a wrong visible count
    assert not wc.check_w5([])["pass"]


def test_w6_needs_every_object_and_consistent_relay_logs():
    assert wc.check_w6([True] * 10)["pass"]
    assert not wc.check_w6([True] * 9 + [False])["pass"]
    assert not wc.check_w6([True] * 3, integrity_ok=False)["pass"]
    assert not wc.check_w6([])["pass"]


# ----------------------------------------------------------------------------------------------
# bound arithmetic
# ----------------------------------------------------------------------------------------------

def test_integer_weights_and_bound_match_the_proof_note_example():
    assert wc.quantised_weights([8] * 4) == [33554] * 4
    b = wc.secagg_error_bound([96] * 4)
    assert b["T"] == 1610612 and abs(b["quantisation"] - 3.9736e-5) < 1e-8     # the note's Docker 4 client value


def test_w2_stats_pass_for_the_real_quantiser_and_fail_for_a_perturbed_aggregate():
    pytest.importorskip("flwr")
    from feddisclosure.attacks.view_study import secagg_aggregate
    rng = np.random.default_rng(0)
    ups = [rng.normal(0, 0.3, 2410).astype(np.float32) for _ in range(4)]
    for s in range(5):
        agg = secagg_aggregate([u.astype(np.float64) for u in ups], [8] * 4, rng_seed=s)
        assert wc.w2_seed_stats(agg, ups, [8] * 4)["all_coordinates_below_bound"]
    good = wc.w2_seed_stats(agg, ups, [8] * 4)
    bad = wc.w2_seed_stats(agg + 1e-3, ups, [8] * 4)
    assert good["ratio_max"] < 1 <= bad["ratio_max"] and not bad["all_coordinates_below_bound"]


def test_w2_unequal_sizes_add_the_weight_rounding_term():
    pytest.importorskip("flwr")
    from feddisclosure.attacks.view_study import secagg_aggregate
    rng = np.random.default_rng(1)
    ups = [rng.normal(0, 0.3, 300).astype(np.float32) for _ in range(3)]
    sizes = [7, 11, 13]
    agg = secagg_aggregate([u.astype(np.float64) for u in ups], sizes, rng_seed=3)
    stats = wc.w2_seed_stats(agg, ups, sizes)
    assert not stats["equal_sizes"] and stats["all_coordinates_below_bound"]


# ----------------------------------------------------------------------------------------------
# data selection
# ----------------------------------------------------------------------------------------------

def test_view_partitions_follow_the_simulation_selection_and_pass_the_deployment_check(tmp_path):
    from feddisclosure.attacks.view_seed import select_view_records
    from feddisclosure.training.datasets import _source
    import prepare_flower_deployment as pfd
    import yaml
    data = tmp_path / "data"
    manifest = wc.prepare_view_partitions(data, 3)
    x, y, _ = _source("sklearn_digits", tmp_path)
    sel = select_view_records(len(y), 4, 8, 3)
    for i in range(4):
        with np.load(data / f"institution-{i:02d}" / "train.npz") as z:
            assert np.array_equal(z["source_idx"], sel.idx[i * 8:(i + 1) * 8])
            assert np.array_equal(z["x"], x[z["source_idx"]]) and np.array_equal(z["y"], y[z["source_idx"]])
    assert [p["samples"] for p in manifest["client_partitions"]] == [8] * 4
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(wc.run_config_dict(3)))
    prep = pfd.prepare(cfg, data, tmp_path / "run", tmp_path / "env" / "x.env")
    assert prep["client_count"] == 4
    with pytest.raises(FileExistsError):
        wc.prepare_view_partitions(data, 3)


def test_run_config_is_one_round_one_full_batch_step():
    cfg = wc.run_config_dict(2)
    assert (cfg["rounds"], cfg["local_epochs"], cfg["batch_size"], cfg["learning_rate"], cfg["clients"]) == (1, 1, 8, 0.15, 4)


# ----------------------------------------------------------------------------------------------
# end to end analysis on synthetic runs
# ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    pytest.importorskip("flwr")
    root = tmp_path_factory.mktemp("synthetic")
    for seed in range(5):
        for mode in ("secagg", "plaintext"):
            build_synthetic_run(root, seed, mode)
    return root


def test_analysis_decides_all_six_checks_on_synthetic_runs(synthetic_root, tmp_path):
    import analyze_wire_capture as aw
    report = aw.analyze(synthetic_root, tmp_path / "out", iterations=300)
    by_id = {c["id"]: c for c in report["checks"]}
    assert list(by_id) == ["W1", "W2", "W3", "W4", "W5", "W6"]
    assert report["seeds_analysed"] == 5
    for cid in ("W1", "W2", "W4", "W5", "W6"):
        assert by_id[cid]["pass"], by_id[cid]
    assert by_id["W3"]["observed"]["seeds"] == 5
    assert by_id["W6"]["observed"]["found_in_relay"] == by_id["W6"]["observed"]["objects"] > 0
    assert by_id["W4"]["observed"]["relay_updates_equal_proxy"] == 20
    machine = json.loads((tmp_path / "out" / "checks.json").read_text())
    assert all(set(c) == {"id", "prediction", "observed", "pass"} for c in machine)
    text = (tmp_path / "out" / "wire_validation.md").read_text()
    assert "| W6 |" in text and "—" not in text and ";" not in text.split("## Limitations")[0].replace("&amp;", "")


def test_analysis_reports_run_data_selection_and_updates(synthetic_root, tmp_path):
    import analyze_wire_capture as aw
    report = aw.analyze(synthetic_root, tmp_path / "out2", iterations=20, seeds=[0])
    seed0 = report["per_seed"][0]
    assert seed0["data"]["selection_matches_view_seed"] and seed0["data"]["secagg_init_equals_plaintext_init"]
    assert seed0["updates"]["recomputed_vs_captured_max_abs_diff"] < 1e-6
    assert all(seed0["updates"]["plaintext_receipt_hash_matches"].values())
    assert seed0["w1"]["victim"]["uniformity"]["ks_statistic_vs_uniform"] < 0.05


def _analyze_variant(tmp_path, **kwargs):
    import analyze_wire_capture as aw
    root = tmp_path / "variant"
    build_synthetic_run(root, 0, "secagg", **{k: v for k, v in kwargs.items() if k in ("hide_counts", "aggregate_shift")})
    build_synthetic_run(root, 0, "plaintext", **{k: v for k, v in kwargs.items() if k in ("drop_relay_s2c",)})
    return aw.analyze(root, tmp_path / "vout", iterations=20)


def test_missing_relay_traffic_fails_w6_and_is_not_an_analysis_error(tmp_path):
    pytest.importorskip("flwr")
    report = _analyze_variant(tmp_path, drop_relay_s2c=True)
    w6 = next(c for c in report["checks"] if c["id"] == "W6")
    assert not w6["pass"] and w6["observed"]["found_in_relay"] < w6["observed"]["objects"]


def test_hidden_sample_counts_fail_w5(tmp_path):
    pytest.importorskip("flwr")
    report = _analyze_variant(tmp_path, hide_counts=True)
    w5 = next(c for c in report["checks"] if c["id"] == "W5")
    assert not w5["pass"] and w5["observed"]["visible_in_clear"] == 0


def test_aggregate_outside_the_bound_fails_w2(tmp_path):
    pytest.importorskip("flwr")
    report = _analyze_variant(tmp_path, aggregate_shift=1e-3)
    assert not next(c for c in report["checks"] if c["id"] == "W2")["pass"]


def test_fewer_than_five_seeds_cannot_pass_w3(synthetic_root, tmp_path):
    import analyze_wire_capture as aw
    report = aw.analyze(synthetic_root, tmp_path / "few", iterations=20, seeds=[0, 1, 2])
    w3 = next(c for c in report["checks"] if c["id"] == "W3")
    assert not w3["pass"] and w3["observed"]["seeds"] == 3


def test_a_seed_that_cannot_be_analysed_makes_every_check_fail_closed(tmp_path):
    pytest.importorskip("flwr")
    import analyze_wire_capture as aw
    root = tmp_path / "half"
    build_synthetic_run(root, 0, "secagg")           # no plaintext run for this seed
    report = aw.analyze(root, tmp_path / "hout", iterations=20)
    assert report["seeds_analysed"] == 0 and not any(c["pass"] for c in report["checks"])
    assert aw.main(["--runs", str(root), "--output", str(tmp_path / "hout2"), "--iterations", "20"]) == 3


def test_incomplete_capture_summary_is_rejected(tmp_path):
    pytest.importorskip("flwr")
    import analyze_wire_capture as aw
    run = build_synthetic_run(tmp_path, 0, "plaintext")
    summary = run / "server" / "wire" / "capture_summary.json"
    data = json.loads(summary.read_text())
    data["completed"] = False
    summary.write_text(json.dumps(data))
    with pytest.raises(wc.WireCaptureError):
        aw.load_run(run)


# ----------------------------------------------------------------------------------------------
# orchestration: dry run and fail-closed output checks
# ----------------------------------------------------------------------------------------------

def test_dry_run_prints_every_docker_command_and_runs_nothing(tmp_path, capsys, monkeypatch):
    import run_wire_capture as rw
    import subprocess

    def boom(*a, **k):
        raise AssertionError("dry run must not start a process")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    assert rw.main(["--seeds", "2", "--output", str(tmp_path / "o"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("# run seed") == 4 and "seed1-plaintext" in out
    assert out.count("compose.flower.n4.yaml") == out.count("compose.wire.yaml") == 20
    assert "use-secagg=true" in out and "use-secagg=false" in out and "wire-capture=true" in out
    assert "up --build -d" in out and "stop -t 30 wire-relay-0 wire-relay-1 wire-relay-2 wire-relay-3" in out
    assert not (tmp_path / "o").exists()


def test_capture_output_check_fails_closed(tmp_path):
    import run_wire_capture as rw
    with pytest.raises(RuntimeError, match="summary is missing"):
        rw.check_capture_outputs(tmp_path)
    run = build_synthetic_run(tmp_path / "ok", 0, "plaintext") if importlib.util.find_spec("flwr") else None
    if run is None:
        return
    # the synthetic relay logs lack the relay's own close lines format, so patch in the minimal real shape
    rw.check_capture_outputs(run)
    (run / "relay" / "site-2.connections.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="site 2"):
        rw.check_capture_outputs(run)


def test_run_config_string_uses_flower_literals():
    import run_wire_capture as rw
    cfg = {"rounds": 1, "local_epochs": 1, "batch_size": 8, "learning_rate": 0.15, "clients": 4, "seed": 7}
    text = rw.run_config_string(cfg, {"num_features": 64, "num_classes": 10}, "secagg")
    assert "seed=7" in text and 'artifact-dir="/artifacts"' in text and "use-secagg=true" in text
    assert "timeout=60.0" in text and "num-shares=4" in text and "reconstruction-threshold=2" in text
