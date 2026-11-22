"""Analyse wire-level capture runs and decide the pre-registered predictions W1 to W6.

Input is a directory written by scripts/run_wire_capture.py. It holds paired runs for every seed
(SecAgg+ and plaintext). The data selection, the initial model and the client updates come from the
runs themselves (manifest, captured global model, plaintext replies, relay streams). The simulation
that W3 compares against is run on the same data and the same initial model.

Output: wire_validation.json (checks with id, prediction, observed, pass), checks.json and
wire_validation.md. A prediction that fails is recorded as FAIL and is never reworded.
Exit code 0 when every seed was analysed, 3 when a seed could not be analysed. With --strict a failed
prediction gives exit code 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from feddisclosure.attacks.view_seed import client_step_seed, select_view_records
from feddisclosure.attacks.view_study import (attack, client_step, flatten, label_floor, score_victim,
                                               secagg_aggregate, unflatten)
from feddisclosure.training import wire_capture as wc
from feddisclosure.training.task import DigitMLP

LR = 0.15
B, N = 8, 4
VICTIM = 0
LIMITATIONS = [
    "Four clients, eight public digits each, one local full batch SGD step, labels known to the attacker",
    "One Docker host and an insecure internal network. This is not a multi-host deployment",
    "The relay sees the SuperNode to SuperLink Fleet API only. The ServerApp to SuperLink link is seen by the proxy",
    "Sample counts and metrics are visible by protocol design (W5 documents this, it is not a finding about a flaw)",
    "The simulated reference uses the repository SecAgg+ quantiser arithmetic with its own rounding draws",
    "A failed reconstruction is evidence about this attack only, not a privacy guarantee",
]


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def load_npz(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]


def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "wire_run.json").read_text())
    cap = run_dir / "server" / "wire"
    summary = json.loads((cap / "capture_summary.json").read_text())
    if not summary.get("completed"):
        raise wc.WireCaptureError(f"{run_dir}: capture did not complete: {summary.get('error')}")
    messages = [json.loads(line) for line in (cap / "messages.jsonl").read_text().splitlines() if line.strip()]
    relays = wc.load_relay_dir(run_dir / "relay")
    if sorted(relays) != [f"site-{i}" for i in range(N)]:
        raise wc.WireCaptureError(f"{run_dir}: expected relay logs for {N} sites, found {sorted(relays)}")
    return {"dir": run_dir, "meta": meta, "cap": cap, "summary": summary, "messages": messages, "relays": relays}


def discover(root: Path) -> dict[int, dict[str, Path]]:
    found: dict[int, dict[str, Path]] = {}
    for f in sorted(root.rglob("wire_run.json")):
        meta = json.loads(f.read_text())
        if not meta.get("ok", False):
            continue
        found.setdefault(int(meta["seed"]), {})[meta["mode"]] = f.parent
    return found


def load_run_data(data_dir: Path) -> dict:
    """Original records exactly as the run's clients held them, plus the public pool."""
    from feddisclosure.training.datasets import _source
    manifest = json.loads((data_dir / "manifest.json").read_text())
    cx, cy, idx = [], [], []
    for part in manifest["client_partitions"]:
        path = data_dir / part["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
            raise wc.WireCaptureError(f"{path} does not match the manifest")
        with np.load(path, allow_pickle=False) as z:
            cx.append(z["x"].copy())
            cy.append(z["y"].copy())
            idx.append(z["source_idx"].copy())
    cx, cy, idx = np.stack(cx), np.stack(cy), np.stack(idx)
    x, y, _ = _source("sklearn_digits", data_dir)
    seed = int(manifest["seed"])
    sel = select_view_records(len(y), N, B, seed)
    matches = bool(np.array_equal(sel.idx.reshape(N, B), idx))
    pool_idx = sel.pool_idx if matches else np.setdiff1d(np.arange(len(y)), idx.ravel())
    return {"seed": seed, "cx": cx.astype(np.float32), "cy": cy, "idx": idx, "sizes": [int(len(c)) for c in cy],
            "pool_x": x[pool_idx], "pool_y": y[pool_idx], "selection_matches_view_seed": matches}


def state_from_arrays(arrays) -> dict:
    names = list(DigitMLP(64, 10).state_dict().keys())
    if len(names) != len(arrays):
        raise wc.WireCaptureError("array count does not match the model")
    return {n: torch.from_numpy(np.array(a, dtype=np.float32)) for n, a in zip(names, arrays)}


def captured_init(run: dict) -> list[np.ndarray]:
    for m in run["messages"]:
        if m["direction"] == "out" and m.get("arrays_file"):
            return load_npz(run["cap"] / m["arrays_file"])
    raise wc.WireCaptureError("no outgoing global model was recorded")


def receipt_digest(run_dir: Path, site: int) -> str | None:
    f = run_dir / "clients" / f"site-{site}" / "events.jsonl"
    if not f.is_file():
        return None
    events = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    trains = [e for e in events if e.get("event_type") == "train"]
    return trains[-1]["weights_sha256"] if trains else None


# ---------------------------------------------------------------------------------------------
# Linking the proxy record to the relay streams
# ---------------------------------------------------------------------------------------------

def relay_blob_index(relays: dict) -> dict:
    return {name: {d: wc.blobs_from_relay(conns, d) for d in ("c2s", "s2c")} for name, conns in relays.items()}


def link_replies(run: dict, index: dict, stages: tuple[str, ...]) -> dict[int, dict]:
    """For each reply of the given stages: node id to {site, entry}. Site is found by content hash."""
    out = {}
    for e in run["messages"]:
        if e["direction"] != "in" or e["stage"] not in stages or e.get("error"):
            continue
        big = sorted((o for o in e["objects"] if o["nbytes"] >= 1024 and o["kind"] in ("array", "bytes")),
                     key=lambda o: -o["nbytes"])
        if not big:
            raise wc.WireCaptureError("reply without a large object, cannot link it to a relay")
        hit = {name for name, per_dir in index.items() if big[0]["sha256"] in per_dir["c2s"]}
        out[e["node_id"]] = {"site": int(next(iter(hit)).split("-")[1]) if len(hit) == 1 else None,
                             "relays": sorted(hit), "entry": e}
    return out


# ---------------------------------------------------------------------------------------------
# Attack helpers
# ---------------------------------------------------------------------------------------------

def run_attack(init, observed_state_vec, labels, truth_x, *, observer, iterations, seed, aggregate=False):
    obs_state = unflatten(observed_state_vec, init)
    delta = {k: init[k] - obs_state[k] for k in init}
    rec = attack(init, delta, labels, observer, lr=LR, iterations=iterations, seed=seed + 1)
    return score_victim(truth_x, rec, B)


def naive_dequantise(masked: list[np.ndarray], num_examples: int, *, max_weight: float = 1000.0) -> np.ndarray:
    """What an observer gets by treating one masked vector as an ordinary quantised update.

    The first array is the masked weight factor and is skipped. The rest is dequantised with the public
    clipping range and quantisation range, then divided by the visible weight."""
    q = wc.quantised_weights([num_examples], max_weight=max_weight)[0]
    parts = [m.astype(np.float64) * (2 * wc.CLIP_RANGE / wc.QUANT_RANGE) - wc.CLIP_RANGE for m in masked[1:]]
    return np.concatenate([p.ravel() for p in parts]) / (q / wc.QUANT_RANGE)


def uniformity(masked: list[np.ndarray], truth_vec: np.ndarray, mod: int = 2 ** 32) -> dict:
    from scipy import stats
    u = np.concatenate([m.astype(np.float64).ravel() for m in masked[1:]]) / mod
    naive_corr = float(np.corrcoef(u, truth_vec)[0, 1]) if np.std(u) > 0 and np.std(truth_vec) > 0 else 0.0
    return {"entries": int(u.size), "mean_over_modulus": float(u.mean()),
            "ks_statistic_vs_uniform": float(stats.kstest(u, "uniform").statistic),
            "correlation_with_true_update": naive_corr}


# ---------------------------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------------------------

class section:
    """Run one check's computation. A failure is recorded for that check only and the others still run."""

    def __init__(self, res: dict, name: str):
        self.res, self.name = res, name

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, tb):
        if evalue is not None:
            self.res.setdefault("section_errors", {})[self.name] = f"{etype.__name__}: {evalue}"
        return evalue is not None

def analyze_seed(seed: int, sec_dir: Path | None, plain_dir: Path | None, *, iterations: int) -> dict:
    res: dict = {"seed": seed, "errors": []}
    if sec_dir is None or plain_dir is None:
        res["errors"].append("both a SecAgg+ run and a plaintext run are needed for every seed")
        return res
    sec, plain = load_run(sec_dir), load_run(plain_dir)
    data = load_run_data(Path(sec["meta"]["data_dir"]))
    cx, cy = data["cx"], data["cy"]
    all_x, all_y = cx.reshape(-1, 64), cy.reshape(-1)
    init_arrays = captured_init(sec)
    init = state_from_arrays(init_arrays)
    res["data"] = {"selection_matches_view_seed": data["selection_matches_view_seed"], "sizes": data["sizes"],
                   "secagg_init_equals_plaintext_init": all(np.array_equal(a, b) for a, b in
                                                            zip(init_arrays, captured_init(plain)))}
    torch.manual_seed(seed)
    seeded = [v.numpy() for v in DigitMLP(64, 10).state_dict().values()]
    res["data"]["captured_init_equals_seeded_torch_init"] = all(np.array_equal(a, b) for a, b in zip(init_arrays, seeded))
    max_weight = float(sec["summary"]["meta"]["max_weight"])

    sec_index, plain_index = relay_blob_index(sec["relays"]), relay_blob_index(plain["relays"])
    sec_links = link_replies(sec, sec_index, ("collect_masked_vectors",))
    plain_links = link_replies(plain, plain_index, ("plain_train",))
    if len(sec_links) != N or any(v["site"] is None for v in sec_links.values()) or \
            sorted(v["site"] for v in sec_links.values()) != list(range(N)):
        res["errors"].append("could not map every SecAgg+ reply to one relay by content hash")
        return res
    if len(plain_links) != N or sorted(v["site"] for v in plain_links.values()) != list(range(N)):
        res["errors"].append("could not map every plaintext reply to one relay by content hash")
        return res

    # exact client updates from the plaintext run (the proxy saw them in clear)
    exact: dict[int, np.ndarray] = {}
    receipts_ok = {}
    for node, link in plain_links.items():
        arrays = load_npz(plain["cap"] / link["entry"]["arrays_file"])
        site = link["site"]
        exact[site] = np.concatenate([a.astype(np.float64).ravel() for a in arrays])
        digest = hashlib.sha256(b"".join(a.astype(np.float32).tobytes() for a in arrays)).hexdigest()
        receipts_ok[site] = receipt_digest(plain["dir"], site) == digest
    updates = [exact[i] for i in range(N)]
    recomputed = [flatten(client_step(init, cx[i], cy[i], lr=LR, sigma=None, seed=client_step_seed(seed, i)))
                  for i in range(N)]
    res["updates"] = {"plaintext_receipt_hash_matches": receipts_ok,
                      "secagg_run_receipt_matches_plaintext_update": {
                          i: receipt_digest(sec["dir"], i) == receipt_digest(plain["dir"], i) for i in range(N)},
                      "recomputed_vs_captured_max_abs_diff": float(max(np.abs(r - u).max()
                                                                       for r, u in zip(recomputed, updates)))}

    # label floor on the victim
    lf = label_floor(cx[VICTIM], cy[VICTIM], data["pool_x"], data["pool_y"], rng=np.random.default_rng(5000 + seed))
    res["floor"] = {k: lf[k] for k in ("class_average_mse", "class_average_rate_mse_below_0_05", "matched_mse",
                                       "matched_rate_mse_below_0_05")}

    aggregate = np.concatenate([a.astype(np.float64).ravel()
                                for a in load_npz(sec["cap"] / sec["summary"]["aggregates"]["dequantised_aggregate"]["file"])])

    # W5 data and W2 inputs from the SecAgg+ run
    stage2 = [v["entry"] for v in sec_links.values()]
    expected = {node: data["sizes"][link["site"]] for node, link in sec_links.items()}
    res["w5_replies"] = [{"node_id": e["node_id"], "num_examples": e.get("num_examples"),
                          "num_examples_visible": e.get("num_examples_visible"),
                          "metrics_visible": e.get("metrics_visible"), "metrics": e.get("metrics")} for e in stage2]
    res["w5_expected"] = {str(k): v for k, v in expected.items()}

    def size_of(link):
        # The clear count when the server saw one, else the public record count. W5 judges visibility.
        seen = link["entry"].get("num_examples")
        return int(seen) if seen is not None else data["sizes"][link["site"]]
    seen_sizes = [size_of(sec_links[n]) for n in sorted(sec_links, key=lambda n: sec_links[n]["site"])]
    with section(res, "w2"):
        res["w2"] = wc.w2_seed_stats(aggregate, updates, seen_sizes, max_weight=max_weight)

    # W1: naive dequantisation of each captured masked vector
    with section(res, "w1"):
        w1_clients = {}
        for node, link in sec_links.items():
            site, e = link["site"], link["entry"]
            masked = load_npz(sec["cap"] / e["masked_file"])
            vec = naive_dequantise(masked, size_of(link), max_weight=max_weight)
            r = run_attack(init, vec, cy[site], cx[site], observer="server_individual_model_delta",
                           iterations=iterations, seed=seed)
            w1_clients[site] = {**r, "uniformity": uniformity(masked, updates[site])}
        res["w1"] = {"victim": w1_clients[VICTIM], "all_clients": w1_clients}

    # W3: attack on the captured aggregate against the simulation on the same data and initial model
    with section(res, "w3"):
        cap_r = run_attack(init, aggregate, all_y, all_x, observer="server_aggregate_model_delta_joint_oracle",
                           iterations=iterations, seed=seed)
        plain_states = [client_step(init, cx[i], cy[i], lr=LR, sigma=None, seed=client_step_seed(seed, i)) for i in range(N)]
        sim_agg = secagg_aggregate([flatten(s) for s in plain_states], data["sizes"], max_weight=max_weight, rng_seed=seed)
        sim_r = run_attack(init, sim_agg, all_y, all_x, observer="server_aggregate_model_delta_joint_oracle",
                           iterations=iterations, seed=seed)
        res["w3"] = {"captured": cap_r, "simulated": sim_r}

    # W4: from the relay streams alone (plaintext run)
    with section(res, "w4"):
        res["w4"] = _w4(plain, plain_links, cx, cy, iterations, seed)

    # W6: every proxied object of the plaintext run against the relay streams
    with section(res, "w6"):
        res["w6"] = _w6(plain, plain_index, sec)
    return res


def _w4(plain, plain_links, cx, cy, iterations, seed) -> dict:
    w4_clients, equal = {}, 0
    for site in range(N):
        conns = plain["relays"][f"site-{site}"]
        relay_init = wc.model_arrays_from_blobs(wc.blobs_from_relay(conns, "s2c"))
        relay_upd = wc.model_arrays_from_blobs(wc.blobs_from_relay(conns, "c2s"))
        proxy_upd = load_npz(plain["cap"] / next(v["entry"]["arrays_file"] for v in plain_links.values() if v["site"] == site))
        equal += int(all(np.array_equal(a, b) for a, b in zip(relay_upd, proxy_upd)))
        vec = np.concatenate([a.astype(np.float64).ravel() for a in relay_upd])
        init_state = state_from_arrays(relay_init)
        obs_state = unflatten(vec, init_state)
        delta = {k: init_state[k] - obs_state[k] for k in init_state}
        rec = attack(init_state, delta, cy[site], "server_individual_model_delta", lr=LR, iterations=iterations, seed=seed + 1)
        w4_clients[site] = score_victim(cx[site], rec, B)
    return {"victim": w4_clients[VICTIM], "all_clients": w4_clients, "relay_equals_proxy": equal}


def _w6(plain, plain_index, sec) -> dict:
    saved = {}
    for e in plain["messages"]:
        if e.get("arrays_file"):
            objs = [o for o in e["objects"] if o["kind"] == "array"]
            for o, a in zip(objs, load_npz(plain["cap"] / e["arrays_file"])):
                saved[o["sha256"]] = a
    unique = {}
    for e in plain["messages"]:
        for o in e["objects"]:
            if o["nbytes"] >= wc.MIN_OBJECT_BYTES:
                unique.setdefault(o["sha256"], o)
    cache: dict = {}
    flags, methods = [], {"blob_hash": 0, "float32_bytes": 0, "either": 0, "missing": 0}
    missing = []
    for sha, o in unique.items():
        by_hash = any(sha in per[d] for per in plain_index.values() for d in per)
        by_bytes = False
        if sha in saved:
            by_bytes = bool(wc.find_in_relays(plain["relays"], np.ascontiguousarray(saved[sha]).tobytes(), _cache=cache))
        methods["blob_hash"] += int(by_hash)
        methods["float32_bytes"] += int(by_bytes)
        methods["either"] += int(by_hash or by_bytes)
        flags.append(by_hash or by_bytes)
        if not (by_hash or by_bytes):
            missing.append({"sha256": sha, "kind": o["kind"], "key": o["key"], "nbytes": o["nbytes"]})
    methods["missing"] = len(missing)
    return {"flags": flags, "methods": methods, "missing": missing[:20],
            "integrity": wc.relay_integrity(plain["relays"]), "secagg_integrity": wc.relay_integrity(sec["relays"])}


# ---------------------------------------------------------------------------------------------
# All seeds, checks and report
# ---------------------------------------------------------------------------------------------

def build_checks(per_seed: list[dict]) -> list[dict]:
    good = [s for s in per_seed if not s["errors"]]
    f = lambda key: [s for s in good if key in s]  # noqa: E731
    checks = [
        wc.check_w1([s["w1"]["victim"]["victim_rate_mse_below_0_05"] for s in f("w1")],
                    [s["floor"]["class_average_rate_mse_below_0_05"] for s in f("w1")],
                    [s["w1"]["victim"]["victim_mse"] for s in f("w1")]),
        wc.check_w2([s["w2"] for s in f("w2")]),
        wc.check_w3([{"rate": s["w3"]["captured"]["victim_rate_mse_below_0_05"], "mse": s["w3"]["captured"]["victim_mse"]}
                     for s in f("w3")],
                    [{"rate": s["w3"]["simulated"]["victim_rate_mse_below_0_05"], "mse": s["w3"]["simulated"]["victim_mse"]}
                     for s in f("w3")]),
        wc.check_w4([s["w4"]["victim"]["victim_rate_mse_below_0_05"] for s in f("w4")],
                    sum(s["w4"]["relay_equals_proxy"] for s in f("w4")), N * len(f("w4"))),
        wc.check_w5([r for s in f("w5_replies") for r in s["w5_replies"]],
                    {int(k): v for s in f("w5_expected") for k, v in s["w5_expected"].items()}),
        wc.check_w6([x for s in f("w6") for x in s["w6"]["flags"]],
                    {k: sum(s["w6"]["methods"][k] for s in f("w6")) for k in ("blob_hash", "float32_bytes", "either", "missing")},
                    all(s["w6"]["integrity"]["ok"] for s in f("w6"))),
    ]
    keys = {"W1": "w1", "W2": "w2", "W3": "w3", "W4": "w4", "W5": "w5_replies", "W6": "w6"}
    for c in checks:
        contributing = len(f(keys[c["id"]]))
        c["observed"]["seeds_with_result"] = contributing
        c["observed"]["seeds_requested"] = len(per_seed)
        if contributing != len(per_seed):
            c["pass"] = False      # fail closed: a seed that could not be evaluated is never ignored
    return checks


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def markdown(report: dict) -> str:
    lines = ["# Wire-level validation (WP3)", "",
             f"Seeds analysed: {report['seeds_analysed']} of {report['seeds_requested']}. "
             f"Attack iterations: {report['iterations']}.", "",
             "| ID | Prediction | Pass rule | Result | Observed |", "|---|---|---|---|---|"]
    for c in report["checks"]:
        obs = ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in c["observed"].items()
                        if not isinstance(v, (dict, list)))
        lines.append(f"| {c['id']} | {c['prediction']} | {c['pass_rule']} | {'PASS' if c['pass'] else 'FAIL'} | {obs} |")
    lines += ["", "## Per seed", "", "| Seed | Errors | Selection matches simulation | Update receipts match | Max recompute diff |",
              "|---|---|---|---|---|"]
    for s in report["per_seed"]:
        d, u = s.get("data", {}), s.get("updates", {})
        lines.append(f"| {s['seed']} | {', '.join(s['errors'] + [f'{k} failed: {v}' for k, v in s.get('section_errors', {}).items()]) or 'none'} | {d.get('selection_matches_view_seed')} | "
                     f"{all(u.get('plaintext_receipt_hash_matches', {}).values()) if u else None} | "
                     f"{u.get('recomputed_vs_captured_max_abs_diff')} |")
    lines += ["", "## Limitations", ""] + [f"- {t}" for t in LIMITATIONS] + [""]
    return "\n".join(lines)


def analyze(root: Path, output: Path, *, iterations: int = 300, seeds: list[int] | None = None) -> dict:
    runs = discover(root)
    if not runs:
        raise wc.WireCaptureError(f"no successful capture runs under {root}")
    wanted = sorted(runs) if seeds is None else seeds
    per_seed = []
    for s in wanted:
        try:
            per_seed.append(analyze_seed(s, runs.get(s, {}).get("secagg"), runs.get(s, {}).get("plaintext"), iterations=iterations))
        except Exception as exc:  # recorded, the seed then counts as not analysed
            per_seed.append({"seed": s, "errors": [f"{type(exc).__name__}: {exc}"]})
    checks = build_checks(per_seed)
    report = {"schema_version": 1, "iterations": iterations, "seeds_requested": len(wanted),
              "seeds_analysed": sum(1 for s in per_seed if not s["errors"]), "checks": checks,
              "per_seed": per_seed, "limitations": LIMITATIONS,
              "thresholds": {"W1_margin": wc.W1_MARGIN, "W3_rate": wc.W3_RATE_TOLERANCE, "W3_mse": wc.W3_MSE_TOLERANCE,
                             "W3_min_seeds": wc.W3_MIN_SEEDS, "W4_min_rate": wc.W4_MIN_RATE}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "wire_validation.json").write_text(json.dumps(report, indent=2, default=_np) + "\n")
    (output / "checks.json").write_text(json.dumps(
        [{k: c[k] for k in ("id", "prediction", "observed", "pass")} for c in checks], indent=2, default=_np) + "\n")
    (output / "wire_validation.md").write_text(markdown(report))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True, help="output root of scripts/run_wire_capture.py")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--strict", action="store_true", help="exit 1 when any prediction fails")
    a = p.parse_args(argv)
    report = analyze(a.runs, a.output or a.runs / "analysis", iterations=a.iterations)
    for c in report["checks"]:
        print(f"{c['id']}: {'PASS' if c['pass'] else 'FAIL'}")
    if report["seeds_analysed"] != report["seeds_requested"]:
        print("some seeds could not be analysed", file=sys.stderr)
        return 3
    return 1 if a.strict and not all(c["pass"] for c in report["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
