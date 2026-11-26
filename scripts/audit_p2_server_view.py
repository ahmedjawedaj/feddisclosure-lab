#!/usr/bin/env python3
"""P2: what the Flower server observes with protections on (v0.16.13, task T7).

Rows S4, S6 and S6b of docs/preregistration/v0.16-p2.md.

* S4: fit metrics returned under SecAgg+ in a real Docker run (compose.flower.n4.yaml plus
  compose.wire.yaml, recording ServerApp, metric returning client). The script can start the run
  (--s4 run) or read an existing run directory (--s4 existing --s4-dir DIR).
* S6 and S6b: in process, against flwr.server.strategy.DifferentialPrivacyServerSideFixedClipping.

Usage: python scripts/audit_p2_server_view.py --out DIR [--quick] [--s4 run|existing|skip] [--s4-dir DIR]
The pre-registered numbers are in PREDICTIONS below. A failed row stays failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import flwr  # noqa: E402
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays  # noqa: E402
from flwr.server.client_manager import SimpleClientManager  # noqa: E402
from flwr.server.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg  # noqa: E402

import audit_flower_dp as p1  # noqa: E402
from feddisclosure.evaluation.intervals import wilson_interval  # noqa: E402

DIM, C, Z, N, DELTA = p1.DIM, p1.C, p1.Z, p1.N, p1.DELTA
OK = Status(code=Code.OK, message="")
SEED_BASE = 20261126
ROUNDS = 2000
P_HIGH, P_LOW, P_CTL = 0.85, 0.15, 0.5
THRESHOLD = 0.8 * C
R_LO, R_HI = 0.2 * C, 1.4 * C
SENS_PAIR_NORM = 5 * C
S4_KEYS = ("probe_const", "train_loss", "n_local")

# Numbers written in docs/preregistration/v0.16-p2.md before this script existed.
PREDICTIONS = {
    "S6_accuracy": 0.85, "S6_accuracy_tol": 0.03,
    "S6_exact_rounds": ROUNDS,
    "S6b_sens": {"M10": 0.1, "M9": 1 / 9}, "S6b_sens_rel_tol": 0.001,
    "S6b_sigma": 0.1, "S6b_sigma_rel_tol": 0.02,
    "S6b_zeff": {"M10": 1.0, "M9": 0.9}, "S6b_zeff_rel_tol": 0.02,
    "S6b_eps": {"M10": 4.377178, "M9": 4.947319}, "S6b_eps_rel_tol": 0.01,
    "S6b_mix_share_m9": 0.5, "S6b_mix_share_tol": 0.03,
    "S6b_mix_zeff_high": 0.985, "S6b_mix_zeff_low": 0.915, "S6b_mix_zeff_tol": 0.01,
}


# ---------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(value: float, target: float, rel: float | None = None, absolute: float | None = None) -> bool:
    tol = absolute if absolute is not None else abs(target) * rel
    return abs(value - target) <= tol


def unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


class Driver:
    """The real flwr.server.strategy fixed clipping wrapper, driven through configure_fit and aggregate_fit."""

    def __init__(self, z: float, n_cfg: int = N) -> None:
        inner = FedAvg(inplace=True, fraction_fit=1.0, fraction_evaluate=0.0,
                       min_fit_clients=n_cfg, min_available_clients=n_cfg)
        self.strategy = DifferentialPrivacyServerSideFixedClipping(inner, z, C, n_cfg)
        self.manager = SimpleClientManager()
        self.proxies = [p1.StubProxy(str(i)) for i in range(n_cfg)]
        for proxy in self.proxies:
            self.manager.register(proxy)
        self.g = [np.zeros(DIM, dtype=np.float32)]

    def round(self, updates: dict[int, np.ndarray], failed: bool = False):
        """updates maps a client index to its update. Returns (released parameters or None, result cids)."""
        self.strategy.configure_fit(1, ndarrays_to_parameters(self.g), self.manager)
        results = [(self.proxies[i], FitRes(OK, ndarrays_to_parameters([self.g[0] + u]), 1, {}))
                   for i, u in sorted(updates.items())]
        failures = [RuntimeError("client reported a failure")] if failed else []
        params, _ = self.strategy.aggregate_fit(1, results, failures)
        released = None if params is None else parameters_to_ndarrays(params)[0].astype(np.float64)
        return released, [p.cid for p, _ in results]


# ---------------------------------------------------------------- S6
def run_arm(name: str, k: int, failure_mode: bool, dependent: bool, rounds: int) -> dict:
    """One S6 arm. failure_mode True: a non-participating target is reported as a failure."""
    rng = np.random.default_rng([SEED_BASE, k])
    np.random.seed(SEED_BASE + k)
    others_rng = np.random.default_rng(1800)
    others = {i: (3 * C * unit(others_rng)).astype(np.float32) for i in range(1, N)}
    drv = Driver(Z)
    rows = []
    for _ in range(rounds):
        r = float(rng.uniform(R_LO, R_HI))
        direction = unit(rng)
        coin = float(rng.random())
        c = min(r, C)
        high = c > THRESHOLD
        p = (P_HIGH if high else P_LOW) if dependent else P_CTL
        part = coin < p
        updates = dict(others)
        if part:
            updates[0] = (r * direction).astype(np.float32)
        released, cids = drv.round(updates, failed=(failure_mode and not part))
        # Observed by the server: the release bit (bit arms) or the participant set (set arms).
        observed = (released is not None) if failure_mode else ("0" in cids)
        rows.append({"c": c, "high": high, "participated": part, "released": released is not None,
                     "target_in_results": "0" in cids, "n_results": len(cids), "observed": bool(observed)})
    # A reported failure is passed in the failure list. The target is then not in the result list.
    correct = sum(1 for x in rows if x["observed"] == x["high"])
    lo, hi = wilson_interval(correct, rounds)
    n_high = sum(x["high"] for x in rows)
    part_rate_high = sum(x["participated"] for x in rows if x["high"]) / max(1, n_high)
    part_rate_low = sum(x["participated"] for x in rows if not x["high"]) / max(1, rounds - n_high)
    if failure_mode:
        exact = sum(1 for x in rows if x["released"] == x["participated"])
    else:
        exact = sum(1 for x in rows if x["released"] and x["n_results"] == 9 + int(x["participated"])
                    and x["target_in_results"] == x["participated"])
    part_c = [x["c"] for x in rows if x["participated"]]
    absent_c = [x["c"] for x in rows if not x["participated"]]
    return {
        "arm": name, "mode": "failure reported (release bit)" if failure_mode else "absent (participant set)",
        "dependent": dependent, "rounds": rounds, "seed": [SEED_BASE, k], "noise_seed": SEED_BASE + k,
        "correct": correct, "accuracy": correct / rounds, "wilson95": [lo, hi],
        "majority_class_accuracy": max(n_high, rounds - n_high) / rounds, "share_high": n_high / rounds,
        "participation_rate_high": part_rate_high, "participation_rate_low": part_rate_low,
        "share_participated": sum(x["participated"] for x in rows) / rounds,
        "mean_clipped_norm_participated": float(np.mean(part_c)) if part_c else None,
        "mean_clipped_norm_absent": float(np.mean(absent_c)) if absent_c else None,
        "exact_release_rule_rounds": exact,
        "share_9_results": sum(1 for x in rows if x["n_results"] == 9) / rounds,
        "rounds_table": [(x["high"], x["participated"]) for x in rows],
    }


def s6_clip(draws: int = 200) -> dict:
    rng = np.random.default_rng([SEED_BASE, 9])
    drv = Driver(0.0, n_cfg=1)
    worst, good = 0.0, 0
    for _ in range(draws):
        r = float(rng.uniform(R_LO, R_HI))
        out, _ = drv.round({0: (r * unit(rng)).astype(np.float32)})
        norm, want = float(np.linalg.norm(out)), min(r, C)
        rel = abs(norm - want) / want
        worst = max(worst, rel)
        good += int(rel <= 1e-5)
    return {"draws": draws, "within_1e-5": good, "worst_relative_error": worst}


def verdicts_s6(arms: dict, clip: dict, rounds: int) -> dict:
    out = {}
    for tag in ("bit", "set"):
        a = arms[f"{tag}-dep"]
        ok = a["wilson95"][0] > 0.5 and within(a["accuracy"], PREDICTIONS["S6_accuracy"], absolute=PREDICTIONS["S6_accuracy_tol"])
        out[f"S6-{tag}"] = {"pass": bool(ok), "accuracy": a["accuracy"], "wilson95": a["wilson95"],
                            "rule": "Wilson lower limit above 0.5 and accuracy 0.85 within 0.03"}
    ctl = {t: arms[f"{t}-ctl"] for t in ("bit", "set")}
    ok = all(c["wilson95"][0] <= 0.5 <= c["wilson95"][1] for c in ctl.values())
    out["S6-ctl"] = {"pass": bool(ok), "accuracy": {t: c["accuracy"] for t, c in ctl.items()},
                     "wilson95": {t: c["wilson95"] for t, c in ctl.items()}, "rule": "both control intervals contain 0.5"}
    ok = all(arms[f"{t}-{d}"]["exact_release_rule_rounds"] == rounds for t in ("bit", "set") for d in ("dep", "ctl"))
    out["S6-exact"] = {"pass": bool(ok), "rounds_matching": {k: v["exact_release_rule_rounds"] for k, v in arms.items()},
                       "rule": "release rule matches in every round of every arm"}
    out["S6-clip"] = {"pass": clip["within_1e-5"] == clip["draws"], **clip}
    out["S6"] = {"pass": all(out[k]["pass"] for k in ("S6-bit", "S6-set", "S6-ctl", "S6-exact", "S6-clip"))}
    return out


# ---------------------------------------------------------------- S6b
def s6b_config(m: int, runs: int) -> dict:
    """M10 (m = 10 results) and M9 (m = 9, target absent). The measured client is client 1."""
    rng = np.random.default_rng(1700 + m)
    others = [(3 * C * unit(rng)).astype(np.float32) for _ in range(m - 1)]
    pool = [i for i in range(N) if i != 1 and (m == 10 or i != 0)]   # client 1 is measured, client 0 is absent in M9
    ids = [1] + pool[: m - 1]
    target = (SENS_PAIR_NORM * np.eye(DIM)[0]).astype(np.float32)
    null = p1.null_update()

    def updates(first):
        return {cid: (first if i == 0 else others[i - 1]) for i, cid in enumerate(ids)}

    quiet = Driver(0.0)
    out1 = quiet.round(updates(target))[0]
    out0 = quiet.round(updates(null))[0]
    sens = float(np.linalg.norm(out1 - out0))
    noisy = Driver(Z)
    np.random.seed(SEED_BASE + 1)
    samples = np.stack([noisy.round(updates(null))[0] for _ in range(runs)])
    ns = p1.noise_stats(samples - out0)
    sigma = ns["sigma"]
    mu = sens / sigma
    return {"results": m, "noise_runs": runs, "sensitivity_zero_out": sens, "noise": ns, "sigma": sigma, "mu": mu,
            "effective_noise_multiplier": sigma / (sens / C), "eps_at_delta": p1.gauss_eps(mu, DELTA),
            "claimed_sigma": Z * C / N, "claimed_eps_at_delta": p1.gauss_eps(1.0 / Z, DELTA),
            "noise_seed": SEED_BASE + 1}


def verdicts_s6b(cfg: dict, set_dep: dict) -> dict:
    P = PREDICTIONS
    out = {}
    out["S6b-sens"] = {"pass": all(within(cfg[k]["sensitivity_zero_out"], P["S6b_sens"][k], rel=P["S6b_sens_rel_tol"]) for k in cfg),
                       "measured": {k: cfg[k]["sensitivity_zero_out"] for k in cfg}, "predicted": P["S6b_sens"]}
    out["S6b-sigma"] = {"pass": all(within(cfg[k]["sigma"], P["S6b_sigma"], rel=P["S6b_sigma_rel_tol"]) for k in cfg),
                        "measured": {k: cfg[k]["sigma"] for k in cfg}, "predicted": P["S6b_sigma"]}
    ratio = cfg["M9"]["effective_noise_multiplier"] / cfg["M10"]["effective_noise_multiplier"]
    ok = all(within(cfg[k]["effective_noise_multiplier"], P["S6b_zeff"][k], rel=P["S6b_zeff_rel_tol"]) for k in cfg)
    ok = ok and within(ratio, 0.9, rel=P["S6b_zeff_rel_tol"])
    out["S6b-zeff"] = {"pass": bool(ok), "measured": {k: cfg[k]["effective_noise_multiplier"] for k in cfg},
                       "ratio_M9_over_M10": ratio, "predicted": P["S6b_zeff"]}
    out["S6b-eps"] = {"pass": all(within(cfg[k]["eps_at_delta"], P["S6b_eps"][k], rel=P["S6b_eps_rel_tol"]) for k in cfg),
                      "measured": {k: cfg[k]["eps_at_delta"] for k in cfg}, "predicted": P["S6b_eps"], "delta": DELTA}
    # Mixture: per round multiplier is the measured M10 value when the target took part and the M9 value otherwise.
    table = set_dep["rounds_table"]
    z10, z9 = cfg["M10"]["effective_noise_multiplier"], cfg["M9"]["effective_noise_multiplier"]
    zs = [(high, z10 if part else z9) for high, part in table]
    hi_z = [z for high, z in zs if high]
    lo_z = [z for high, z in zs if not high]
    share9 = sum(1 for _, part in table if not part) / len(table)
    ok = (within(share9, P["S6b_mix_share_m9"], absolute=P["S6b_mix_share_tol"])
          and within(float(np.mean(hi_z)), P["S6b_mix_zeff_high"], absolute=P["S6b_mix_zeff_tol"])
          and within(float(np.mean(lo_z)), P["S6b_mix_zeff_low"], absolute=P["S6b_mix_zeff_tol"]))
    out["S6b-mix"] = {"pass": bool(ok), "share_rounds_9_results": share9, "mean_zeff_high": float(np.mean(hi_z)),
                      "mean_zeff_low": float(np.mean(lo_z)),
                      "eps_mean_over_rounds": float(np.mean([cfg["M10"]["eps_at_delta"] if part else cfg["M9"]["eps_at_delta"]
                                                             for _, part in table]))}
    out["S6b"] = {"pass": all(out[k]["pass"] for k in ("S6b-sens", "S6b-sigma", "S6b-zeff", "S6b-eps", "S6b-mix"))}
    return out


# ---------------------------------------------------------------- S4
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def count_arrays(content: dict) -> int:
    return sum(len(rec) for rec in content.get("array_records", {}).values())


def check_s4_run(run_dir: Path) -> dict:
    """Check S4a to S4d for one SecAgg+ run directory. Pure function of the saved files."""
    wire = run_dir / "server" / "wire"
    summary = json.loads((wire / "capture_summary.json").read_text())
    meta = json.loads((run_dir / "wire_run.json").read_text())
    res = {"run_dir": str(run_dir), "run_ok": bool(meta.get("ok")), "capture_completed": bool(summary.get("completed")),
           "mode": summary.get("mode")}
    if not (res["run_ok"] and res["capture_completed"]) or res["mode"] != "secagg":
        res.update(status="NOT RUN", pass_=False)
        return res
    msgs = _read_jsonl(wire / "messages.jsonl")
    replies = [m for m in msgs if m["direction"] == "in" and m["stage"] == "collect_masked_vectors" and not m["error"]]
    seen_path = wire / "fit_results_seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.is_file() else {"results": []}
    seen_by_node = {r["node_id"]: r for r in seen["results"]}
    receipts = []
    for p in sorted((run_dir / "clients").glob("site-*/fit_metrics_sent.jsonl")):
        for line in _read_jsonl(p):
            receipts.append({"site": line["site_id"], "metrics": line["metrics"]})
    unused = list(range(len(receipts)))
    per = []
    for m in replies:
        rec = m["content"]["config_records"].get("fitres.metrics")
        a = rec is not None and set(rec) == set(S4_KEYS)
        match = None
        if a:
            for idx in unused:
                if receipts[idx]["metrics"] == rec:
                    match = idx
                    break
        if match is not None:
            unused.remove(match)
        b = match is not None
        s = seen_by_node.get(str(m["node_id"]))
        c = bool(s is not None and a and s["metrics"] == rec)
        n_arrays = count_arrays(m["content"])
        d = n_arrays == 0 and m.get("arrays_file") is None
        per.append({"node_id": m["node_id"], "S4a": bool(a), "S4b": bool(b), "S4c": c, "S4d": bool(d),
                    "metrics_at_grid": rec, "metrics_at_strategy": None if s is None else s["metrics"],
                    "receipt_site": None if match is None else receipts[match]["site"],
                    "array_records": {k: len(v) for k, v in m["content"]["array_records"].items()},
                    "payload_bytes": m["payload_bytes"], "num_examples_visible": m.get("num_examples"),
                    "masked_vector_in_config_record": m.get("masked_file") is not None})
    complete = len(per) == 4 and len(receipts) == 4
    res.update(replies=per, receipts=len(receipts), reply_count=len(replies),
               participant_node_ids=sorted(str(m["node_id"]) for m in replies),
               checks_true={k: sum(1 for x in per if x[k]) for k in ("S4a", "S4b", "S4c", "S4d")},
               status="RUN", pass_=bool(complete and all(all(x[k] for k in ("S4a", "S4b", "S4c", "S4d")) for x in per)))
    return res


def run_s4_docker(out: Path, seeds: int, timeout: int) -> tuple[int, str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_wire_capture.py"), "--seeds", str(seeds), "--modes", "secagg",
           "--app-dir", str(ROOT / "docker" / "wire_relay" / "app_metrics"), "--output", str(out), "--timeout", str(timeout)]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:" + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


def s4_verdict(dirs: list[Path], seeds: int) -> dict:
    runs = []
    for d in dirs:
        try:
            runs.append(check_s4_run(d))
        except (OSError, ValueError, KeyError) as exc:
            runs.append({"run_dir": str(d), "status": "NOT RUN", "pass_": False, "error": f"{type(exc).__name__}: {exc}"})
    ran = [r for r in runs if r.get("status") == "RUN"]
    if not ran:
        return {"status": "NOT RUN", "pass": False, "runs": runs, "reason": "no run finished with a complete capture"}
    replies_ok = sum(1 for r in ran for x in r["replies"] if all(x[k] for k in ("S4a", "S4b", "S4c", "S4d")))
    total = sum(len(r["replies"]) for r in ran)
    return {"status": "RUN", "pass": bool(len(ran) == seeds and all(r["pass_"] for r in ran)), "runs_finished": len(ran),
            "runs_requested": seeds, "replies_all_checks_true": replies_ok, "replies_total": total, "runs": runs}


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="small run for tests, not evidence")
    ap.add_argument("--s4", choices=("run", "existing", "skip"), default="skip")
    ap.add_argument("--s4-dir", help="run directory root of an existing wire capture (--s4 existing)")
    ap.add_argument("--s4-seeds", type=int, default=2)
    ap.add_argument("--s4-timeout", type=int, default=900)
    args = ap.parse_args(argv)
    rounds = 200 if args.quick else ROUNDS
    runs = 500 if args.quick else 20_000
    logging.getLogger("flwr").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    arms = {}
    for k, (name, failure, dep) in enumerate((("bit-dep", True, True), ("bit-ctl", True, False),
                                              ("set-dep", False, True), ("set-ctl", False, False))):
        arms[name] = run_arm(name, k, failure, dep, rounds)
        print(f"arm {name} done {time.time() - t0:.0f}s accuracy {arms[name]['accuracy']:.4f}", flush=True)
    clip = s6_clip()
    s6 = verdicts_s6(arms, clip, rounds)
    cfg = {"M10": s6b_config(10, runs), "M9": s6b_config(9, runs)}
    print(f"S6b configs done {time.time() - t0:.0f}s", flush=True)
    s6b = verdicts_s6b(cfg, arms["set-dep"])

    s4 = {"status": "NOT RUN", "pass": False, "reason": "S4 skipped by option"}
    docker_log = None
    if args.s4 in ("run", "existing"):
        root = Path(args.s4_dir) if args.s4 == "existing" else out / "s4_docker"
        if args.s4 == "run":
            code, docker_log = run_s4_docker(root, args.s4_seeds, args.s4_timeout)
            print(f"S4 docker run exit {code} {time.time() - t0:.0f}s", flush=True)
        dirs = [root / "runs" / f"seed{s}-secagg" for s in range(args.s4_seeds)]
        s4 = s4_verdict(dirs, args.s4_seeds)
        if docker_log is not None:
            s4["docker_runner_tail"] = docker_log

    for arm in arms.values():
        arm.pop("rounds_table")
    doc = {
        "schema": "feddisclosure.p2_server_view.v1", "quick": args.quick, "preregistration": "docs/preregistration/v0.16-p2.md",
        "environment": {"flwr": flwr.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "parameters": {"dim": DIM, "clipping_norm": C, "noise_multiplier": Z, "num_sampled_clients": N, "delta": DELTA,
                       "rounds_per_arm": rounds, "noise_runs": runs, "participation_high": P_HIGH, "participation_low": P_LOW,
                       "threshold": THRESHOLD, "raw_norm_range": [R_LO, R_HI], "seed_base": SEED_BASE},
        "predictions": PREDICTIONS, "S4": s4, "S6_arms": arms, "S6_verdicts": s6, "S6b_configs": cfg, "S6b_verdicts": s6b,
        "script_sha256": sha256_file(Path(__file__)), "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "p2_server_view.json").write_text(json.dumps(doc, indent=2, default=repr) + "\n")
    print(f"wrote {out / 'p2_server_view.json'}")
    for name in ("S6", "S6b"):
        print(name, "PASS" if (s6 if name == "S6" else s6b)[name]["pass"] else "FAIL")
    print("S4", s4["status"], "PASS" if s4["pass"] else "not passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
