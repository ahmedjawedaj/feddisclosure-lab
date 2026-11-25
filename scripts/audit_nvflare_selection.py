#!/usr/bin/env python3
"""Selection check of NVIDIA FLARE 2.9.1 SVTPrivacy, with stated power (v0.16.5).

Written after the pre-registration commit cdd2ce031f7da61d642f1afe5b20f950a90dab08
(docs/preregistration/v0.16-nvflare-selection.md). Every row, number and threshold comes from that
file. The script changes none of them. Anything that differs from the file goes in DEVIATIONS below,
is copied to the JSON evidence, and is appended to the "Deviation log" of the pre-registration.

Rows
  S0  the ceiling of the v0.16.4 N6b design (no filter call)
  S1, S2, S3  real process_dxo calls, 5 blocks of 100,000 per world, event E = the set of nonzero entries
      equals {0, ..., n-1}, 95% family lower bound (Clopper-Pearson, 1.25% per tail)
  C   positive control (LB against 0.1 times the claim)
  V   the two filter files from the wheels of 2.8.3, 2.9.0 and 2.9.1, hashed and run once each
  T   run time

Usage: python scripts/audit_nvflare_selection.py [--out DIR] [--quick] [--skip-versions]
Full mode writes reports/results/privacy/v0160/nvflare_selection/nvflare_selection.json.
Quick mode (few runs, for tests) requires --out.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from statistics import NormalDist

import numpy as np

PREREG_COMMIT = "cdd2ce031f7da61d642f1afe5b20f950a90dab08"
PREREG_FILE = "docs/preregistration/v0.16-nvflare-selection.md"
GAMMA = 1e-5
TAU = 1e-6
BASE_SEED = 20261126
REPLICATION_OFFSET = 5_000_000
ALPHA_TAIL = 0.0125
BLOCKS, BLOCK_RUNS = 5, 100_000
CHUNK = 10_000
OUT_NAME = "nvflare_selection.json"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "results" / "privacy" / "v0160" / "nvflare_selection"
SVT_SHA_291 = "c543bbea2ce3bfe2e35a1d85a26acac8dffc0e95024e777010cdbf0cd0f92858"
PP_SHA_291 = "29bcf457e2c20ff8e5fe92411673b128b0c85a94efd9a0513a102095bdbb58c0"
VERSIONS = ("2.8.3", "2.9.0", "2.9.1")

# Deviations from the pre-registration. Each entry is a dict with "date", "item" and "reason".
# Empty means none were found when the script was written. Entries added later are copied to the
# pre-registration Deviation log by hand.
DEVIATIONS: list[dict] = []

# From the pre-registration "Predictions" table. Tolerances are 4 binomial standard errors.
ROWS = {
    "S1": {"n": 2, "d": 6, "fraction": 0.3, "epsilon": 1.0, "replace": False, "r": 1,
           "p_a": (0.0835, 0.0016), "p_c": (0.0532, 0.0013), "lb": (0.434, 0.030), "mde": 1.021},
    "S2": {"n": 2, "d": 6, "fraction": 0.3, "epsilon": 1.0, "replace": True, "r": 2,
           "p_a": (0.0551, 0.0013), "p_c": (0.0369, 0.0011), "lb": (0.379, 0.037), "mde": 1.026},
    "S3": {"n": 3, "d": 6, "fraction": 0.5, "epsilon": 0.1, "replace": False, "r": 3,
           "p_a": (0.0511, 0.0012), "p_c": (0.0489, 0.0012), "lb": (0.025, 0.035), "mde": 0.126},
}


# ---------------------------------------------------------------- numeric computation (verbatim from the scratch file
# with sha256 3282d8d048052cb0d56555d0ca4ff5bbf58ebd2a2774ad6dd18a27dab4592c19, formulas only, no nvflare)
def lap_sf(x, b):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 1.0 - 0.5 * np.exp(x / b), 0.5 * np.exp(-x / b))


def surjections(n, k):
    return sum((-1) ** i * math.comb(k, i) * (k - i) ** n for i in range(k + 1))


def support_prob(values, special, n, eps_t, eps_q, replace, grid=3000, kmax=3000):
    d = len(values)
    b_t = GAMMA / eps_t
    b_q = GAMMA * 2.0 * n / eps_q
    u = (np.arange(grid) + 0.5) / grid
    z = np.where(u < 0.5, b_t * np.log(2 * u), -b_t * np.log(2 * (1 - u)))
    T = TAU + z
    q = np.clip(np.stack([lap_sf(T - v * GAMMA, b_q) for v in values], axis=0), 1e-300, 1.0)
    others = [i for i in range(d) if i not in special]
    k_s = len(special)
    total = np.zeros(grid)
    alive = np.ones(grid)
    for K in range(1, kmax + 1):
        un = (1 - q) ** (K - 1)
        a, v, w = 1 - un, un * q, un * (1 - q)
        dp = {(0, 0): np.ones(grid)}
        for i in others:
            new = {}
            for (cb, ca), p in dp.items():
                for (dc, dca, pr) in ((1, 0, a[i]), (0, 1, v[i]), (0, 0, w[i])):
                    key = (cb + dc, ca + dca)
                    new[key] = new.get(key, 0) + p * pr
            dp = new
        stop = np.zeros(grid)
        sel = np.zeros(grid)
        for states in itertools.product("avw", repeat=k_s):
            pr = np.ones(grid)
            sb = sa = 0
            for s, i in zip(states, special):
                pr = pr * {"a": a, "v": v, "w": w}[s][i]
                sb += s == "a"
                sa += s == "v"
            for (cb, ca), p in dp.items():
                nb, na = cb + sb, ca + sa
                if nb < n <= nb + na:
                    mass = p * pr
                    stop += mass
                    if "w" not in states:
                        m = nb + na
                        sel += mass * (surjections(n, k_s) / m ** n if replace else (1.0 / math.comb(m, n) if k_s == n else 0.0))
        total += sel
        alive -= stop
        if alive.max() < 1e-12:
            break
    return float(total.mean()), float(alive.max())


def split(eps, n):
    et = eps / (1 + (2 * n) ** (2 / 3))
    return et, eps - et


# ---------------------------------------------------------------- statistics
ND = NormalDist()
Z_TAIL = ND.inv_cdf(1 - ALPHA_TAIL)
Z_80 = ND.inv_cdf(0.8)


def chk(name, expected, observed, held):
    return {"check": name, "expected": expected, "observed": observed, "held": bool(held)}


def interval(k: int, n: int):
    from audit_nvflare_dp import cp_lower, cp_upper
    return cp_lower(k, n, ALPHA_TAIL), cp_upper(k, n, ALPHA_TAIL)


def lower_bound(k_a: int, n_a: int, k_c: int, n_c: int) -> dict:
    """Largest of four log ratios, floored at 0, with 1.25% per tail on each interval."""
    lo_a, hi_a = interval(k_a, n_a)
    lo_c, hi_c = interval(k_c, n_c)
    cand = {}
    cand["ln(lo_a/hi_c)"] = math.log(lo_a / hi_c) if lo_a > 0 and hi_c > 0 else 0.0
    cand["ln(lo_c/hi_a)"] = math.log(lo_c / hi_a) if lo_c > 0 and hi_a > 0 else 0.0
    cand["ln((1-hi_c)/(1-lo_a))"] = math.log((1 - hi_c) / (1 - lo_a)) if lo_a < 1 and hi_c < 1 else 0.0
    cand["ln((1-hi_a)/(1-lo_c))"] = math.log((1 - hi_a) / (1 - lo_c)) if lo_c < 1 and hi_a < 1 else 0.0
    return {"p_hat_a": k_a / n_a, "p_hat_c": k_c / n_c, "lo_a": lo_a, "hi_a": hi_a, "lo_c": lo_c, "hi_c": hi_c,
            "candidates": cand, "lb": max(0.0, *cand.values())}


def se_log_ratio(p_a: float, p_c: float, n: int) -> float:
    return math.sqrt((1 - p_a) / (n * p_a) + (1 - p_c) / (n * p_c))


def min_detectable(p_c: float, n: int, claim: float) -> dict:
    """Smallest event epsilon found with 80% power: e = claim + (z + z80) * SE(e), p_c held fixed."""
    e = claim
    for _ in range(500):
        p_a = min(p_c * math.exp(e), 0.999999)
        e = claim + (Z_TAIL + Z_80) * se_log_ratio(p_a, p_c, n)
    p_a = min(p_c * math.exp(e), 0.999999)
    return {"epsilon_80": e, "excess": e - claim, "excess_over_claim": (e - claim) / claim,
            "se_at_epsilon_80": se_log_ratio(p_a, p_c, n)}


def power_at(true_eps: float, p_c: float, n: int, claim: float) -> float:
    p_a = min(p_c * math.exp(true_eps), 0.999999)
    return ND.cdf((true_eps - claim) / se_log_ratio(p_a, p_c, n) - Z_TAIL)


def chi2_df4_sf(x: float) -> float:
    return math.exp(-x / 2) * (1 + x / 2)


def block_agreement(counts: list[int], n_block: int) -> dict:
    pooled = sum(counts) / (len(counts) * n_block)
    e1, e0 = n_block * pooled, n_block * (1 - pooled)
    x = sum((k - e1) ** 2 / e1 + ((n_block - k) - e0) ** 2 / e0 for k in counts)
    return {"block_counts": counts, "chi2": x, "df": len(counts) - 1, "p_value": chi2_df4_sf(x)}


# ---------------------------------------------------------------- real runs
def load_nvflare():
    import types
    from nvflare.apis.dxo import DXO, DataKind
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.shareable import Shareable
    from nvflare.app_common.filters.svt_privacy import SVTPrivacy
    return types.SimpleNamespace(DXO=DXO, DataKind=DataKind, FLContext=FLContext, Shareable=Shareable, SVTPrivacy=SVTPrivacy,
                                 version=importlib.metadata.version("nvflare"))


def run_block(F, kw: dict, base: np.ndarray, n_set: int, runs: int, seed: int):
    np.random.seed(seed)
    count, passes, first_ledger, done = 0, 0, None, 0
    sh, fl = F.Shareable(), F.FLContext()
    while done < runs:
        filt = F.SVTPrivacy(**kw)
        m = min(CHUNK, runs - done)
        for _ in range(m):
            out = filt.process_dxo(F.DXO(data_kind=F.DataKind.WEIGHT_DIFF, data={"w": base}), sh, fl).data["w"]
            if (out[:n_set] != 0.0).all() and (out[n_set:] == 0.0).all():
                count += 1
        led = filt.get_privacy_ledger()
        passes += sum(e["acceptance_passes"] for e in led)
        if first_ledger is None:
            first_ledger = led[0]
        done += m
    return count, passes, first_ledger


def row_s(F, rid: str, cfg: dict, runs: int, offset: int, quick: bool) -> dict:
    n, d = cfg["n"], cfg["d"]
    kw = {"fraction": cfg["fraction"], "epsilon": cfg["epsilon"], "replace": cfg["replace"]}
    world_a = np.array([GAMMA] * n + [0.0] * (d - n))
    world_c = np.array([0.0] * n + [GAMMA] * (d - n))
    worlds, led0 = {}, None
    for w, base in ((0, world_a), (1, world_c)):
        counts, passes = [], 0
        for b in range(BLOCKS):
            seed = BASE_SEED + 10_000 * cfg["r"] + 100 * w + b + offset
            k, ps, led = run_block(F, kw, base, n, runs, seed)
            counts.append(k)
            passes += ps
            led0 = led0 or led
        worlds["a" if w == 0 else "c"] = {"counts": counts, "passes": passes, "calls": BLOCKS * runs}
    N = BLOCKS * runs
    k_a, k_c = sum(worlds["a"]["counts"]), sum(worlds["c"]["counts"])
    lb = lower_bound(k_a, N, k_c, N)
    claim = led0["epsilon_threshold"] + led0["epsilon_query"]
    # numeric computation for reference and for the power numbers
    et, eq = split(cfg["epsilon"], n)
    pa_an, _ = support_prob([1] * n + [0] * (d - n), tuple(range(n)), n, et, eq, cfg["replace"])
    pc_an, _ = support_prob([0] * n + [1] * (d - n), tuple(range(n)), n, et, eq, cfg["replace"])
    log_ratio = math.log(pa_an / pc_an)
    se_true = se_log_ratio(pa_an, pc_an, N)
    mde = min_detectable(pc_an, N, claim)
    ba, bc = block_agreement(worlds["a"]["counts"], runs), block_agreement(worlds["c"]["counts"], runs)
    control_claim = 0.1 * claim
    pr_a, tol_a = cfg["p_a"]
    pr_c, tol_c = cfg["p_c"]
    pr_lb, tol_lb = cfg["lb"]
    checks = [
        chk("ledger n_upload equals n and epsilon_threshold + epsilon_query equals epsilon", [n, cfg["epsilon"]],
            [led0["n_upload"], claim], led0["n_upload"] == n and abs(claim / cfg["epsilon"] - 1) < 1e-12),
        chk("p_hat world a within tolerance of the prediction", f"{pr_a} +- {tol_a}", lb["p_hat_a"], abs(lb["p_hat_a"] - pr_a) <= tol_a),
        chk("p_hat world c within tolerance of the prediction", f"{pr_c} +- {tol_c}", lb["p_hat_c"], abs(lb["p_hat_c"] - pr_c) <= tol_c),
        chk("numeric computation agrees with the prediction table", "abs diff at most 5e-4 in both worlds",
            [pa_an, pc_an], abs(pa_an - pr_a) <= 5e-4 and abs(pc_an - pr_c) <= 5e-4),
        chk("LB within tolerance of the predicted LB", f"{pr_lb} +- {tol_lb}", lb["lb"], abs(lb["lb"] - pr_lb) <= tol_lb),
        chk("LB at most the claim (verdict holds)", f"<= {claim}", lb["lb"], lb["lb"] <= claim),
        chk("positive control: LB above 0.1 times the claim", f"> {control_claim}", lb["lb"], lb["lb"] > control_claim),
        chk("block agreement world a not rejected at 0.01", "p >= 0.01", ba["p_value"], ba["p_value"] >= 0.01),
        chk("block agreement world c not rejected at 0.01", "p >= 0.01", bc["p_value"], bc["p_value"] >= 0.01),
        chk("smallest epsilon found with 80% power matches the pre-registration", f"{cfg['mde']} +- 0.002", mde["epsilon_80"],
            abs(mde["epsilon_80"] - cfg["mde"]) <= 0.002 or quick),
        chk("informative: excess at 80% power at most half of the claim", f"<= {0.5 * claim}", mde["excess"], mde["excess"] <= 0.5 * claim or quick),
    ]
    return {
        "id": rid,
        "configuration": {**kw, "n": n, "d": d, "gamma": GAMMA, "world_a": "first n entries gamma, rest 0",
                          "world_c": "first n entries 0, rest gamma", "blocks": BLOCKS, "runs_per_block": runs,
                          "seed_rule": "20261126 + 10000 r + 100 w + b" + (f" + {offset}" if offset else ""), "seed_offset": offset},
        "measured": {"claim": claim, "ledger_first_call": led0, "worlds": worlds, "interval_and_lb": lb,
                     "block_agreement": {"a": ba, "c": bc}, "numeric_prediction": {"p_a": pa_an, "p_c": pc_an, "log_ratio": log_ratio,
                     "log_ratio_over_claim": log_ratio / claim, "predicted_lb": log_ratio - Z_TAIL * se_true},
                     "power": {"min_detectable": mde, "control_claim": control_claim,
                               "control_power_at_numeric_truth": power_at(log_ratio, pc_an, N, control_claim),
                               "power_at_claim_numeric_truth": power_at(log_ratio, pc_an, N, claim)},
                     "mean_acceptance_passes_per_call": {"a": worlds["a"]["passes"] / worlds["a"]["calls"],
                                                         "c": worlds["c"]["passes"] / worlds["c"]["calls"]}},
        "selection_verdict": "holds" if lb["lb"] <= claim else "excess found",
        "excess_found": lb["lb"] > claim,
        "checks": checks,
        "prediction_held": all(c["held"] for c in checks),
    }


def row_s0() -> dict:
    out = {}
    for n_runs in (200_000,):
        lo, _ = interval(n_runs, n_runs)
        _, hi = interval(0, n_runs)
        out[str(n_runs)] = math.log(lo / hi)
    ceiling = out["200000"]
    from audit_nvflare_dp import cp_lower, cp_upper
    old = math.log(cp_lower(200_000, 200_000, 0.025) / cp_upper(0, 200_000, 0.025))
    checks = [chk("ceiling at 200,000 runs and 1.25% per tail", "10.73 +- 0.01", ceiling, abs(ceiling - 10.73) <= 0.01),
              chk("ceiling below the claim of 20", "< 20", ceiling, ceiling < 20.0)]
    return {"id": "S0", "configuration": {"claim": 20.0, "runs_per_input": 200_000, "alpha_tail": ALPHA_TAIL},
            "measured": {"ceiling_new_tails": ceiling, "ceiling_v0164_tails": old, "power_to_find_any_excess_over_20": 0.0},
            "checks": checks, "prediction_held": all(c["held"] for c in checks)}


def row_c(rows: list[dict]) -> dict:
    checks = []
    for r in rows:
        m = r["measured"]
        thr = m["power"]["control_claim"]
        checks.append(chk(f"{r['id']}: LB above {thr}", f"> {thr}", m["interval_and_lb"]["lb"], m["interval_and_lb"]["lb"] > thr))
    return {"id": "C", "configuration": "LB of S1, S2, S3 tested against 0.1 times each claim, no new runs",
            "measured": {r["id"]: {"control_claim": r["measured"]["power"]["control_claim"],
                                   "power_at_numeric_truth": r["measured"]["power"]["control_power_at_numeric_truth"]} for r in rows},
            "checks": checks, "prediction_held": all(c["held"] for c in checks)}


# ---------------------------------------------------------------- version identity
def child_main(unpacked: str) -> int:
    """Runs under python -I in a separate process. Prints one JSON line."""
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, unpacked)
    import nvflare
    assert str(Path(nvflare.__file__).resolve()).startswith(str(Path(unpacked).resolve())), nvflare.__file__
    from nvflare.apis.dxo import DXO, DataKind
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.shareable import Shareable
    from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
    from nvflare.app_common.filters.svt_privacy import SVTPrivacy
    g = 1e-5
    data = np.random.RandomState(7).uniform(-2 * g, 2 * g, 64)
    np.random.seed(20261126)
    filt = SVTPrivacy(fraction=0.25, epsilon=1.0)
    out = filt.process_dxo(DXO(data_kind=DataKind.WEIGHT_DIFF, data={"w": data}), Shareable(), FLContext())
    svt_digest = hashlib.sha256(out.data["w"].tobytes() + json.dumps(filt.get_privacy_ledger()[0], sort_keys=True).encode()).hexdigest()
    pdata = np.random.RandomState(8).standard_normal(10_000)
    pout = PercentilePrivacy(percentile=10, gamma=0.01).process_dxo(DXO(data_kind=DataKind.WEIGHT_DIFF, data={"w": pdata}), Shareable(), FLContext())
    pp_digest = hashlib.sha256(pout.data["w"].tobytes()).hexdigest()
    print(json.dumps({"nvflare_file": nvflare.__file__, "numpy": np.__version__, "svt_digest": svt_digest, "pp_digest": pp_digest}))
    return 0


def row_v(skip: bool) -> dict:
    if skip:
        return {"id": "V", "skipped": True, "checks": [], "prediction_held": None}
    tmp = Path(tempfile.mkdtemp(prefix="fdl_nvsel_"))
    per = {}
    try:
        for v in VERSIONS:
            wheels = tmp / "wheels" / v
            wheels.mkdir(parents=True)
            r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "-q", f"nvflare=={v}", "-d", str(wheels)],
                               capture_output=True, text=True)
            whl = list(wheels.glob("*.whl"))
            if r.returncode != 0 or len(whl) != 1:
                per[v] = {"error": (r.stderr or "no wheel")[-300:]}
                continue
            dest = tmp / "unpacked" / v
            with zipfile.ZipFile(whl[0]) as z:
                z.extractall(dest)
                zip_hash = {n: hashlib.sha256(z.read(f"nvflare/app_common/filters/{n}")).hexdigest() for n in ("svt_privacy.py", "percentile_privacy.py")}
            files = {n: hashlib.sha256((dest / "nvflare/app_common/filters" / n).read_bytes()).hexdigest() for n in ("svt_privacy.py", "percentile_privacy.py")}
            c = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--child", str(dest)], capture_output=True, text=True, timeout=300)
            call = json.loads(c.stdout.strip().splitlines()[-1]) if c.returncode == 0 and c.stdout.strip() else {"error": (c.stderr or "no output")[-400:]}
            per[v] = {"wheel": whl[0].name, "wheel_bytes": whl[0].stat().st_size, "svt_sha256": files["svt_privacy.py"],
                      "pp_sha256": files["percentile_privacy.py"], "zip_member_hashes_equal_unpacked": zip_hash == files, "seeded_call": call}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all("svt_sha256" in per.get(v, {}) for v in VERSIONS)
    checks = []
    if ok:
        checks = [
            chk("svt_privacy.py equal across the three wheels and equal to the 2.9.1 hash of v0.16.4", SVT_SHA_291,
                {v: per[v]["svt_sha256"] for v in VERSIONS}, all(per[v]["svt_sha256"] == SVT_SHA_291 for v in VERSIONS)),
            chk("percentile_privacy.py equal across the three wheels and equal to the 2.9.1 hash of v0.16.4", PP_SHA_291,
                {v: per[v]["pp_sha256"] for v in VERSIONS}, all(per[v]["pp_sha256"] == PP_SHA_291 for v in VERSIONS)),
            chk("seeded SVTPrivacy call gives one digest in all versions", "1 distinct",
                sorted({per[v]["seeded_call"].get("svt_digest", f"error-{v}") for v in VERSIONS}),
                len({per[v]["seeded_call"].get("svt_digest", f"error-{v}") for v in VERSIONS}) == 1),
            chk("seeded PercentilePrivacy call gives one digest in all versions", "1 distinct",
                sorted({per[v]["seeded_call"].get("pp_digest", f"error-{v}") for v in VERSIONS}),
                len({per[v]["seeded_call"].get("pp_digest", f"error-{v}") for v in VERSIONS}) == 1),
        ]
    else:
        checks = [chk("all three wheels downloaded and unpacked", "3 wheels", per, False)]
    return {"id": "V", "configuration": {"versions": list(VERSIONS), "method": "pip download --no-deps, unzip into a temporary directory, run python -I in a child"},
            "measured": per, "checks": checks, "prediction_held": all(c["held"] for c in checks)}


# ---------------------------------------------------------------- main
def enc(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def main() -> int:
    if "--child" in sys.argv:
        return child_main(sys.argv[sys.argv.index("--child") + 1])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true", help="fewer runs, for tests, requires --out")
    ap.add_argument("--skip-versions", action="store_true")
    args = ap.parse_args()
    if args.quick and not args.out:
        ap.error("--quick requires --out so that the committed evidence is not overwritten")
    logging.disable(logging.CRITICAL)
    runs = 2_000 if args.quick else BLOCK_RUNS
    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    F = load_nvflare()
    t0 = time.time()
    doc = {
        "schema": "feddisclosure.nvflare_selection.v1", "quick": args.quick, "preregistration": PREREG_FILE,
        "preregistration_commit": PREREG_COMMIT, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"nvflare": F.version, "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform()},
        "parameters": {"base_seed": BASE_SEED, "blocks": BLOCKS, "runs_per_block": runs, "alpha_tail": ALPHA_TAIL, "z_tail": Z_TAIL, "z_80": Z_80},
        "deviations": DEVIATIONS, "rows": [],
    }

    def save():
        rows = doc["rows"]
        checks = [(r["id"], c) for r in rows for c in r.get("checks", [])]
        doc["summary"] = {"rows_total": len(rows), "rows_held": sum(1 for r in rows if r.get("prediction_held")),
                          "rows_failed": [r["id"] for r in rows if r.get("prediction_held") is False],
                          "checks_total": len(checks), "checks_held": sum(1 for _, c in checks if c["held"]),
                          "checks_failed": [{"row": rid, **c} for rid, c in checks if not c["held"]],
                          "excess_found_rows": [r["id"] for r in rows if r.get("excess_found")]}
        doc["runtime_seconds"] = round(time.time() - t0, 1)
        (out_dir / OUT_NAME).write_text(json.dumps(doc, indent=2, default=enc) + "\n")

    doc["rows"].append(row_s0())
    for rid, cfg in ROWS.items():
        try:
            doc["rows"].append(row_s(F, rid, cfg, runs, 0, args.quick))
        except Exception as e:  # a failed row stays in the record
            doc["rows"].append({"id": rid, "measured": {"exception": f"{type(e).__name__}: {e}"[:300]},
                                "checks": [chk("row ran", "no exception", str(e)[:200], False)], "prediction_held": False})
        print(f"{rid} done {time.time() - t0:.0f}s held={doc['rows'][-1]['prediction_held']}", flush=True)
        save()
    doc["rows"].append(row_c([r for r in doc["rows"] if r["id"] in ROWS and "interval_and_lb" in r.get("measured", {})]))
    save()
    # Replication, only if some LB is above its claim (stopping rule of the pre-registration).
    flagged = [r["id"] for r in doc["rows"] if r.get("excess_found")]
    doc["replication"] = []
    for rid in flagged:
        rep = row_s(F, rid, ROWS[rid], runs, REPLICATION_OFFSET, args.quick)
        rep["id"] = rid + "-replication"
        doc["replication"].append(rep)
        print(f"{rep['id']} done {time.time() - t0:.0f}s excess={rep['excess_found']}", flush=True)
        save()
    t_mc = time.time() - t0
    doc["rows"].append(row_v(args.skip_versions))
    print(f"V done {time.time() - t0:.0f}s held={doc['rows'][-1]['prediction_held']}", flush=True)
    total = time.time() - t0
    doc["rows"].append({"id": "T", "measured": {"monte_carlo_seconds": round(t_mc, 1), "total_seconds": round(total, 1)},
                        "checks": [chk("whole script under 900 seconds", "< 900", round(total, 1), total < 900),
                                   chk("Monte Carlo part between 360 and 600 seconds", "360 to 600", round(t_mc, 1), 360 <= t_mc <= 600)],
                        "prediction_held": total < 900 and 360 <= t_mc <= 600})
    save()
    s = doc["summary"]
    print(f"wrote {out_dir / OUT_NAME} in {doc['runtime_seconds']}s, {s['rows_held']} of {s['rows_total']} rows held, "
          f"{s['checks_held']} of {s['checks_total']} checks held, excess rows {s['excess_found_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
