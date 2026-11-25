#!/usr/bin/env python3
"""Events check of NVIDIA FLARE 2.9.1 SVTPrivacy: other events, larger sizes, release values (v0.16.8).

Written after the pre-registration commit a1d1ea5ecb8c774e68536b52d2d94596c56eb419
(docs/preregistration/v0.16-nvflare-events.md). Every row, number and threshold comes from that file.
The script changes none of them. Anything that differs goes in DEVIATIONS below, is copied to the JSON
evidence, and is appended to the "Deviation log" of the pre-registration.

Rows
  K0  the ceiling of the check at 500,000 runs per world (no filter call)
  E1 to E6  real process_dxo calls, 5 blocks of 100,000 per world, 95% family lower bound
      (Clopper-Pearson, 1.25% per tail)
  C   positive control (LB against 0.1 times the claim)
  P   hash of the installed svt_privacy.py
  T   run time

Usage: python scripts/audit_nvflare_events.py [--out DIR] [--quick]
Full mode writes reports/results/privacy/v0160/nvflare_events/nvflare_events.json.
Quick mode (few runs, for tests) requires --out.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PREREG_COMMIT = "a1d1ea5ecb8c774e68536b52d2d94596c56eb419"
PREREG_FILE = "docs/preregistration/v0.16-nvflare-events.md"
GAMMA = 1e-5
TAU = 1e-6
BASE_SEED = 20261201
REPLICATION_OFFSET = 5_000_000
ALPHA_TAIL = 0.0125
BLOCKS, BLOCK_RUNS = 5, 100_000
CHUNK = 10_000
OUT_NAME = "nvflare_events.json"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "results" / "privacy" / "v0160" / "nvflare_events"
SVT_SHA_291 = "c543bbea2ce3bfe2e35a1d85a26acac8dffc0e95024e777010cdbf0cd0f92858"

# Deviations from the pre-registration. Each entry is a dict with "date", "item" and "reason".
# Empty means none were found when the script was written. Entries added later are copied to the
# pre-registration Deviation log by hand.
DEVIATIONS: list[dict] = []

# From the pre-registration "Predictions" and "Power" tables.
# kind: "set" (nonzero entries are exactly 0..n-1), "atleast" (at least h of entries 0..n-1 nonzero),
# "allpos" (exactly n entries nonzero and all positive), "joint" (entries 0..n-1 positive, the rest 0).
ROWS = {
    "E1": {"r": 1, "d": 10, "fraction": 0.4, "n": 4, "epsilon": 1.0, "replace": False, "kind": "set", "h": 4,
           "a": ([1.0] * 4 + [0.0] * 6), "c": ([0.0] * 4 + [1.0] * 6), "claim_from": "sel", "claim": 1.0,
           "p_a": (0.00620, 0.00044), "p_c": (0.00367, 0.00034), "lr": (0.525, 0.118), "lb": (0.459, 0.118), "mde": 1.083},
    "E2": {"r": 2, "d": 24, "fraction": 0.25, "n": 6, "epsilon": 1.0, "replace": True, "kind": "atleast", "h": 4,
           "a": ([1.0] * 6 + [0.0] * 18), "c": ([0.0] * 6 + [1.0] * 18), "claim_from": "sel", "claim": 1.0,
           "p_a": (0.00857, 0.00052), "p_c": (0.00634, 0.00045), "lr": (0.302, 0.093), "lb": (0.250, 0.093), "mde": 1.063},
    "E3": {"r": 3, "d": 10, "fraction": 0.4, "n": 4, "epsilon": 1.0, "replace": False, "kind": "set", "h": 4,
           "a": ([0.8] * 4 + [0.2] * 6), "c": ([0.2] * 4 + [0.8] * 6), "claim_from": "sel", "claim": 1.0,
           "p_a": (0.00558, 0.00042), "p_c": (0.00407, 0.00036), "lr": (0.315, 0.116), "lb": (0.250, 0.116), "mde": 1.079},
    "E4": {"r": 4, "d": 40, "fraction": 0.25, "n": 10, "epsilon": 1.0, "replace": False, "kind": "atleast", "h": 6,
           "a": ([1.0] * 10 + [0.0] * 30), "c": ([0.0] * 10 + [1.0] * 30), "claim_from": "sel", "claim": 1.0,
           "p_a": (0.00871, 0.00053), "p_c": (0.00627, 0.00045), "lr": (0.330, 0.093), "lb": (0.277, 0.093), "mde": 1.064},
    "E5": {"r": 5, "d": 12, "fraction": 0.25, "n": 3, "epsilon": 1.0, "replace": False, "kind": "allpos", "h": 3,
           "a": [1.0] * 12, "c": [-1.0] * 12, "claim_from": "rel", "claim": 0.3,
           "p_a": (0.1442, 0.0020), "p_c": (0.1076, 0.0018), "lr": (0.2929, 0.021), "lb": (0.281, 0.021), "mde": 0.316},
    "E6": {"r": 6, "d": 8, "fraction": 0.25, "n": 2, "epsilon": 1.0, "replace": False, "kind": "joint", "h": 2,
           "a": ([1.0] * 2 + [0.0] * 6), "c": ([0.0] * 2 + [-1.0] * 6), "claim_from": "total", "claim": 1.2,
           "p_a": (0.01252, 0.00063), "p_c": (0.00704, 0.00047), "lr": (0.575, 0.084), "lb": (0.528, 0.084), "mde": 1.259},
}


# ---------------------------------------------------------------- numeric computation (verbatim from the scratch files
# numeric.py sha256 e116346a19fc218c5369d9f7b8598661984cee21bfeac8d7abc5e0c28d447158 and numeric2.py sha256
# af1066709a8204575d23afffeff3cd18d8a5b4adad23995ce6df15f1e88d45bf, formulas only, no nvflare)
def lap_sf(x, b):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 1.0 - 0.5 * np.exp(x / b), 0.5 * np.exp(-x / b))


def split(eps, n):
    et = eps / (1 + (2 * n) ** (2 / 3))
    return et, eps - et


def p_pos(a):
    return 1 - 0.5 * math.exp(-a)


def g_table(d, n, h, replace):
    """g[m][s] = P(at least h special entries drawn | m accepted entries of which s special, n draws)."""
    g = np.zeros((d + 1, n + 1))
    for m in range(n, d + 1):
        for s in range(0, min(n, m) + 1):
            if replace:
                tot = 0.0
                for x in range(h, s + 1):
                    # P(hit set within special is exactly a given x-set) summed over C(s,x) sets
                    p_set = sum((-1) ** i * math.comb(x, i) * ((m - (s - x) - i) / m) ** n for i in range(x + 1))
                    tot += math.comb(s, x) * p_set
                g[m][s] = tot
            else:
                g[m][s] = sum(math.comb(s, x) * math.comb(m - s, n - x) for x in range(h, min(s, n) + 1) if n - x <= m - s) / math.comb(m, n)
    return g


def count_prob(vs, vo, n, d, h, eps_t, eps_q, replace, grid=3000, kmax=3000):
    r = d - n
    b_t = GAMMA / eps_t
    b_q = GAMMA * 2.0 * n / eps_q
    u = (np.arange(grid) + 0.5) / grid
    z = np.where(u < 0.5, b_t * np.log(2 * u), -b_t * np.log(2 * (1 - u)))
    T = TAU + z
    qs = np.clip(lap_sf(T - vs * GAMMA, b_q), 1e-300, 1.0)
    qo = np.clip(lap_sf(T - vo * GAMMA, b_q), 1e-300, 1.0)
    g = g_table(d, n, h, replace)
    total = np.zeros(grid)

    def pows(x, k):
        out = [np.ones(grid)]
        for _ in range(k):
            out.append(out[-1] * x)
        return out

    def p_lt(K):
        ps, po = 1 - (1 - qs) ** K, 1 - (1 - qo) ** K
        pS, qS, pO, qO = pows(ps, n), pows(1 - ps, n), pows(po, r), pows(1 - po, r)
        res = np.zeros(grid)
        for i in range(n + 1):
            for j in range(r + 1):
                if i + j < n:
                    res += math.comb(n, i) * pS[i] * qS[n - i] * math.comb(r, j) * pO[j] * qO[r - j]
        return res

    for K in range(1, kmax + 1):
        un_s, un_o = (1 - qs) ** (K - 1), (1 - qo) ** (K - 1)
        a_s, v_s, w_s = 1 - un_s, un_s * qs, un_s * (1 - qs)
        a_o, v_o, w_o = 1 - un_o, un_o * qo, un_o * (1 - qo)
        As, Vs, Ws = pows(a_s, n), pows(v_s, n), pows(w_s, n)
        Ao, Vo, Wo = pows(a_o, r), pows(v_o, r), pows(w_o, r)
        PO = {}
        for cb in range(r + 1):
            for ca in range(r - cb + 1):
                PO[(cb, ca)] = math.comb(r, cb) * math.comb(r - cb, ca) * Ao[cb] * Vo[ca] * Wo[r - cb - ca]
        sel = np.zeros(grid)
        for ja in range(n + 1):
            for jv in range(n - ja + 1):
                jw = n - ja - jv
                ps = math.factorial(n) // (math.factorial(ja) * math.factorial(jv) * math.factorial(jw)) * As[ja] * Vs[jv] * Ws[jw]
                s = ja + jv
                for (cb, ca), po in PO.items():
                    nb, na = ja + cb, jv + ca
                    if nb < n <= nb + na:
                        gg = g[nb + na][s]
                        if gg != 0.0:
                            sel += ps * po * gg
        total += sel
        lt = p_lt(K)
        if lt.max() < 1e-12:
            break
    return float(total.mean()), float(lt.max())


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


def event_test(kind: str, n: int, h: int):
    """Returns a function of the output vector that says whether the event happened."""
    if kind == "set":
        return lambda out: bool((out[:n] != 0.0).all() and (out[n:] == 0.0).all())
    if kind == "atleast":
        return lambda out: int(np.count_nonzero(out[:n])) >= h
    if kind == "allpos":
        def f(out):
            nz = out != 0.0
            return int(nz.sum()) == n and bool((out[nz] > 0.0).all())
        return f
    if kind == "joint":
        return lambda out: bool((out[:n] > 0.0).all() and (out[n:] == 0.0).all())
    raise ValueError(kind)


def run_block(F, kw: dict, base: np.ndarray, test, runs: int, seed: int):
    np.random.seed(seed)
    count, passes, first_ledger, done = 0, 0, None, 0
    sh, fl = F.Shareable(), F.FLContext()
    while done < runs:
        filt = F.SVTPrivacy(**kw)
        m = min(CHUNK, runs - done)
        for _ in range(m):
            out = filt.process_dxo(F.DXO(data_kind=F.DataKind.WEIGHT_DIFF, data={"w": base}), sh, fl).data["w"]
            if test(out):
                count += 1
        led = filt.get_privacy_ledger()
        passes += sum(e["acceptance_passes"] for e in led)
        if first_ledger is None:
            first_ledger = led[0]
        done += m
    return count, passes, first_ledger


def numeric_probs(cfg: dict) -> tuple[float, float]:
    """Predicted event probabilities in worlds a and c: numeric computation or closed form."""
    n, d = cfg["n"], cfg["d"]
    kind = cfg["kind"]
    if kind == "allpos":
        a = cfg["claim"] / (2 * n)
        return p_pos(a) ** n, (1 - p_pos(a)) ** n
    et, eq = split(cfg["epsilon"], n)
    va, vc = cfg["a"], cfg["c"]
    # worlds are two groups of equal entries: entries 0..n-1 and the rest, in absolute value for the selection step
    pa = count_prob(abs(va[0]), abs(va[n]), n, d, cfg["h"], et, eq, cfg["replace"])[0]
    pc = count_prob(abs(vc[0]), abs(vc[n]), n, d, cfg["h"], et, eq, cfg["replace"])[0]
    if kind == "joint":
        a = (n * 0.1) / (2 * n)  # epsilon_release = n * noise_var with the default noise_var 0.1
        return pa * p_pos(a) ** n, pc * 0.5 ** n
    return pa, pc


def row_e(F, rid: str, cfg: dict, runs: int, offset: int, quick: bool) -> dict:
    n, d = cfg["n"], cfg["d"]
    kw = {"fraction": cfg["fraction"], "epsilon": cfg["epsilon"], "replace": cfg["replace"]}
    test = event_test(cfg["kind"], n, cfg["h"])
    worlds, led0 = {}, None
    for w, key in ((0, "a"), (1, "c")):
        base = np.array(cfg[key], dtype=np.float64) * GAMMA
        counts, passes = [], 0
        for b in range(BLOCKS):
            seed = BASE_SEED + 10_000 * cfg["r"] + 100 * w + b + offset
            k, ps, led = run_block(F, kw, base, test, runs, seed)
            counts.append(k)
            passes += ps
            led0 = led0 or led
        worlds[key] = {"counts": counts, "passes": passes, "calls": BLOCKS * runs}
    N = BLOCKS * runs
    k_a, k_c = sum(worlds["a"]["counts"]), sum(worlds["c"]["counts"])
    lb = lower_bound(k_a, N, k_c, N)
    claim = {"sel": led0["epsilon_threshold"] + led0["epsilon_query"], "rel": led0["epsilon_release"],
             "total": led0["epsilon_total"]}[cfg["claim_from"]]
    pa_an, pc_an = numeric_probs(cfg)
    log_ratio = math.log(pa_an / pc_an)
    se_true = se_log_ratio(pa_an, pc_an, N)
    mde = min_detectable(pc_an, N, claim)
    ba, bc = block_agreement(worlds["a"]["counts"], runs), block_agreement(worlds["c"]["counts"], runs)
    control_claim = 0.1 * claim
    pr_a, tol_a = cfg["p_a"]
    pr_c, tol_c = cfg["p_c"]
    pr_lr, tol_lr = cfg["lr"]
    pr_lb, tol_lb = cfg["lb"]
    lr_hat = math.log(lb["p_hat_a"] / lb["p_hat_c"]) if lb["p_hat_a"] > 0 and lb["p_hat_c"] > 0 else float("nan")
    rel = lambda x, y: abs(x / y - 1)
    ledger_ok = (led0["n_upload"] == n and abs(claim / cfg["claim"] - 1) < 1e-9 and
                 abs(led0["epsilon_total"] - (led0["epsilon_threshold"] + led0["epsilon_query"] + led0["epsilon_release"])) < 1e-9 and
                 abs(led0["epsilon_threshold"] + led0["epsilon_query"] - cfg["epsilon"]) < 1e-9)
    checks = [
        chk("ledger n_upload, claim and epsilon_total as pre-registered", [n, cfg["claim"]], [led0["n_upload"], claim], ledger_ok),
        chk("p_hat world a within tolerance of the prediction", f"{pr_a} +- {tol_a}", lb["p_hat_a"], abs(lb["p_hat_a"] - pr_a) <= tol_a),
        chk("p_hat world c within tolerance of the prediction", f"{pr_c} +- {tol_c}", lb["p_hat_c"], abs(lb["p_hat_c"] - pr_c) <= tol_c),
        chk("numeric computation agrees with the prediction table to 1 percent", "relative difference at most 0.01 in both worlds",
            [pa_an, pc_an], rel(pa_an, pr_a) <= 0.01 and rel(pc_an, pr_c) <= 0.01),
        chk("log ratio point estimate within 4 SE of the prediction", f"{pr_lr} +- {tol_lr}", lr_hat, abs(lr_hat - pr_lr) <= tol_lr),
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
        "configuration": {**kw, "n": n, "d": d, "gamma": GAMMA, "event": cfg["kind"], "h": cfg["h"],
                          "world_a_in_gamma": cfg["a"], "world_c_in_gamma": cfg["c"], "blocks": BLOCKS, "runs_per_block": runs,
                          "seed_rule": f"{BASE_SEED} + 10000 r + 100 w + b" + (f" + {offset}" if offset else ""), "seed_offset": offset},
        "measured": {"claim": claim, "ledger_first_call": led0, "worlds": worlds, "interval_and_lb": lb, "log_ratio_hat": lr_hat,
                     "block_agreement": {"a": ba, "c": bc},
                     "numeric_prediction": {"p_a": pa_an, "p_c": pc_an, "log_ratio": log_ratio, "log_ratio_over_claim": log_ratio / claim,
                                            "predicted_lb": log_ratio - Z_TAIL * se_true},
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


def row_k0() -> dict:
    from audit_nvflare_dp import cp_lower, cp_upper
    N = BLOCKS * BLOCK_RUNS
    ceiling = math.log(cp_lower(N, N, ALPHA_TAIL) / cp_upper(0, N, ALPHA_TAIL))
    top = max(c["claim"] for c in ROWS.values())
    checks = [chk("ceiling at 500,000 runs per world and 1.25% per tail", "11.64 +- 0.01", ceiling, abs(ceiling - 11.64) <= 0.01),
              chk("ceiling above every claim", f"> {top}", ceiling, ceiling > top)]
    return {"id": "K0", "configuration": {"runs_per_world": N, "alpha_tail": ALPHA_TAIL},
            "measured": {"ceiling": ceiling, "largest_claim": top}, "checks": checks, "prediction_held": all(c["held"] for c in checks)}


def row_c(rows: list[dict]) -> dict:
    checks = []
    for r in rows:
        m = r["measured"]
        thr = m["power"]["control_claim"]
        checks.append(chk(f"{r['id']}: LB above {thr}", f"> {thr}", m["interval_and_lb"]["lb"], m["interval_and_lb"]["lb"] > thr))
    return {"id": "C", "configuration": "LB of E1 to E6 tested against 0.1 times each claim, no new runs",
            "measured": {r["id"]: {"control_claim": r["measured"]["power"]["control_claim"],
                                   "power_at_numeric_truth": r["measured"]["power"]["control_power_at_numeric_truth"]} for r in rows},
            "checks": checks, "prediction_held": all(c["held"] for c in checks)}


def row_p() -> dict:
    import nvflare.app_common.filters.svt_privacy as mod
    digest = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    checks = [chk("installed svt_privacy.py has the 2.9.1 sha256", SVT_SHA_291, digest, digest == SVT_SHA_291)]
    return {"id": "P", "measured": {"file": mod.__file__, "sha256": digest}, "checks": checks, "prediction_held": all(c["held"] for c in checks)}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true", help="fewer runs, for tests, requires --out")
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
        "schema": "feddisclosure.nvflare_events.v1", "quick": args.quick, "preregistration": PREREG_FILE,
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

    doc["rows"].append(row_k0())
    for rid, cfg in ROWS.items():
        try:
            doc["rows"].append(row_e(F, rid, cfg, runs, 0, args.quick))
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
        rep = row_e(F, rid, ROWS[rid], runs, REPLICATION_OFFSET, args.quick)
        rep["id"] = rid + "-replication"
        doc["replication"].append(rep)
        print(f"{rep['id']} done {time.time() - t0:.0f}s excess={rep['excess_found']}", flush=True)
        save()
    t_mc = time.time() - t0
    doc["rows"].append(row_p())
    total = time.time() - t0
    doc["rows"].append({"id": "T", "measured": {"monte_carlo_seconds": round(t_mc, 1), "total_seconds": round(total, 1)},
                        "checks": [chk("whole script under 1500 seconds", "< 1500", round(total, 1), total < 1500),
                                   chk("Monte Carlo part between 800 and 1300 seconds", "800 to 1300", round(t_mc, 1), 800 <= t_mc <= 1300)],
                        "prediction_held": total < 1500 and 800 <= t_mc <= 1300})
    save()
    s = doc["summary"]
    print(f"wrote {out_dir / OUT_NAME} in {doc['runtime_seconds']}s, {s['rows_held']} of {s['rows_total']} rows held, "
          f"{s['checks_held']} of {s['checks_total']} checks held, excess rows {s['excess_found_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
