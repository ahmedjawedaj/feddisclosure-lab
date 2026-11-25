#!/usr/bin/env python3
"""Audit of NVIDIA FLARE 2.9.1 SVTPrivacy and PercentilePrivacy (v0.16 T10).

Runs the real classes from nvflare.app_common.filters in process with synthetic
updates and compares the mechanism that executes with the one that the
documented parameters and the accountant describe. See
docs/preregistration/v0.16-nvflare.md for every row, configuration, prediction
and tolerance. Nothing about a prediction is changed here.

The Clopper-Pearson helpers follow scripts/audit_flower_dp.py (cp_upper,
cp_lower, empirical bound at confidence 95%). They are re-implemented with a
pure Python incomplete beta function, so that this file needs only numpy and
nvflare. A test compares them with scipy when scipy is installed.

Randomness is observed by wrapping numpy.random.laplace (recording scale and
size and passing the call through) and the module function
_sample_partition_counts (recording its group_counts and passing through).

Usage: python scripts/audit_nvflare_dp.py [--out DIR] [--quick]
Full mode writes to reports/results/privacy/v0160/nvflare_dp_audit/nvflare-<version>/
unless --out is given. Quick mode requires --out.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

SEED = 20261125
ALPHA = 0.05
GAMMA = 1e-5  # SVTPrivacy default gamma
OUT_NAME = "nvflare_dp_audit.json"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PARENT = ROOT / "reports" / "results" / "privacy" / "v0160" / "nvflare_dp_audit"


# ---------------------------------------------------------------- analytic
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile of the Beta(a, b) law by bisection on the incomplete beta function."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-17 * max(hi, 1e-300):
            break
    return 0.5 * (lo + hi)


def cp_upper(k: int, n: int, a: float) -> float:
    return 1.0 if k >= n else beta_ppf(1 - a, k + 1, n - k)


def cp_lower(k: int, n: int, a: float) -> float:
    return 0.0 if k <= 0 else beta_ppf(a, k, n - k + 1)


def empirical_eps_lower(k_a: int, n_a: int, k_b: int, n_b: int) -> dict:
    """95% Clopper-Pearson lower bound on pure epsilon from one event.

    Each probability gets a two-sided 95% interval (a = ALPHA / 2 per tail).
    epsilon_lb = ln(p_lo_a / p_hi_b), and the larger of the two directions is taken.
    """
    a = ALPHA / 2
    lo_a, hi_a = cp_lower(k_a, n_a, a), cp_upper(k_a, n_a, a)
    lo_b, hi_b = cp_lower(k_b, n_b, a), cp_upper(k_b, n_b, a)
    d_ab = math.log(lo_a / hi_b) if lo_a > 0 else 0.0
    d_ba = math.log(lo_b / hi_a) if lo_b > 0 else 0.0
    return {
        "runs_world_a": n_a,
        "runs_world_b": n_b,
        "events_world_a": k_a,
        "events_world_b": k_b,
        "p_hat_a": k_a / n_a,
        "p_hat_b": k_b / n_b,
        "p_lower_a": lo_a,
        "p_upper_a": hi_a,
        "p_lower_b": lo_b,
        "p_upper_b": hi_b,
        "eps_lower_a_over_b": max(d_ab, 0.0),
        "eps_lower_b_over_a": max(d_ba, 0.0),
        "eps_lower_bound": max(d_ab, d_ba, 0.0),
        "confidence": 1 - ALPHA,
        "ceiling_at_this_run_count": math.log(cp_lower(n_a, n_a, a) / cp_upper(0, n_b, a)),
    }


def chk(name: str, expected, observed, held: bool) -> dict:
    return {"check": name, "expected": expected, "observed": observed, "held": bool(held)}


def rel(a: float, b: float) -> float:
    """Relative difference |a / b - 1|."""
    return abs(a / b - 1.0) if b else float("inf")


def exc_str(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:300]


# ---------------------------------------------------------------- loading
def load_nvflare():
    """Import the two filters and the objects needed to call them."""
    import nvflare
    from nvflare.apis.dxo import DXO, DataKind, MetaKey
    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.shareable import Shareable
    from nvflare.app_common.filters import percentile_privacy as pp_mod
    from nvflare.app_common.filters import svt_privacy as svt_mod

    try:
        version = importlib.metadata.version("nvflare")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(nvflare, "__version__", "unknown")
    return types.SimpleNamespace(
        version=version, DXO=DXO, DataKind=DataKind, MetaKey=MetaKey, EventType=EventType,
        FLContext=FLContext, Shareable=Shareable, svt_mod=svt_mod, pp_mod=pp_mod,
        SVTPrivacy=svt_mod.SVTPrivacy, PercentilePrivacy=pp_mod.PercentilePrivacy,
        svt_sha256=hashlib.sha256(Path(svt_mod.__file__).read_bytes()).hexdigest(),
        pp_sha256=hashlib.sha256(Path(pp_mod.__file__).read_bytes()).hexdigest(),
    )


# ---------------------------------------------------------------- helpers
def make_dxo(F, data: dict, steps=None):
    dxo = F.DXO(data_kind=F.DataKind.WEIGHT_DIFF, data=data)
    if steps is not None:
        dxo.set_meta_prop(F.MetaKey.NUM_STEPS_CURRENT_ROUND, steps)
    return dxo


def run_filter(F, filt, data: dict, steps=None):
    """One real process_dxo call. Returns the output data dict."""
    out = filt.process_dxo(make_dxo(F, data, steps), F.Shareable(), F.FLContext())
    return out.data, out


class Recorder:
    def __init__(self) -> None:
        self.laplace: list[dict] = []
        self.group_counts: list[list[int]] = []
        self.selected_counts: list[list[int]] = []


@contextlib.contextmanager
def record_randomness(F):
    """Wrap numpy.random.laplace and _sample_partition_counts, passing every call through."""
    rec = Recorder()
    orig_laplace = np.random.laplace
    orig_partition = F.svt_mod._sample_partition_counts

    def laplace(*args, **kwargs):
        scale = kwargs.get("scale", args[1] if len(args) > 1 else 1.0)
        size = kwargs.get("size", args[2] if len(args) > 2 else None)
        rec.laplace.append({"scale": float(scale), "size": None if size is None else int(size)})
        return orig_laplace(*args, **kwargs)

    def partition(group_counts, total_to_sample, replace):
        counts = orig_partition(group_counts, total_to_sample, replace)
        rec.group_counts.append([int(g) for g in group_counts])
        rec.selected_counts.append([int(c) for c in counts])
        return counts

    np.random.laplace = laplace
    F.svt_mod._sample_partition_counts = partition
    try:
        yield rec
    finally:
        np.random.laplace = orig_laplace
        F.svt_mod._sample_partition_counts = orig_partition


def expected_split(epsilon: float, n: int) -> tuple[float, float]:
    """Standard SVT split written out independently of the class."""
    eps_t = epsilon / (1.0 + (2.0 * n) ** (2.0 / 3.0))
    return eps_t, epsilon - eps_t


def make_row(row_id: str, configuration: dict, prediction: str, tolerance: str, measured: dict,
             checks: list[dict], note: str) -> dict:
    return {
        "id": row_id,
        "configuration": configuration,
        "prediction": prediction,
        "tolerance": tolerance,
        "measured": measured,
        "checks": checks,
        "prediction_held": bool(checks) and all(c["held"] for c in checks),
        "note": note,
    }


# ---------------------------------------------------------------- SVTPrivacy rows
def row_n1(F) -> dict:
    cells, checks = [], []
    k = 0
    for eps in (0.1, 1.0, 10.0, 100.0):
        for n in (1, 10, 100, 1000):
            k += 1
            np.random.seed(SEED + 100 + k)
            filt = F.SVTPrivacy(epsilon=eps)
            data = {"w": np.full(10 * n, 0.5 * GAMMA, dtype=np.float64)}
            run_filter(F, filt, data)
            led = filt.get_privacy_ledger()[-1]
            e_t, e_q = expected_split(eps, n)
            r_t, r_q = rel(led["epsilon_threshold"], e_t), rel(led["epsilon_query"], e_q)
            ok = led["n_upload"] == n and r_t <= 1e-12 and r_q <= 1e-12
            cells.append({
                "epsilon": eps, "n_upload_expected": n, "total_params": 10 * n,
                "n_upload": led["n_upload"],
                "epsilon_threshold": led["epsilon_threshold"], "epsilon_threshold_expected": e_t,
                "epsilon_query": led["epsilon_query"], "epsilon_query_expected": e_q,
                "rel_diff_threshold": r_t, "rel_diff_query": r_q,
                "epsilon_release": led["epsilon_release"], "epsilon_total": led["epsilon_total"],
                "acceptance_passes": led["acceptance_passes"],
            })
            checks.append(chk(f"epsilon {eps}, n_upload {n}: threshold and query split", "relative diff <= 1e-12",
                              {"rel_threshold": r_t, "rel_query": r_q, "n_upload": led["n_upload"]}, ok))
    return make_row(
        "N1",
        {"epsilon": [0.1, 1, 10, 100], "n_upload": [1, 10, 100, 1000], "fraction": 0.1, "entries": "0.5 gamma",
         "seed_base": SEED + 100},
        "Ledger epsilon_threshold equals epsilon / (1 + (2 n)^(2/3)) and epsilon_query equals epsilon minus that, in every cell.",
        "relative 1e-12", {"cells": cells}, checks,
        "Input values were not named in the pre-registration. All entries were set to 0.5 gamma.")


def row_n2(F) -> dict:
    eps, n, d = 1.0, 100, 1000
    np.random.seed(SEED + 200)
    filt = F.SVTPrivacy(fraction=0.1, epsilon=eps, noise_var=0.1)
    with record_randomness(F) as rec:
        run_filter(F, filt, {"w": np.full(d, GAMMA, dtype=np.float64)})
    led = filt.get_privacy_ledger()[-1]
    e_t, e_q = expected_split(eps, n)
    e_rel = n * 0.1
    exp_thr = GAMMA / e_t
    exp_qry = 2.0 * GAMMA * n / e_q
    exp_rel = 2.0 * GAMMA * n / e_rel
    calls = rec.laplace
    first, mid, last = calls[0], calls[1:-1], calls[-1]
    passes = led["acceptance_passes"]
    mid_scales = sorted({c["scale"] for c in mid})
    measured = {
        "n_upload": led["n_upload"], "acceptance_passes": passes, "laplace_calls": calls,
        "expected_threshold_scale": exp_thr, "expected_query_scale": exp_qry, "expected_release_scale": exp_rel,
        "expected_epsilon_release": e_rel,
    }
    checks = [
        chk("call structure: one scalar draw, one draw per pass, one release draw",
            {"calls": passes + 2, "first_size": None},
            {"calls": len(calls), "first_size": first["size"]},
            len(calls) == passes + 2 and first["size"] is None and all(c["size"] is not None for c in calls[1:])),
        chk("threshold draw scale gamma / epsilon_threshold", exp_thr, first["scale"], rel(first["scale"], exp_thr) <= 1e-12),
        chk("query draw scale 2 gamma n / epsilon_query in every pass", exp_qry, mid_scales,
            bool(mid) and all(rel(c["scale"], exp_qry) <= 1e-12 for c in mid)),
        chk("release draw scale 2 gamma n / epsilon_release", exp_rel, last["scale"], rel(last["scale"], exp_rel) <= 1e-12),
        chk("release draw size equals n_upload", n, last["size"], last["size"] == n),
    ]
    return make_row(
        "N2", {"d": d, "fraction": 0.1, "n_upload": n, "epsilon": eps, "noise_var": 0.1, "entries": "1e-5",
               "seed": SEED + 200},
        "The recorded Laplace scales are one scalar draw of scale gamma / epsilon_threshold, query draws of scale "
        "2 gamma n / epsilon_query and a release draw of scale 2 gamma n / epsilon_release.",
        "relative 1e-12", measured, checks, "Expected scales are computed from the configuration, not from the ledger.")


def row_n3(F) -> dict:
    d = 25_578
    data = {"w": np.full(d, 0.5 * GAMMA, dtype=np.float64)}
    np.random.seed(SEED + 300)
    one = F.SVTPrivacy()
    _, dxo = run_filter(F, one, data)
    led1 = one.get_privacy_ledger()[-1]
    meta = dxo.get_meta_prop("svt_privacy_accountant")
    exp_one = 0.1 + 0.1 * 2558
    passes = []
    ten = F.SVTPrivacy()
    for i in range(10):
        np.random.seed(SEED + 310 + i)
        run_filter(F, ten, data)
        passes.append(ten.get_privacy_ledger()[-1]["acceptance_passes"])
    spent10 = ten.get_privacy_spent()
    exp_ten = 10 * exp_one
    measured = {
        "d": d, "n_upload_one_call": led1["n_upload"], "ledger_one_call": led1,
        "epsilon_total_expected_one_call": exp_one, "cumulative_meta_one_call": meta,
        "privacy_spent_ten_calls": spent10, "epsilon_total_expected_ten_calls": exp_ten,
        "acceptance_passes_ten_calls": passes, "ledger_delta_values_ten_calls": sorted({e["delta"] for e in ten.get_privacy_ledger()}),
    }
    checks = [
        chk("n_upload is 2558", 2558, led1["n_upload"], led1["n_upload"] == 2558),
        chk("one call epsilon_total is 255.9", exp_one, led1["epsilon_total"], abs(led1["epsilon_total"] - exp_one) <= 1e-9),
        chk("ten calls get_privacy_spent is 2559.0", exp_ten, spent10, abs(spent10 - exp_ten) <= 1e-9),
        chk("delta is 0 in the ledger and in the cumulative meta", 0.0,
            {"ledger": led1["delta"], "meta": meta["delta"]}, led1["delta"] == 0.0 and meta["delta"] == 0.0),
    ]
    return make_row(
        "N3", {"d": d, "class_defaults": True, "calls": [1, 10], "entries": "0.5 gamma", "seed_base": SEED + 300},
        "Ledger epsilon_total for one call is 0.1 + 0.1 * 2558 = 255.9. After ten calls get_privacy_spent() is 2559.0. delta is 0.",
        "absolute 1e-9", measured, checks,
        "The ten calls use one filter instance with one seed per call. Entries were set to 0.5 gamma because the pre-registration named none.")


def row_n4(F) -> dict:
    d, n = 1000, 100
    np.random.seed(SEED + 400)
    filt = F.SVTPrivacy(fraction=0.1, epsilon=100.0)
    with record_randomness(F) as rec:
        run_filter(F, filt, {"w": np.full(d, GAMMA, dtype=np.float64)})
    led = filt.get_privacy_ledger()[-1]
    accepted = int(sum(rec.group_counts[0]))
    ratio = accepted / led["n_upload"]
    measured = {"n_upload": led["n_upload"], "accepted_before_selection": accepted, "ratio_to_n_upload": ratio,
                "acceptance_passes": led["acceptance_passes"], "group_counts": rec.group_counts[0]}
    checks = [chk("accepted entries at least 5 times n_upload", ">= 5", ratio, ratio >= 5.0 and led["n_upload"] == n)]
    return make_row(
        "N4", {"d": d, "n_upload": n, "epsilon": 100.0, "entries": "all equal to gamma", "seed": SEED + 400},
        "The number of accepted entries recorded before selection is at least 5 times n_upload. "
        "The standard sparse vector technique stops after c positive answers and here c is n_upload.",
        "ratio at least 5", measured, checks, "One call. Acceptance runs over a whole pass, so the count is not capped at n_upload.")


N5_CHILD = r"""
import json, sys, time
t0 = time.time()
import numpy as np
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.filters.svt_privacy import SVTPrivacy
t_import = time.time() - t0
eps, seed = float(sys.argv[1]), int(sys.argv[2])
np.random.seed(seed)
w = np.zeros(1000)
w[:3] = 1e-5
f = SVTPrivacy(fraction=0.1, epsilon=eps)
t1 = time.time()
f.process_dxo(DXO(data_kind=DataKind.WEIGHT_DIFF, data={"w": w}), Shareable(), FLContext())
print(json.dumps({"ledger": f.get_privacy_ledger()[-1], "import_seconds": t_import, "call_seconds": time.time() - t1}))
"""


def row_n5(F) -> dict:
    limit = 10.0
    cells, checks = [], []
    for i, eps in enumerate((0.1, 1e4, 1e6)):
        t0 = time.time()
        rec: dict = {"epsilon": eps, "seed": SEED + 500 + i, "timeout_seconds": limit}
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", N5_CHILD, repr(eps), str(SEED + 500 + i)],
                                  capture_output=True, text=True, timeout=limit)
            rec["elapsed_seconds"] = time.time() - t0
            if proc.returncode == 0:
                child = json.loads(proc.stdout.strip().splitlines()[-1])
                rec.update(outcome="returned", acceptance_passes=child["ledger"]["acceptance_passes"],
                           n_upload=child["ledger"]["n_upload"], import_seconds=child["import_seconds"],
                           call_seconds=child["call_seconds"], epsilon_total=child["ledger"]["epsilon_total"])
            else:
                rec.update(outcome="error", stderr_tail=proc.stderr[-300:])
        except subprocess.TimeoutExpired:
            rec.update(outcome="timed out", elapsed_seconds=time.time() - t0)
        cells.append(rec)
    c1, c2, c3 = cells
    checks = [
        chk("epsilon 0.1 returns with 1 acceptance pass", {"outcome": "returned", "passes": 1},
            {"outcome": c1["outcome"], "passes": c1.get("acceptance_passes")},
            c1["outcome"] == "returned" and c1.get("acceptance_passes") == 1),
        chk("epsilon 1e4 returns with 5 to 200 passes", {"outcome": "returned", "passes": "5..200"},
            {"outcome": c2["outcome"], "passes": c2.get("acceptance_passes")},
            c2["outcome"] == "returned" and 5 <= c2.get("acceptance_passes", -1) <= 200),
        chk("epsilon 1e6 does not return within 10 seconds", "timed out", c3["outcome"], c3["outcome"] == "timed out"),
    ]
    return make_row(
        "N5", {"d": 1000, "n_upload": 100, "entries": "three equal to gamma, rest 0", "epsilon": [0.1, 1e4, 1e6],
               "subprocess_timeout_seconds": limit, "seeds": [SEED + 500, SEED + 501, SEED + 502]},
        "At 0.1 the call returns with 1 acceptance pass. At 1e4 it returns with between 5 and 200 passes. At 1e6 it does not return within 10 seconds.",
        "as stated", {"cells": cells}, checks,
        "Each epsilon runs in a fresh python subprocess. The 10 second limit covers interpreter start and the nvflare import, whose time is recorded.")


def _run_world(F, filter_kwargs: dict, base: np.ndarray, runs: int, seed: int, event, chunk: int = 10_000):
    """Call the real process_dxo `runs` times and count events. Returns (count, first ledger entry)."""
    np.random.seed(seed)
    count, first_ledger = 0, None
    fl, sh = F.FLContext(), F.Shareable()
    done = 0
    while done < runs:
        filt = F.SVTPrivacy(**filter_kwargs)  # fresh instance per chunk keeps the ledger small
        for _ in range(min(chunk, runs - done)):
            out = filt.process_dxo(F.DXO(data_kind=F.DataKind.WEIGHT_DIFF, data={"w": base}), sh, fl)
            count += 1 if event(out.data["w"]) else 0
        if first_ledger is None:
            first_ledger = filt.get_privacy_ledger()[0]
        done += min(chunk, runs - done)
    return count, first_ledger


def row_n6a(F, runs: int) -> dict:
    kw = {"fraction": 0.25, "epsilon": 20.0, "replace": False}
    k_pos, led = _run_world(F, kw, np.full(8, GAMMA), runs, SEED + 610, lambda o: float(o.sum()) > 0.0)
    k_neg, _ = _run_world(F, kw, np.full(8, -GAMMA), runs, SEED + 611, lambda o: float(o.sum()) > 0.0)
    emp = empirical_eps_lower(k_pos, runs, k_neg, runs)
    claimed = led["epsilon_release"]
    measured = {"runs_per_input": runs, "ledger_first_call": led, "epsilon_release_claimed": claimed, "empirical": emp,
                "event": "sum of released values above 0", "real_process_dxo_per_run": True}
    checks = [chk("empirical 95% lower bound at most the ledger epsilon_release", f"<= {claimed}",
                  emp["eps_lower_bound"], emp["eps_lower_bound"] <= claimed)]
    return make_row(
        "N6a", {"d": 8, "fraction": 0.25, "n_upload": 2, "epsilon": 20.0, "replace": False,
                "inputs": "all +gamma against all -gamma", "runs_per_input": runs, "seeds": [SEED + 610, SEED + 611]},
        "The empirical 95% lower bound on epsilon is at most the ledger epsilon_release for this call.",
        "lower bound at most claimed", measured, checks,
        "Every run calls the real process_dxo. Selection depends on absolute values, so the two inputs differ only in the release step.")


def row_n6b(F, runs: int) -> dict:
    kw = {"fraction": 0.25, "epsilon": 20.0, "replace": False}
    a = np.zeros(8)
    a[0] = GAMMA
    k_a, led = _run_world(F, kw, a, runs, SEED + 620, lambda o: o[0] != 0.0)
    k_b, _ = _run_world(F, kw, np.zeros(8), runs, SEED + 621, lambda o: o[0] != 0.0)
    emp = empirical_eps_lower(k_a, runs, k_b, runs)
    claimed = led["epsilon_threshold"] + led["epsilon_query"]
    measured = {"runs_per_input": runs, "ledger_first_call": led, "epsilon_threshold_plus_query_claimed": claimed,
                "empirical": emp, "event": "entry 0 of the output is nonzero", "real_process_dxo_per_run": True}
    checks = [chk("empirical 95% lower bound at most epsilon_threshold + epsilon_query", f"<= {claimed}",
                  emp["eps_lower_bound"], emp["eps_lower_bound"] <= claimed)]
    return make_row(
        "N6b", {"d": 8, "fraction": 0.25, "n_upload": 2, "epsilon": 20.0, "replace": False,
                "inputs": "(gamma, 0, ..., 0) against all 0", "runs_per_input": runs, "seeds": [SEED + 620, SEED + 621]},
        "The empirical 95% lower bound is at most epsilon_threshold + epsilon_query of the call.",
        "lower bound at most claimed", measured, checks,
        f"The bound cannot exceed about {emp['ceiling_at_this_run_count']:.1f} at this run count, so with a claim of 20 the check has little power.")


def row_n7(F, runs: int) -> dict:
    d, n = 1000, 100
    base = np.full(d, GAMMA)
    out: dict = {}
    for replace in (True, False):
        np.random.seed(SEED + (700 if replace else 701))
        counts = []
        for _ in range(runs):
            filt = F.SVTPrivacy(fraction=0.1, epsilon=100.0, replace=replace)
            data, _ = run_filter(F, filt, {"w": base})
            counts.append(int(np.count_nonzero(data["w"])))
        out[replace] = counts
    c_t, c_f = np.array(out[True]), np.array(out[False])
    frac_below = float((c_t < n).mean())
    measured = {"runs": runs, "replace_true": {"mean_nonzero": float(c_t.mean()), "min": int(c_t.min()), "max": int(c_t.max()),
                                             "fraction_below_n_upload": frac_below},
                "replace_false": {"mean_nonzero": float(c_f.mean()), "min": int(c_f.min()), "max": int(c_f.max())}}
    checks = [
        chk("replace True: fraction of runs with fewer than n_upload nonzero entries", ">= 0.95", frac_below, frac_below >= 0.95),
        chk("replace True: mean nonzero entries between 90 and 100", "[90, 100]", float(c_t.mean()), 90.0 <= float(c_t.mean()) <= 100.0),
        chk("replace False: exactly n_upload nonzero entries in every run", n, [int(c_f.min()), int(c_f.max())],
            bool((c_f == n).all())),
    ]
    return make_row(
        "N7", {"d": d, "n_upload": n, "epsilon": 100.0, "entries": "all equal to gamma", "runs": runs, "replace": [True, False],
               "seeds": [SEED + 700, SEED + 701]},
        "With replace True the number of nonzero released entries is below n_upload in at least 95% of runs and its mean is "
        "between 90 and 100. With replace False it is exactly n_upload in every run.",
        "as stated", measured, checks, "Sampling with replacement writes repeated picks to the same output cell.")


def row_n8(F) -> dict:
    rng = np.random.default_rng(SEED + 800)
    arr = rng.standard_normal(100) * GAMMA
    data = {"w": arr, "s": np.float64(0.123)}
    f0 = F.SVTPrivacy(fraction=0.0)
    np.random.seed(SEED + 800)
    out0, dxo0 = run_filter(F, f0, data)
    f1 = F.SVTPrivacy(fraction=1e-12)
    np.random.seed(SEED + 801)
    out1, _ = run_filter(F, f1, data)
    led1 = f1.get_privacy_ledger()
    measured = {
        "fraction_0": {"array_all_zero": bool(np.all(out0["w"] == 0.0)), "array_dtype": str(out0["w"].dtype),
                       "scalar_output": float(out0["s"]), "ledger_length": len(f0.get_privacy_ledger()),
                       "privacy_spent": f0.get_privacy_spent(), "has_budget_meta": dxo0.get_meta_prop("svt_privacy_budget") is not None},
        "fraction_1e-12": {"n_upload": led1[0]["n_upload"] if led1 else None, "ledger_length": len(led1)},
    }
    checks = [
        chk("fraction 0: array entries all zero", True, measured["fraction_0"]["array_all_zero"], measured["fraction_0"]["array_all_zero"]),
        chk("fraction 0: scalar unchanged", 0.123, measured["fraction_0"]["scalar_output"], float(out0["s"]) == 0.123),
        chk("fraction 0: ledger empty", 0, measured["fraction_0"]["ledger_length"], measured["fraction_0"]["ledger_length"] == 0),
        chk("fraction 1e-12: n_upload is 1", 1, measured["fraction_1e-12"]["n_upload"], measured["fraction_1e-12"]["n_upload"] == 1),
        chk("fraction 1e-12: ledger has one entry", 1, len(led1), len(led1) == 1),
    ]
    return make_row(
        "N8", {"d": 100, "extra": "one scalar entry 0.123", "fraction": [0.0, 1e-12], "seeds": [SEED + 800, SEED + 801]},
        "At 0 the output is all zeros for array entries, the scalar is unchanged and the ledger is empty. At 1e-12 n_upload is 1 and the ledger has one entry.",
        "exact", measured, checks, "At fraction 0 the call returns before the accountant is reached, so no budget metadata is attached.")


def row_n9(F) -> dict:
    rng = np.random.default_rng(SEED + 900)
    data = {"w": rng.standard_normal(100) * GAMMA, "s": np.float64(0.123)}
    np.random.seed(SEED + 900)
    filt = F.SVTPrivacy()
    with record_randomness(F) as rec:
        out, dxo = run_filter(F, filt, data)
    led = filt.get_privacy_ledger()[-1]
    cum = dxo.get_meta_prop("svt_privacy_accountant")
    measured = {"ledger": led, "scalar_output": float(out["s"]), "scalar_output_type": type(out["s"]).__name__,
                "cumulative_meta": cum, "group_counts_sorted_by_name": rec.group_counts[0],
                "selected_counts_sorted_by_name": rec.selected_counts[0], "group_names_sorted": sorted(data)}
    checks = [
        chk("scalar returned unchanged", 0.123, float(out["s"]), float(out["s"]) == 0.123),
        chk("ledger has_scalar_passthrough is True", True, led["has_scalar_passthrough"], led["has_scalar_passthrough"] is True),
        chk("ledger entry has a note", "present", led.get("note"), bool(led.get("note"))),
        chk("n_upload computed from 101 entries is 11", 11, led["n_upload"], led["n_upload"] == 11),
    ]
    return make_row(
        "N9", {"array_entries": 100, "scalar_entries": 1, "scalar_value": 0.123, "fraction": 0.1, "seed": SEED + 900},
        "The scalar is returned unchanged, the ledger entry has has_scalar_passthrough True and a note, and n_upload is computed from 101 entries with fraction 0.1 (11).",
        "exact", measured, checks,
        "The scalar takes part in acceptance and in the partition of selected counts. The group counts and selected counts are recorded as an observation.")


def row_n10(F) -> dict:
    d = 1000
    base = np.random.default_rng(SEED + 1000).standard_normal(d) * GAMMA
    sig = inspect.signature(F.SVTPrivacy.__init__)
    params = [p for p in sig.parameters if p != "self"]
    seedlike = [p for p in params if any(t in p.lower() for t in ("seed", "rng", "random"))]
    outs, states = [], []
    for seed in (123, 123, 124):
        np.random.seed(seed)
        s0 = np.random.get_state()
        data, _ = run_filter(F, F.SVTPrivacy(), {"w": base})
        s1 = np.random.get_state()
        outs.append(data["w"].copy())
        states.append({"key_changed": bool(not np.array_equal(s0[1], s1[1])), "pos_before": int(s0[2]), "pos_after": int(s1[2])})
    same = bool(np.array_equal(outs[0], outs[1]))
    diff = bool(not np.array_equal(outs[0], outs[2]))
    advanced = all(s["key_changed"] or s["pos_after"] != s["pos_before"] for s in states)
    measured = {"constructor_parameters": params, "seed_like_parameters": seedlike, "same_seed_identical": same,
                "different_seed_differs": diff, "global_state_advances": states}
    checks = [
        chk("constructor has no seed argument", [], seedlike, not seedlike),
        chk("same seed gives identical output", True, same, same),
        chk("different seed differs", True, diff, diff),
        chk("global numpy state advances during a call", True, advanced, advanced),
    ]
    return make_row(
        "N10", {"d": d, "seeds": [123, 123, 124], "seeding": "numpy.random.seed"},
        "The constructor has no seed argument. The same seed gives identical output, a different seed differs, and the global numpy state advances.",
        "exact", measured, checks, "The filter draws from the global numpy generator.")


def row_n11(F) -> dict:
    base = np.full(200, 0.5 * GAMMA)
    np.random.seed(SEED + 1100)
    filt = F.SVTPrivacy()
    for _ in range(3):
        run_filter(F, filt, {"w": base})
    led3 = filt.get_privacy_ledger()
    spent3 = filt.get_privacy_spent()
    seq = 0.0
    for e in led3:
        seq += e["epsilon_total"]
    filt.handle_event(F.EventType.START_RUN, F.FLContext())
    led_after, spent_after = filt.get_privacy_ledger(), filt.get_privacy_spent()
    run_filter(F, filt, {"w": base})
    led_next, spent_next = filt.get_privacy_ledger(), filt.get_privacy_spent()
    measured = {"ledger_length_after_3_calls": len(led3), "spent_after_3_calls": spent3, "sequential_sum_of_ledger": seq,
                "ledger_length_after_start_run": len(led_after), "spent_after_start_run": spent_after,
                "ledger_length_after_next_call": len(led_next), "spent_after_next_call": spent_next,
                "next_call_epsilon_total": led_next[0]["epsilon_total"] if led_next else None}
    checks = [
        chk("ledger has 3 entries after 3 calls", 3, len(led3), len(led3) == 3),
        chk("get_privacy_spent equals the sum of the entries", seq, spent3, spent3 == seq),
        chk("after START_RUN the ledger is empty and spent is 0", {"ledger": 0, "spent": 0.0},
            {"ledger": len(led_after), "spent": spent_after}, len(led_after) == 0 and spent_after == 0.0),
        chk("next call starts a new sum", led_next[0]["epsilon_total"] if led_next else None, spent_next,
            len(led_next) == 1 and spent_next == led_next[0]["epsilon_total"]),
    ]
    return make_row(
        "N11", {"calls": "3, then START_RUN via handle_event, then 1", "d": 200, "seed": SEED + 1100},
        "The ledger has 3 entries and get_privacy_spent() equals their sum. After START_RUN it is empty and the next call starts a new sum.",
        "exact", measured, checks, "The sum is taken in ledger order, which is the order the filter adds them.")


def row_n12(F, calls: int) -> dict:
    d, eps = 1000, 1.0
    res: dict = {}
    for steps in (1, 5):
        np.random.seed(SEED + 1200 + steps)
        base = np.full(d, GAMMA * steps)
        devs, scales = [], None
        for _ in range(calls):
            filt = F.SVTPrivacy(fraction=0.1, epsilon=eps)
            with record_randomness(F) as rec:
                data, _ = run_filter(F, filt, {"w": base}, steps=steps)
            w = data["w"]
            nz = w[w != 0.0]
            devs.append(np.abs(nz - GAMMA * steps))
            if scales is None:
                scales = {"threshold": rec.laplace[0]["scale"], "query": rec.laplace[1]["scale"], "release": rec.laplace[-1]["scale"]}
        dev = np.concatenate(devs)
        res[steps] = {"recorded_scales": scales, "mean_abs_output_deviation": float(dev.mean()), "released_values": int(dev.size)}
    s1, s5 = res[1]["recorded_scales"], res[5]["recorded_scales"]
    ratio = res[5]["mean_abs_output_deviation"] / res[1]["mean_abs_output_deviation"]
    scale_rel = {k: rel(s5[k], s1[k]) for k in s1}
    expected_release = 2.0 * GAMMA * 100 / (100 * 0.1)
    measured = {"calls_per_step_count": calls, "steps_1": res[1], "steps_5": res[5], "recorded_scale_relative_differences": scale_rel,
                "measured_ratio_steps5_over_steps1": ratio, "expected_release_scale": expected_release,
                "measured_release_noise_mean_abs_steps1": res[1]["mean_abs_output_deviation"]}
    checks = [
        chk("recorded Laplace scales unchanged between 1 and 5 steps", "relative diff <= 1e-9", scale_rel,
            all(v <= 1e-9 for v in scale_rel.values())),
        chk("release noise in output units is 5 times larger with 5 steps", "5 within 10%", ratio, abs(ratio - 5.0) <= 0.5),
    ]
    return make_row(
        "N12", {"d": d, "epsilon": eps, "NUM_STEPS_CURRENT_ROUND": [1, 5], "entries": "1e-5 times the step count",
                "calls_per_step_count": calls, "seeds": [SEED + 1201, SEED + 1205]},
        "The release noise in output units is 5 times larger with 5 steps (scale ratio 5), and the recorded Laplace scales are unchanged.",
        "relative 1e-9 on recorded scales, 10% on measured ratio", measured, checks,
        "The ratio is the mean absolute deviation of released values from steps times gamma, pooled over all calls. "
        "The number of calls was not named in the pre-registration.")


# ---------------------------------------------------------------- PercentilePrivacy rows
def pp_input(seed: int, size: int = 10_000) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(size)
    assert np.unique(x).size == size and np.unique(np.abs(x)).size == size
    return x


def run_pp(F, filt, data: dict, steps=None):
    out = filt.process_dxo(make_dxo(F, data, steps), F.Shareable(), F.FLContext())
    return out.data


def row_p0(F) -> dict:
    x = pp_input(SEED + 2000)
    filt = F.PercentilePrivacy()
    before = x.copy()
    np.random.seed(1)
    state0 = np.random.get_state()
    o1 = run_pp(F, filt, {"w": x})["w"]
    state1 = np.random.get_state()
    o2 = run_pp(F, filt, {"w": x})["w"]
    np.random.seed(2)
    o3 = run_pp(F, filt, {"w": x})["w"]
    np.random.seed(3)
    o4 = run_pp(F, filt, {"w": x})["w"]
    same = bool(np.array_equal(o1, o2) and np.array_equal(o1, o3) and np.array_equal(o1, o4))
    untouched = bool(np.array_equal(state0[1], state1[1]) and state0[2] == state1[2])
    measured = {"outputs_identical_in_four_calls": same, "global_numpy_state_unchanged_by_call": untouched,
                "input_unchanged": bool(np.array_equal(before, x))}
    checks = [chk("output identical every time", True, same, same),
              chk("global numpy random state is not advanced", True, untouched, untouched)]
    return make_row(
        "P0", {"d": 10_000, "calls": "same input twice without reseed, then after numpy.random.seed(2) and (3)", "defaults": True},
        "The output is identical every time, because the filter uses no randomness.", "exact", measured, checks,
        "The state check is an addition that supports the prediction. It is not a separate prediction.")


def row_p1(F) -> dict:
    x = pp_input(SEED + 2100)
    cells, checks = [], []
    for q in (10, 50, 90):
        out = run_pp(F, F.PercentilePrivacy(percentile=q), {"w": x})["w"]
        zeros = int((out == 0.0).sum())
        expected = q / 100 * x.size
        cells.append({"percentile": q, "zero_entries": zeros, "expected": expected})
        checks.append(chk(f"percentile {q}: zero entries within 1 of {expected:.0f}", expected, zeros, abs(zeros - expected) <= 1))
    return make_row("P1", {"d": 10_000, "percentile": [10, 50, 90], "gamma": 0.01},
                    "The number of entries that become exactly 0 is within 1 of percentile percent of the entries.",
                    "1 entry", {"cells": cells}, checks, "Input is standard normal, 10,000 distinct values.")


def row_p2(F) -> dict:
    x = pp_input(SEED + 2200)
    steps, gamma, q = 4, 0.01, 10
    out = run_pp(F, F.PercentilePrivacy(percentile=q, gamma=gamma), {"w": x}, steps=steps)["w"]
    delta = x / steps
    cutoff = float(np.percentile(np.abs(delta), q))
    mask = (np.abs(delta) > cutoff) & (np.abs(delta) > gamma)
    expected = np.sign(x[mask]) * steps * gamma
    worst = float(np.max(np.abs(out[mask] / expected - 1.0)))
    measured = {"steps": steps, "gamma": gamma, "cutoff": cutoff, "entries_checked": int(mask.sum()), "worst_relative_difference": worst,
                "expected_magnitude": steps * gamma}
    checks = [chk("entries above cutoff and above gamma equal sign * 4 * gamma", "relative diff <= 1e-12", worst,
                  worst <= 1e-12 and int(mask.sum()) > 0)]
    return make_row("P2", {"d": 10_000, "NUM_STEPS_CURRENT_ROUND": steps, "percentile": q, "gamma": gamma},
                    "Entries with absolute value above the cutoff and above gamma are clipped to gamma after division by the step count and multiplied back, so the output equals sign times 4 times gamma.",
                    "relative 1e-12", measured, checks, "The cutoff is computed from the input independently of the filter.")


def row_p3(F) -> dict:
    x = pp_input(SEED + 2300)
    gamma = 0.01
    out = run_pp(F, F.PercentilePrivacy(percentile=100, gamma=gamma), {"w": x})["w"]
    a = np.abs(x)
    top = int(np.argmax(a))
    below_zero = bool(np.all(out[a < a[top]] == 0.0))
    kept = bool(out[top] != 0.0)
    nonzero = int(np.count_nonzero(out))
    measured = {"max_abs_input": float(a[top]), "output_at_max": float(out[top]), "all_below_max_zero": below_zero,
                "entry_at_max_kept": kept, "nonzero_entries": nonzero, "expected_output_at_max": float(np.sign(x[top]) * gamma)}
    checks = [chk("every entry strictly below the maximum is 0", True, below_zero, below_zero),
              chk("entry at the maximum is kept", True, kept, kept)]
    return make_row("P3", {"d": 10_000, "percentile": 100, "gamma": gamma},
                    "Every entry with absolute value strictly below the maximum becomes 0, and the entry at the maximum is kept.",
                    "exact", measured, checks, "The kept entry is clipped to gamma.")


def row_p4(F) -> dict:
    rng = np.random.default_rng(SEED + 2400)
    steps = 3
    data = {"a": rng.standard_normal(10_000), "b": rng.standard_normal(50), "big": np.float64(3.7), "tiny": np.float64(1e-9)}
    out = run_pp(F, F.PercentilePrivacy(), data, steps=steps)
    r_big, r_tiny = rel(float(out["big"]), 3.7), rel(float(out["tiny"]), 1e-9)
    try:
        run_pp(F, F.PercentilePrivacy(), {"a": rng.standard_normal(100), "s": 0.5})
        py_float = "accepted"
    except Exception as e:  # recorded, not hidden
        py_float = exc_str(e)
    measured = {"steps": steps, "big_scalar_input": 3.7, "big_scalar_output": float(out["big"]), "rel_diff_big": r_big,
                "tiny_scalar_input": 1e-9, "tiny_scalar_output": float(out["tiny"]), "rel_diff_tiny": r_tiny,
                "python_float_scalar_entry": py_float}
    checks = [chk("scalar above gamma returned unchanged (no clipping)", "relative diff <= 1e-12", r_big, r_big <= 1e-12),
              chk("scalar below the cutoff returned unchanged (not zeroed)", "relative diff <= 1e-12", r_tiny, r_tiny <= 1e-12)]
    return make_row("P4", {"arrays": [10_000, 50], "scalars": ["numpy float64 3.7", "numpy float64 1e-9"], "NUM_STEPS_CURRENT_ROUND": steps},
                    "The scalar is returned unchanged by value (multiplied by the step count after division), with no clipping.",
                    "relative 1e-12", measured, checks,
                    "Scalars are numpy float64. A plain Python float entry is recorded in the measured block as an observation.")


# ---------------------------------------------------------------- main
def build_rows(F, quick: bool) -> list[dict]:
    mc = 2_000 if quick else 200_000
    n7 = 40 if quick else 200
    n12 = 20 if quick else 200
    plan = [("N1", lambda: row_n1(F)), ("N2", lambda: row_n2(F)), ("N3", lambda: row_n3(F)), ("N4", lambda: row_n4(F)),
            ("N5", lambda: row_n5(F)), ("N6a", lambda: row_n6a(F, mc)), ("N6b", lambda: row_n6b(F, mc)),
            ("N7", lambda: row_n7(F, n7)), ("N8", lambda: row_n8(F)), ("N9", lambda: row_n9(F)), ("N10", lambda: row_n10(F)),
            ("N11", lambda: row_n11(F)), ("N12", lambda: row_n12(F, n12)),
            ("P0", lambda: row_p0(F)), ("P1", lambda: row_p1(F)), ("P2", lambda: row_p2(F)), ("P3", lambda: row_p3(F)),
            ("P4", lambda: row_p4(F))]
    rows = []
    t0 = time.time()
    for rid, fn in plan:
        try:
            rows.append(fn())
        except Exception as e:  # a failed row stays in the record
            rows.append({"id": rid, "configuration": None, "prediction": None, "tolerance": None,
                         "measured": {"exception": exc_str(e)}, "checks": [chk("row ran", "no exception", exc_str(e), False)],
                         "prediction_held": False, "note": "The row raised an exception."})
        print(f"{rid} done {time.time() - t0:.0f}s held={rows[-1]['prediction_held']}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output directory (required with --quick)")
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo runs, for tests")
    args = ap.parse_args()
    if args.quick and not args.out:
        ap.error("--quick requires --out so that the committed evidence is not overwritten")

    logging.disable(logging.CRITICAL)
    F = load_nvflare()
    out = Path(args.out) if args.out else DEFAULT_OUT_PARENT / f"nvflare-{F.version}"
    t0 = time.time()
    rows = build_rows(F, args.quick)
    mc = 2_000 if args.quick else 200_000
    checks = [(r["id"], c) for r in rows for c in r["checks"]]
    summary = {
        "rows_total": len(rows),
        "rows_held": sum(1 for r in rows if r["prediction_held"]),
        "rows_failed": [r["id"] for r in rows if not r["prediction_held"]],
        "checks_total": len(checks),
        "checks_held": sum(1 for _, c in checks if c["held"]),
        "checks_failed": [{"row": rid, **c} for rid, c in checks if not c["held"]],
        "quick_mode_note": ("Quick mode uses fewer runs. Rows that need many runs may fail and quick output is not evidence."
                            if args.quick else None),
    }
    doc = {
        "schema": "feddisclosure.nvflare_dp_audit.v1",
        "quick": args.quick,
        "environment": {
            "nvflare": F.version,
            "nvflare_svt_privacy_sha256": F.svt_sha256,
            "nvflare_percentile_privacy_sha256": F.pp_sha256,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": {"seed_base": SEED, "alpha": ALPHA, "gamma": GAMMA, "mc_runs_per_input_n6": mc,
                       "n7_runs": 40 if args.quick else 200, "n12_calls_per_step_count": 20 if args.quick else 200,
                       "n5_timeout_seconds": 10.0, "preregistration": "docs/preregistration/v0.16-nvflare.md"},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary,
        "rows": rows,
        "runtime_seconds": round(time.time() - t0, 1),
    }

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

    out.mkdir(parents=True, exist_ok=True)
    (out / OUT_NAME).write_text(json.dumps(doc, indent=2, default=enc) + "\n")
    print(f"wrote {out / OUT_NAME} in {doc['runtime_seconds']}s, {summary['rows_held']} of {summary['rows_total']} rows held, "
          f"{summary['checks_held']} of {summary['checks_total']} checks held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
