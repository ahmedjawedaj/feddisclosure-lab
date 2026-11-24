#!/usr/bin/env python3
"""Audit of FedML 0.9.6 differential privacy mechanisms (v0.16 P2).

Runs the real classes from fedml/core/dp in process and compares the mechanism
that executes with the one implied by the configured parameters. See
docs/preregistration/v0.16-fedml.md for the rows and predictions. The method
is the one of scripts/audit_flower_dp.py and its math is copied from there
(gauss_delta, gauss_eps, cp_upper, cp_lower, planned_threshold,
empirical_eps_lower), so that this file needs no Flower install. A test checks
the copies against the Flower script source.

Isolation. fedml/__init__.py starts MLOps, tracking and telemetry code and is
never executed. Bare module objects for fedml, fedml.core and fedml.core.common
are registered in sys.modules with their package paths, and only fedml.core.dp
and fedml.core.common.ml_engine_backend are imported from the installed
package. During mechanism runs socket creation raises, so a network call fails.

Usage: python scripts/audit_fedml_dp.py --out DIR [--quick]
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import math
import platform
import re
import socket
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

DIM = 16
DELTA = 1e-5
ALPHA = 0.05
SEED = 20261124


# ---------------------------------------------------------------- analytic
# Copied from scripts/audit_flower_dp.py.
def gauss_delta(mu: float, eps: float) -> float:
    """Exact delta of a Gaussian mechanism with parameter mu (Balle and Wang 2018)."""
    if mu <= 0:
        return 0.0
    a = stats.norm.cdf(mu / 2 - eps / mu)
    log_b = eps + stats.norm.logcdf(-mu / 2 - eps / mu)
    return float(max(a - math.exp(log_b), 0.0))


def gauss_eps(mu: float, delta: float) -> float:
    """Smallest epsilon such that the Gaussian mechanism is (eps, delta)-DP."""
    if gauss_delta(mu, 0.0) <= delta:
        return 0.0
    hi = 1.0
    while gauss_delta(mu, hi) > delta:
        hi *= 2
    return float(optimize.brentq(lambda e: gauss_delta(mu, e) - delta, 0.0, hi))


def cp_upper(k: int, n: int, a: float) -> float:
    return 1.0 if k >= n else float(stats.beta.ppf(1 - a, k + 1, n - k))


def cp_lower(k: int, n: int, a: float) -> float:
    return 0.0 if k <= 0 else float(stats.beta.ppf(a, k, n - k + 1))


def planned_threshold(mu: float, trials: int) -> float:
    """Threshold (in sigma units) fixed from predicted mu before the run."""
    best_t, best = mu / 2, -1.0
    for t in np.linspace(0.0, mu, 401):
        fpr = stats.norm.sf(t)
        tpr = stats.norm.sf(t - mu)
        ub = cp_upper(int(round(fpr * trials)), trials, ALPHA / 2)
        lb = cp_lower(int(round(tpr * trials)), trials, ALPHA / 2)
        val = math.log(max(lb - DELTA, 1e-300) / ub)
        if val > best:
            best, best_t = val, float(t)
    return best_t


def empirical_eps_lower(p0: np.ndarray, p1: np.ndarray, thr: float) -> dict:
    """Clopper-Pearson lower bound on epsilon at DELTA from one threshold test."""
    n0, n1 = len(p0), len(p1)
    fp, tp = int((p0 > thr).sum()), int((p1 > thr).sum())
    fpr_ub = cp_upper(fp, n0, ALPHA / 2)
    tpr_lb = cp_lower(tp, n1, ALPHA / 2)
    eps = math.log((tpr_lb - DELTA) / fpr_ub) if tpr_lb > DELTA else 0.0
    return {
        "threshold": thr,
        "trials_per_world": n0,
        "false_positives": fp,
        "true_positives": tp,
        "fpr_upper": fpr_ub,
        "tpr_lower": tpr_lb,
        "eps_lower_bound": max(eps, 0.0),
        "confidence": 1 - ALPHA,
    }


# New in this file: Laplace.
def laplace_delta(eps_pure: float, eps: float) -> float:
    """Exact delta at eps of a Laplace mechanism that is eps_pure-DP and tight."""
    return float(max(1.0 - math.exp((eps - eps_pure) / 2.0), 0.0))


def laplace_eps(eps_pure: float, delta: float) -> float:
    """Smallest epsilon such that the tight Laplace mechanism is (eps, delta)-DP."""
    if delta <= 0:
        return eps_pure
    return float(max(eps_pure + 2.0 * math.log1p(-delta), 0.0))


def laplace_planned_threshold(b: float, shift: np.ndarray, trials: int) -> float:
    """Threshold for the Laplace likelihood-ratio statistic, from the predicted law."""
    rng = np.random.default_rng(7)
    n = 400_000
    x0 = rng.laplace(0, b, (n, len(shift)))
    x1 = rng.laplace(0, b, (n, len(shift))) + shift
    t0, t1 = lr_stat(x0, shift, b), lr_stat(x1, shift, b)
    best_t, best = 0.0, -1e9
    for thr in np.linspace(0.0, float(np.abs(shift).sum() / b), 161):
        fp = int((t0 > thr).sum() * trials / n)
        tp = int((t1 > thr).sum() * trials / n)
        ub = cp_upper(fp, trials, ALPHA / 2)
        lb = cp_lower(tp, trials, ALPHA / 2)
        val = math.log((lb - DELTA) / ub) if lb > DELTA else -1e9
        if val > best:
            best, best_t = val, float(thr)
    return best_t


def lr_stat(x: np.ndarray, shift: np.ndarray, b: float) -> np.ndarray:
    """Log likelihood ratio of Laplace(shift, b) against Laplace(0, b), summed over coordinates."""
    return ((np.abs(x) - np.abs(x - shift)) / b).sum(axis=1)


def noise_stats(samples: np.ndarray, law: str = "normal") -> dict:
    """Per-coordinate scale, a distribution check, and independence checks."""
    sd = samples.std(axis=0, ddof=1)
    col = samples[:, 1]
    if law == "normal":
        ks = stats.kstest((col - col.mean()) / col.std(ddof=1), "norm")
    else:
        ks = stats.kstest(col / np.abs(col).mean(), "laplace")
    c = np.corrcoef(samples[:, :4].T)
    lag1 = float(np.corrcoef(samples[:-1, 0], samples[1:, 0])[0, 1])
    return {
        "runs": int(samples.shape[0]),
        "sigma": float(np.sqrt((sd**2).mean())),
        "mean_abs": float(np.abs(samples).mean()),
        "sigma_min_coord": float(sd.min()),
        "sigma_max_coord": float(sd.max()),
        "ks_pvalue": float(ks.pvalue),
        "max_abs_column_corr": float(np.abs(c - np.eye(4)).max()),
        "lag1_corr": lag1,
    }


def chk(name: str, expected, observed, held: bool) -> dict:
    return {"check": name, "expected": expected, "observed": observed, "held": bool(held)}


def rel(a: float, b: float) -> float:
    return abs(a / b - 1.0) if b else float("inf")


# ---------------------------------------------------------------- loading
def load_fedml():
    """Import fedml.core.dp without running fedml/__init__.py."""
    spec = importlib.util.find_spec("fedml")
    if spec is None:
        raise ImportError("fedml is not installed")
    root = Path(list(spec.submodule_search_locations)[0])
    if "fedml" not in sys.modules:
        for name, sub in (("fedml", ""), ("fedml.core", "core"), ("fedml.core.common", "core/common")):
            mod = types.ModuleType(name)
            mod.__path__ = [str(root / sub)]
            sys.modules[name] = mod
        sys.modules["fedml"].core = sys.modules["fedml.core"]
        sys.modules["fedml.core"].common = sys.modules["fedml.core.common"]
    from fedml.core.dp.budget_accountant import rdp_accountant as acc_mod
    from fedml.core.dp.budget_accountant.rdp_analysis import compute_rdp
    from fedml.core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
    from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
    from fedml.core.dp.frames.cdp import GlobalDP
    from fedml.core.dp.frames.dp_clip import DP_Clip
    from fedml.core.dp.frames.ldp import LocalDP
    from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism
    from fedml.core.dp.mechanisms.gaussian import Gaussian
    from fedml.core.dp.mechanisms.laplace import Laplace

    dp_dir = root / "core" / "dp"
    files = sorted(p for p in dp_dir.rglob("*.py") if "test" not in p.relative_to(dp_dir).parts)
    tree = hashlib.sha256(
        "".join(f"{p.relative_to(dp_dir)}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in files).encode()
    ).hexdigest()
    try:
        version = importlib.metadata.version("fedml")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    loaded = sorted(m for m in sys.modules if m.startswith("fedml"))
    return types.SimpleNamespace(
        root=root, dp_dir=dp_dir, version=version, tree_sha256=tree, loaded_modules=loaded,
        RDP_Accountant=acc_mod.RDP_Accountant, compute_rdp=compute_rdp,
        FedMLDifferentialPrivacy=FedMLDifferentialPrivacy, BaseDPFrame=BaseDPFrame,
        GlobalDP=GlobalDP, DP_Clip=DP_Clip, LocalDP=LocalDP, DPMechanism=DPMechanism,
        Gaussian=Gaussian, Laplace=Laplace,
    )


def block_network() -> None:
    def refuse(*_a, **_k):
        raise RuntimeError("network access is blocked during the FedML audit")

    socket.socket = refuse  # type: ignore[assignment]
    socket.create_connection = refuse  # type: ignore[assignment]


def e1() -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def exc_str(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:300]


# ---------------------------------------------------------------- helpers
def noised(F, mech, base: np.ndarray, seed: int, key: str = "w") -> np.ndarray:
    """Run DPMechanism.add_noise once on a batched float32 tensor of shape (n, DIM)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    out = mech.add_noise(collections.OrderedDict([(key, torch.from_numpy(base.astype(np.float32)))]))
    return out[key].numpy().astype(np.float64)


def gaussian_cp(p0: np.ndarray, p1: np.ndarray, mu_pred: float, sigma_pred: float) -> dict:
    trials = len(p0)
    thr_sigma = planned_threshold(mu_pred, trials)
    emp = empirical_eps_lower(p0, p1, thr_sigma * sigma_pred)
    emp["threshold_in_predicted_sigma"] = thr_sigma
    return emp


# ---------------------------------------------------------------- rows
def row_g1(F, runs: int) -> dict:
    cells = []
    checks = []
    cell_id = 0
    for delta in (1e-5, 0.1):
        for eps in (0.1, 0.5, 1.0):
            cell_id += 1
            mech = F.DPMechanism("gaussian", eps, delta, sensitivity=1)
            formula = math.sqrt(2 * math.log(1.25 / delta)) / eps
            ns = noise_stats(noised(F, mech, np.zeros((runs, DIM)), SEED + cell_id))
            mu_claim = 1.0 / formula
            mu = 1.0 / ns["sigma"]
            d_exact = gauss_delta(mu, eps)
            cell = {
                "epsilon": eps,
                "delta": delta,
                "sensitivity": 1.0,
                "sigma_attribute": float(mech.dp.scale),
                "sigma_formula": formula,
                "noise": ns,
                "sigma_ratio_measured_over_formula": ns["sigma"] / formula,
                "mu_claimed": mu_claim,
                "mu_measured": mu,
                "exact_delta_at_configured_eps": d_exact,
                "exact_delta_ratio_to_configured": d_exact / delta,
                "exact_eps_at_configured_delta": gauss_eps(mu, delta),
                "meets_configured_guarantee": bool(d_exact <= delta),
            }
            cells.append(cell)
            checks.append(chk(f"sigma within 1% of formula (eps {eps}, delta {delta})", 1.0,
                              cell["sigma_ratio_measured_over_formula"], rel(ns["sigma"], formula) <= 0.01))
            checks.append(chk(f"meets (eps, delta) (eps {eps}, delta {delta})", True,
                              cell["meets_configured_guarantee"], cell["meets_configured_guarantee"]))
    return {"row": "G1", "api": "fedml.core.dp.mechanisms.gaussian.Gaussian via DPMechanism",
            "cells": cells, "checks": checks}


def row_g2(F) -> dict:
    out = []
    checks = []
    for eps in (1.0, 1.0000001, 2, 5, 10):
        try:
            m = F.DPMechanism("gaussian", eps, 1e-5, sensitivity=1)
            rec = {"epsilon": eps, "accepted": True, "sigma": float(m.dp.scale), "exception": None}
        except Exception as e:  # recorded, not hidden
            rec = {"epsilon": eps, "accepted": False, "exception": exc_str(e)}
        out.append(rec)
        want_accept = eps == 1.0
        ok = rec["accepted"] if want_accept else (not rec["accepted"] and rec["exception"].startswith("ValueError"))
        checks.append(chk(f"epsilon {eps}", "accepted" if want_accept else "ValueError",
                          "accepted" if rec["accepted"] else rec["exception"], ok))
    return {"row": "G2", "cells": out, "checks": checks}


def row_lp1(F, runs: int) -> dict:
    cells = []
    checks = []
    cell_id = 100
    for delta in (0.0, 1e-5, 0.1, 0.5):
        for eps in (0.1, 1, 5):
            cell_id += 1
            mech = F.DPMechanism("laplace", eps, delta, sensitivity=1)
            b_formula = 1.0 / (eps - math.log(1 - delta))
            ns = noise_stats(noised(F, mech, np.zeros((runs, DIM)), SEED + cell_id), law="laplace")
            b_hat = ns["mean_abs"]
            eps_pure = 1.0 / b_hat
            eps_pure_formula = eps - math.log(1 - delta)
            d_exact = laplace_delta(eps_pure, eps)
            cell = {
                "epsilon": eps,
                "delta": delta,
                "sensitivity_l1": 1.0,
                "scale_attribute": float(mech.dp.scale),
                "scale_formula": b_formula,
                "scale_measured": b_hat,
                "scale_ratio_measured_over_formula": b_hat / b_formula,
                "noise": ns,
                "eps_pure_formula": eps_pure_formula,
                "eps_pure_measured": eps_pure,
                "exact_delta_at_configured_eps": d_exact,
                "predicted_exact_delta_1_minus_sqrt_1_minus_delta": 1 - math.sqrt(1 - delta),
                "exact_eps_at_configured_delta": laplace_eps(eps_pure, delta),
                "exact_delta_from_formula_scale": laplace_delta(eps_pure_formula, eps),
                "pure_eps_within_2pct_of_claimed": bool(rel(eps_pure, eps_pure_formula) <= 0.02),
                "meets_configured_guarantee": bool(laplace_delta(eps_pure_formula, eps) <= delta + 1e-12),
            }
            cells.append(cell)
            checks.append(chk(f"scale within 1% (eps {eps}, delta {delta})", 1.0,
                              cell["scale_ratio_measured_over_formula"], rel(b_hat, b_formula) <= 0.01))
            checks.append(chk(f"meets claim (eps {eps}, delta {delta})", True,
                              cell["meets_configured_guarantee"], cell["meets_configured_guarantee"]))
            checks.append(chk(f"measured pure epsilon within 2% of claimed (eps {eps}, delta {delta})", True,
                              cell["eps_pure_measured"], cell["pure_eps_within_2pct_of_claimed"]))
    return {"row": "LP1", "api": "fedml.core.dp.mechanisms.laplace.Laplace via DPMechanism",
            "cells": cells, "checks": checks}


def row_lp2(F, runs: int, trials: int) -> dict:
    eps, delta, sens = 1.0, DELTA, 1.0
    mech = F.DPMechanism("laplace", eps, delta, sensitivity=sens)
    b_formula = sens / (eps - math.log(1 - delta))
    cases = {
        "concentrated": e1().astype(np.float64),
        "spread": np.full(DIM, 1.0 / math.sqrt(DIM)),
    }
    out = {"row": "LP2", "note": "configuration note. The sensitivity argument is read as L1 by the Laplace "
           "class. The update below has L2 norm 1 equal to the configured value.",
           "configured": {"epsilon": eps, "delta": delta, "sensitivity": sens, "scale_formula": b_formula},
           "cases": {}, "checks": []}
    ns = noise_stats(noised(F, mech, np.zeros((runs, DIM)), SEED + 200), law="laplace")
    b_hat = ns["mean_abs"]
    out["measured_scale"] = b_hat
    for name, u in cases.items():
        l1, l2 = float(np.abs(u).sum()), float(np.linalg.norm(u))
        eps_pure = l1 / b_hat
        rec = {"update_l1": l1, "update_l2": l2, "eps_pure_measured": eps_pure,
               "eps_pure_claimed": eps - math.log(1 - delta),
               "ratio_to_claimed": eps_pure / (eps - math.log(1 - delta)),
               "exact_eps_at_configured_delta": laplace_eps(eps_pure, delta)}
        if trials:
            thr = laplace_planned_threshold(b_formula, u, trials)
            x0 = noised(F, mech, np.zeros((trials, DIM)), SEED + 201)
            x1 = noised(F, mech, np.tile(u, (trials, 1)), SEED + 202)
            emp = empirical_eps_lower(lr_stat(x0, u, b_formula), lr_stat(x1, u, b_formula), thr)
            emp["statistic"] = "log likelihood ratio of the predicted Laplace law"
            rec["empirical"] = emp
        out["cases"][name] = rec
    c, s = out["cases"]["concentrated"], out["cases"]["spread"]
    out["checks"] = [
        chk("concentrated eps_pure within 2% of 1.00001", 1.00001, c["eps_pure_measured"],
            rel(c["eps_pure_measured"], 1.00001) <= 0.02),
        chk("spread eps_pure within 2% of 4.00004", 4.00004, s["eps_pure_measured"],
            rel(s["eps_pure_measured"], 4.00004) <= 0.02),
    ]
    if trials:
        out["checks"].append(chk("empirical bound at or below exact (spread)", "<= 4.0004",
                                 s["empirical"]["eps_lower_bound"],
                                 s["empirical"]["eps_lower_bound"] <= s["eps_pure_measured"] * 1.0 + 1e-9))
    return out


def row_lp3(F) -> dict:
    import torch

    out = {"row": "LP3", "note": "observation, not a privacy measurement", "outputs": {}}
    checks = []
    for name, args in (("gaussian", (1.0, 1e-5)), ("laplace", (1.0, 1e-5))):
        mech = F.DPMechanism(name, *args, sensitivity=1)
        torch.manual_seed(1)
        np.random.seed(1)
        o = mech.add_noise(collections.OrderedDict([
            ("w", torch.zeros(4, dtype=torch.float32)),
            ("n", torch.tensor(7, dtype=torch.int64)),
        ]))
        out["outputs"][name] = {"float32_input_becomes": str(o["w"].dtype),
                                "int64_buffer_becomes": str(o["n"].dtype),
                                "int64_buffer_value": float(o["n"])}
    checks.append(chk("gaussian float32 stays float32", "torch.float32",
                      out["outputs"]["gaussian"]["float32_input_becomes"],
                      out["outputs"]["gaussian"]["float32_input_becomes"] == "torch.float32"))
    checks.append(chk("laplace float32 becomes float64", "torch.float64",
                      out["outputs"]["laplace"]["float32_input_becomes"],
                      out["outputs"]["laplace"]["float32_input_becomes"] == "torch.float64"))
    for n in ("gaussian", "laplace"):
        dt = out["outputs"][n]["int64_buffer_becomes"]
        checks.append(chk(f"{n} int64 buffer becomes floating", "floating", dt, "float" in dt))
    out["checks"] = checks
    return out


def dp_clip_args(**kw):
    base = dict(clipping_norm=1.0, noise_multiplier=1.0, train_data_num_in_total=1000,
                client_num_per_round=10, client_num_in_total=100)
    base.update(kw)
    return types.SimpleNamespace(**base)


def tensor_dict(vec_list):
    import torch

    return collections.OrderedDict((f"t{i}", torch.tensor(v, dtype=torch.float32)) for i, v in enumerate(vec_list))


def total_norm(d) -> float:
    return float(np.sqrt(sum(float((v.double() ** 2).sum()) for v in d.values())))


def row_k1(F) -> dict:
    import torch

    rng = np.random.default_rng(1)
    cells, checks = [], []
    for target in (0.5, 1.0, 5.0, 50.0):
        v = rng.normal(size=12)
        v = v / np.linalg.norm(v) * target
        d = tensor_dict([v[:4], v[4:]])
        frame = F.DP_Clip(dp_clip_args())
        before = total_norm(d)
        first_tensor_before = d["t0"].clone()
        res = frame.clip_local_update(d)
        after = total_norm(res)
        rec = {"norm_before": before, "norm_after": after, "returns_same_object": res is d,
               "input_tensors_modified": bool(not torch.equal(first_tensor_before, d["t0"]))}
        cells.append(rec)
        if target > 1.0:
            ok = after <= 1.0 + 1e-6 and rel(after, 1.0) <= 1e-5
            checks.append(chk(f"norm {target} clipped to 1 within 1e-5", 1.0, after, ok))
            checks.append(chk(f"norm {target} modified in place", True, rec["input_tensors_modified"],
                              rec["input_tensors_modified"]))
        elif target < 1.0:
            checks.append(chk(f"norm {target} unchanged", target, after,
                              rel(after, target) <= 1e-5 and not rec["input_tensors_modified"]))
        else:
            checks.append(chk("norm 1 stays 1 within 1e-5", 1.0, after, rel(after, 1.0) <= 1e-5))
    # add_local_noise adds no noise
    frame = F.DP_Clip(dp_clip_args())
    g = tensor_dict([np.zeros(4), np.zeros(8)])
    v = rng.normal(size=12) * 3
    outs = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        local = tensor_dict([v[:4], v[4:]])
        outs.append(frame.add_local_noise(local, extra_auxiliary_info=g))
    same = all(torch.equal(outs[0][k], outs[1][k]) for k in outs[0])
    checks.append(chk("add_local_noise is deterministic across seeds (no noise)", True, same, same))
    return {"row": "K1", "api": "fedml.core.dp.frames.dp_clip.DP_Clip", "cells": cells,
            "add_local_noise_deterministic": same, "checks": checks}


def row_k2(F, n: int) -> dict:
    import torch

    args = dp_clip_args()
    qw = args.train_data_num_in_total * (args.client_num_per_round / args.client_num_in_total)
    frame = F.DP_Clip(args)
    z, c = args.noise_multiplier, args.clipping_norm
    calls = []
    torch.manual_seed(SEED)
    first_zero = None
    for k in range(1, 31):
        x = frame.compute_noise((n,), qw)
        sig = float(x.double().std())
        nonzero = float((x != 0).float().mean())
        calls.append({"call": k, "state_scale_after": float(frame._scale), "sigma_measured": sig,
                      "sigma_predicted_z_c_over_qw_pow_k": z * c / qw**k, "nonzero_fraction": nonzero})
        if first_zero is None and nonzero == 0.0:
            first_zero = k
    # within add_noise and across rounds
    frame2 = F.DP_Clip(args)
    torch.manual_seed(SEED + 1)
    rounds = []
    for r in (1, 2):
        w = collections.OrderedDict((k, torch.zeros(n)) for k in ("a", "b"))
        o = frame2.add_noise(w, qw)
        rounds.append({"round": r, "sigma_a": float(o["a"].double().std()), "sigma_b": float(o["b"].double().std())})
    checks = []
    ok = all(rel(c_["sigma_measured"], c_["sigma_predicted_z_c_over_qw_pow_k"]) <= 0.02 for c_ in calls[:15])
    checks.append(chk("calls 1 to 15 within 2% of z C / qw^k", "within 2%", [c_["sigma_measured"] for c_ in calls[:3]], ok))
    checks.append(chk("first all-zero call between 20 and 24", "20..24", first_zero,
                      first_zero is not None and 20 <= first_zero <= 24))
    exp = [(1, "a", 1), (1, "b", 2), (2, "a", 3), (2, "b", 4)]
    okr = all(rel(rounds[r - 1]["sigma_" + key], z * c / qw**k) <= 0.02 for r, key, k in exp)
    checks.append(chk("keys and rounds follow z C / qw^k with k continuing across rounds", "k = 1,2,3,4",
                      [rounds[0]["sigma_a"], rounds[0]["sigma_b"], rounds[1]["sigma_a"], rounds[1]["sigma_b"]], okr))
    return {"row": "K2", "api": "fedml.core.dp.frames.dp_clip.DP_Clip.compute_noise",
            "configured": {"noise_multiplier": z, "clipping_norm": c, "qw": qw}, "calls": calls,
            "first_call_with_all_zero_noise": first_zero, "across_keys_and_rounds": rounds, "checks": checks}


def row_k3(F) -> dict:
    import torch

    args = dp_clip_args()
    qw = args.train_data_num_in_total * (args.client_num_per_round / args.client_num_in_total)
    rng = np.random.default_rng(3)
    orig = collections.OrderedDict((k, torch.tensor(rng.normal(size=8), dtype=torch.float32)) for k in ("a", "b"))
    model = collections.OrderedDict((k, v.clone()) for k, v in orig.items())
    frame = F.DP_Clip(args)
    torch.manual_seed(SEED + 2)
    ret = frame.add_global_noise(model)
    dev = max(float((model[k].double() - orig[k].double() / qw).abs().max()) for k in orig)
    scale_vs_orig = max(float((model[k].double() * qw - orig[k].double()).abs().max()) for k in orig)
    checks = [
        chk("returns None", None, ret, ret is None),
        chk("input dict equals original / qw with no noise (max abs dev below 1e-6)", "< 1e-6", dev, dev < 1e-6),
    ]
    return {"row": "K3", "api": "fedml.core.dp.frames.dp_clip.DP_Clip.add_global_noise",
            "returned": None if ret is None else str(type(ret)),
            "max_abs_deviation_input_from_original_over_qw": dev,
            "max_abs_deviation_input_times_qw_from_original": scale_vs_orig,
            "state_scale_after": float(frame._scale), "checks": checks}


def row_k4(F) -> dict:
    out = {"row": "K4", "api": "fedml.core.dp.fedml_differential_privacy.FedMLDifferentialPrivacy",
           "flags": {}, "checks": []}
    common = dict(enable_dp=True, delta=1e-5, mechanism_type="gaussian", epsilon=1.0, sensitivity=1.0,
                  clipping_norm=1.0, noise_multiplier=1.0, train_data_num_in_total=1000,
                  client_num_per_round=10, client_num_in_total=100)
    for kind in ("ldp", "cdp", "dp_clip"):
        inst = F.FedMLDifferentialPrivacy()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            inst.init(types.SimpleNamespace(dp_solution_type=kind, **common))
        out["flags"][kind] = {
            "frame_class": type(inst.dp_solution).__name__,
            "is_dp_enabled": inst.is_dp_enabled(),
            "is_local_dp_enabled": inst.is_local_dp_enabled(),
            "is_global_dp_enabled": inst.is_global_dp_enabled(),
            "is_clipping": inst.is_clipping(),
        }
    f = out["flags"]["dp_clip"]
    out["checks"] = [
        chk("dp_clip: local and global predicates both false", [False, False],
            [f["is_local_dp_enabled"], f["is_global_dp_enabled"]],
            not f["is_local_dp_enabled"] and not f["is_global_dp_enabled"]),
        chk("ldp reachable through local predicate", True, out["flags"]["ldp"]["is_local_dp_enabled"],
            out["flags"]["ldp"]["is_local_dp_enabled"]),
        chk("cdp reachable through global predicate", True, out["flags"]["cdp"]["is_global_dp_enabled"],
            out["flags"]["cdp"]["is_global_dp_enabled"]),
    ]
    return out


def cdp_args(**kw):
    base = dict(enable_dp=True, dp_solution_type="cdp", mechanism_type="gaussian", epsilon=1.0, delta=1e-5,
                sensitivity=0.1, client_num_per_round=10, client_num_in_total=100)
    base.update(kw)
    return types.SimpleNamespace(**base)


def row_cd1(F, runs: int) -> dict:
    import torch

    args = cdp_args()
    inst = F.FedMLDifferentialPrivacy()
    with contextlib.redirect_stdout(io.StringIO()):
        inst.init(args)
    exc = None
    try:
        torch.manual_seed(SEED + 3)
        o = inst.add_global_noise(collections.OrderedDict([("w", torch.zeros(runs, DIM))]))
        ns = noise_stats(o["w"].numpy().astype(np.float64))
    except Exception as e:  # recorded, not hidden
        exc, ns = exc_str(e), None
    formula = math.sqrt(2 * math.log(1.25 / args.delta)) * args.sensitivity / args.epsilon
    frame = inst.dp_solution
    dir_ = F.dp_dir / "frames"
    reads = []
    for p in sorted(dir_.glob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"self\.enable_rdp_accountant(?!\s*=)", line):
                reads.append(f"{p.name}:{i}: {line.strip()}")
    checks = [
        chk("no exception with the accountant off", None, exc, exc is None),
        chk("sigma within 1% of formula", 1.0, None if ns is None else ns["sigma"] / formula,
            ns is not None and rel(ns["sigma"], formula) <= 0.01),
        chk("is_rdp_accountant_enabled exists and is False (candidate refuted)", False,
            getattr(frame, "is_rdp_accountant_enabled", "missing"),
            getattr(frame, "is_rdp_accountant_enabled", "missing") is False),
        chk("self.enable_rdp_accountant is not read anywhere in frames", [], reads, not reads),
    ]
    return {"row": "CD1", "api": "fedml.core.dp.frames.cdp.GlobalDP", "exception": exc,
            "configured": {"sensitivity": args.sensitivity, "epsilon": args.epsilon, "delta": args.delta},
            "sigma_formula": formula, "noise": ns,
            "frame_is_rdp_accountant_enabled": getattr(frame, "is_rdp_accountant_enabled", "missing"),
            "frame_enable_rdp_accountant": getattr(frame, "enable_rdp_accountant", "missing"),
            "reads_of_enable_rdp_accountant_in_frames": reads, "checks": checks}


def row_cd2(F) -> dict:
    args = cdp_args(enable_rdp_accountant=True, rdp_alpha=2.0, sigma=1.0, max_grad_norm=1.0, clipping=True)
    out = {"row": "CD2", "api": "fedml.core.dp.frames.cdp.GlobalDP"}
    try:
        F.GlobalDP(args)
        out["exception"] = None
    except Exception as e:  # recorded, not hidden
        out["exception"] = exc_str(e)
    inst = F.FedMLDifferentialPrivacy()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            inst.init(args)
        out["init_exception"] = None
    except Exception as e:
        out["init_exception"] = exc_str(e)
    out["checks"] = [
        chk("construction raises TypeError naming dp_param", "TypeError ... dp_param", out["exception"],
            bool(out["exception"]) and out["exception"].startswith("TypeError") and "dp_param" in out["exception"]),
        chk("FedMLDifferentialPrivacy.init raises the same", "TypeError", out["init_exception"],
            bool(out["init_exception"]) and out["init_exception"].startswith("TypeError")),
    ]
    return out


def row_cd3(F) -> dict:
    import torch

    rng = np.random.default_rng(5)
    norms = (0.5, 3.0, 30.0)

    def make():
        lst = []
        for t in norms:
            v = rng.normal(size=12)
            v = v / np.linalg.norm(v) * t
            lst.append((10, tensor_dict([v[:4], v[4:]])))
        return lst

    frame = F.GlobalDP(cdp_args(max_grad_norm=1.0))
    lst = make()
    originals = [(n, collections.OrderedDict((k, t.clone()) for k, t in d.items())) for n, d in lst]
    res = frame.global_clip(lst)
    after = [total_norm(d) for _, d in res]
    modified = [not all(torch.equal(d[k], o[k]) for k in d) for (_, d), (_, o) in zip(res, originals)]
    frame2 = F.GlobalDP(cdp_args())
    lst2 = make()
    res2 = frame2.global_clip(lst2)
    unchanged = res2 is lst2 and all(abs(total_norm(d) - t) < 1e-5 * t for (_, d), t in zip(res2, norms))
    checks = [
        chk("all clipped norms at most 1", "<= 1", after, all(a <= 1.0 + 1e-6 for a in after)),
        chk("large inputs equal 1 within 1e-4", 1.0, after[1:], all(rel(a, 1.0) <= 1e-4 for a in after[1:])),
        chk("small input unchanged", 0.5, after[0], rel(after[0], 0.5) <= 1e-5),
        chk("inputs modified in place", [False, True, True], modified, modified == [False, True, True]),
        chk("max_grad_norm unset: list returned unchanged", True, unchanged, unchanged),
    ]
    return {"row": "CD3", "api": "fedml.core.dp.frames.base_dp_solution.BaseDPFrame.global_clip",
            "max_grad_norm": 1.0, "input_norms": list(norms), "output_norms": after,
            "modified_in_place": modified, "unset_returns_same_list_unchanged": unchanged, "checks": checks}


def ldp_inst(F, **kw):
    args = types.SimpleNamespace(enable_dp=True, dp_solution_type="ldp", mechanism_type="gaussian",
                                 epsilon=1.0, delta=DELTA, sensitivity=2.0, **kw)
    inst = F.FedMLDifferentialPrivacy()
    with contextlib.redirect_stdout(io.StringIO()):
        inst.init(args)
    return inst, args


def ldp_world(inst, base: np.ndarray, seed: int) -> np.ndarray:
    import torch

    torch.manual_seed(seed)
    out = inst.add_local_noise(collections.OrderedDict([("w", torch.from_numpy(base.astype(np.float32)))]))
    return out["w"].numpy().astype(np.float64)


def row_ld(F, runs: int, trials: int, with_clip_setting: bool) -> list[dict]:
    kw = {"max_grad_norm": 1.0} if with_clip_setting else {}
    inst, args = ldp_inst(F, **kw)
    formula = math.sqrt(2 * math.log(1.25 / DELTA)) * args.sensitivity / args.epsilon
    rows = []
    amps = (25,) if with_clip_setting else (1, 5, 25)
    for a in amps:
        up, down = a * e1(), -a * e1()
        sens = float(np.linalg.norm(up - down))
        noise = ldp_world(inst, np.tile(up, (runs, 1)), SEED + 300 + a) - up
        ns = noise_stats(noise)
        mean_dev = float(np.abs(ldp_world(inst, np.tile(up, (runs, 1)), SEED + 301 + a).mean(axis=0) - up).max())
        mu = sens / ns["sigma"]
        mu_pred = sens / formula
        rec = {
            "row": "LD2" if with_clip_setting else "LD1",
            "api": "fedml.core.dp.frames.ldp.LocalDP",
            "configured": {"epsilon": args.epsilon, "delta": DELTA, "sensitivity": args.sensitivity,
                           "max_grad_norm": kw.get("max_grad_norm"), "update_norm": float(a)},
            "measured": {"sensitivity_replace_one": sens, "noise": ns, "sigma_formula": formula,
                         "sigma_ratio_measured_over_formula": ns["sigma"] / formula, "mu": mu,
                         "max_abs_mean_deviation_from_unclipped_input": mean_dev},
            "exact_from_measured_sigma": {"delta_at_configured_eps": gauss_delta(mu, args.epsilon),
                                          "eps_at_configured_delta": gauss_eps(mu, DELTA)},
            "meets_configured_guarantee": bool(gauss_delta(mu, args.epsilon) <= DELTA),
        }
        if trials:
            p0 = ldp_world(inst, np.tile(down, (trials, 1)), SEED + 310 + a)[:, 0] + a
            p1 = ldp_world(inst, np.tile(up, (trials, 1)), SEED + 311 + a)[:, 0] + a
            rec["empirical"] = gaussian_cp(p0, p1, mu_pred, formula)
        pred_eps = {1: 0.751, 5: 4.54, 25: 34.6}[a]
        e = rec["exact_from_measured_sigma"]["eps_at_configured_delta"]
        rec["checks"] = [
            chk("sigma within 1% of formula", 1.0, rec["measured"]["sigma_ratio_measured_over_formula"],
                rel(ns["sigma"], formula) <= 0.01),
            chk("exact epsilon within 5% of predicted", pred_eps, e, rel(e, pred_eps) <= 0.05),
            chk("output is not clipped (mean equals unclipped input)", "< 5 SE", mean_dev,
                mean_dev < 5 * ns["sigma"] / math.sqrt(runs)),
        ]
        if a == 1 and not with_clip_setting:
            rec["checks"].append(chk("meets the configured guarantee", True, rec["meets_configured_guarantee"],
                                     rec["meets_configured_guarantee"]))
        if trials and "empirical" in rec:
            rec["checks"].append(chk("empirical bound at or below exact", f"<= {e:.3f}",
                                     rec["empirical"]["eps_lower_bound"], rec["empirical"]["eps_lower_bound"] <= e))
        rows.append(rec)
    return rows


def acct_args():
    return types.SimpleNamespace(sigma=1.0, max_grad_norm=1.0, clipping=True)


def row_ra1(F) -> dict:
    out = {"row": "RA1", "api": "fedml.core.dp.budget_accountant.rdp_accountant.RDP_Accountant", "cases": {}}
    checks = []
    cases = {
        "q0.1_default_orders": (0.1, None),
        "q1.0_default_orders": (1.0, None),
        "q0.1_integer_orders": (0.1, [2, 4, 8, 16, 32]),
    }
    for name, (q, alphas) in cases.items():
        acc = F.RDP_Accountant(dp_param=1.0, alpha=2.0, dp_mechanism="gaussian", args=acct_args())
        rec = {}
        v0 = acc.get_epsilon(1e-5, alphas)
        rec["before_step"] = {"value": repr(v0), "type": type(v0).__name__}
        acc.step(noise_multiplier=1.0, sample_rate=q)
        try:
            rec["after_step"] = {"value": repr(acc.get_epsilon(1e-5, alphas)), "exception": None}
        except Exception as e:  # recorded, not hidden
            rec["after_step"] = {"exception": exc_str(e)}
        acc.step(noise_multiplier=1.0, sample_rate=q)
        rec["history_after_two_equal_steps"] = [list(h) for h in acc.history]
        acc.step(noise_multiplier=2.0, sample_rate=q)
        rec["history_after_third_different_step"] = [list(h) for h in acc.history]
        out["cases"][name] = rec
        exc = rec["after_step"]["exception"]
        checks.append(chk(f"{name}: before any step returns the tuple (0, 0)", "(0, 0)", rec["before_step"]["value"],
                          rec["before_step"]["value"] == "(0, 0)"))
        checks.append(chk(f"{name}: after a step raises TypeError", "TypeError", exc,
                          bool(exc) and exc.startswith("TypeError")))
        checks.append(chk(f"{name}: two equal steps merge into one entry with 2 steps", [[1.0, q, 2]],
                          rec["history_after_two_equal_steps"], rec["history_after_two_equal_steps"] == [[1.0, q, 2]]))
    out["checks"] = checks
    return out


def row_ra2(F) -> dict:
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    pred = {0.5: 10.73, 1.0: 4.73, 2.0: 2.17, 4.0: 1.01}
    cells, checks = [], []
    for sigma in (0.5, 1.0, 2.0, 4.0):
        rdp = F.compute_rdp(q=1.0, noise_multiplier=sigma, steps=1, orders=alphas)
        eps, alpha = F.RDP_Accountant.get_privacy_spent(orders=alphas, rdp=rdp, delta=DELTA)
        exact = gauss_eps(1.0 / sigma, DELTA)
        cells.append({"sigma": sigma, "rdp_epsilon": float(eps), "best_alpha": float(alpha),
                      "exact_epsilon": exact, "ratio": float(eps) / exact})
        checks.append(chk(f"sigma {sigma}: epsilon within 0.5% of {pred[sigma]}", pred[sigma], float(eps),
                          rel(float(eps), pred[sigma]) <= 0.005))
        checks.append(chk(f"sigma {sigma}: at or above exact", ">= exact", float(eps) / exact, float(eps) >= exact))
    return {"row": "RA2", "note": "conversion function reached through the class, because the bound call in "
            "get_epsilon fails (RA1)", "cells": cells, "checks": checks}


def row_ra3(F) -> dict:
    def numeric(q: float, sigma: float, alpha: int) -> float:
        def f(z):
            logp0 = -z * z / (2 * sigma**2) - 0.5 * math.log(2 * math.pi * sigma**2)
            logr = np.logaddexp(math.log1p(-q), math.log(q) + (2 * z - 1) / (2 * sigma**2))
            return math.exp(logp0 + alpha * logr)

        lo, hi = -10 * sigma, alpha + 10 * sigma
        val, _ = integrate.quad(f, lo, hi, points=[0.0, 1.0, float(alpha)], epsabs=0, epsrel=1e-12, limit=800)
        return math.log(val) / (alpha - 1)

    cells, worst = [], 0.0
    for q in (0.01, 0.1):
        for sigma in (1.0, 2.0):
            for alpha in (2, 4, 8, 16):
                got = float(F.compute_rdp(q=q, noise_multiplier=sigma, steps=1, orders=[alpha])[0])
                ref = numeric(q, sigma, alpha)
                r = abs(got - ref) / abs(ref)
                worst = max(worst, r)
                cells.append({"q": q, "sigma": sigma, "alpha": alpha, "shipped": got, "numeric": ref, "rel_diff": r})
    return {"row": "RA3", "cells": cells, "worst_relative_difference": worst,
            "checks": [chk("worst relative difference at most 1e-6", "<= 1e-6", worst, worst <= 1e-6)]}


def row_ra4(F) -> dict:
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    cells, checks = [], []
    true_mult = math.sqrt(2 * math.log(1.25 / DELTA)) / 1.0
    exact = gauss_eps(1.0 / true_mult, DELTA)
    for s in (0.1, 1.0, 10.0):
        g = F.DPMechanism("gaussian", 1.0, DELTA, sensitivity=s)
        passed = float(g.get_rdp_scale())
        rdp = F.compute_rdp(q=1.0, noise_multiplier=passed, steps=1, orders=alphas)
        eps_passed = float(F.RDP_Accountant.get_privacy_spent(orders=alphas, rdp=rdp, delta=DELTA)[0])
        rdp_t = F.compute_rdp(q=1.0, noise_multiplier=true_mult, steps=1, orders=alphas)
        eps_true_mult = float(F.RDP_Accountant.get_privacy_spent(orders=alphas, rdp=rdp_t, delta=DELTA)[0])
        lap = F.DPMechanism("laplace", 1.0, DELTA, sensitivity=s)
        cells.append({"sensitivity": s, "gaussian_get_rdp_scale": passed,
                      "true_noise_multiplier_sigma_over_sensitivity": true_mult,
                      "epsilon_the_accountant_would_report": eps_passed,
                      "epsilon_with_true_multiplier_rdp": eps_true_mult, "exact_epsilon_one_release": exact,
                      "laplace_get_rdp_scale": float(lap.get_rdp_scale()),
                      "laplace_scale_over_sensitivity": float(lap.dp.scale) / s})
        checks.append(chk(f"s={s}: get_rdp_scale is sigma, which is s times the true noise multiplier",
                          s * true_mult, passed, rel(passed, s * true_mult) < 1e-9))
    c10 = [c for c in cells if c["sensitivity"] == 10.0][0]
    checks.append(chk("s=10: reported epsilon below the exact epsilon", "< exact",
                      c10["epsilon_the_accountant_would_report"],
                      c10["epsilon_the_accountant_would_report"] < c10["exact_epsilon_one_release"]))
    checks.append(chk("laplace get_rdp_scale equals scale / sensitivity", True,
                      all(rel(c["laplace_get_rdp_scale"], c["laplace_scale_over_sensitivity"]) < 1e-12 for c in cells),
                      all(rel(c["laplace_get_rdp_scale"], c["laplace_scale_over_sensitivity"]) < 1e-12 for c in cells)))
    return {"row": "RA4", "note": "hypothetical. The frame cannot construct the accountant (CD2). Values come from "
            "the conversion function called through the class with the multiplier the frame would pass.",
            "cells": cells, "checks": checks}


def row_ra5(F) -> dict:
    acc = F.RDP_Accountant(dp_param=1.0, alpha=2.0, dp_mechanism="laplace", args=acct_args())
    out = {"row": "RA5", "api": "fedml.core.dp.budget_accountant.rdp_accountant.RDP_Accountant"}
    for label in ("before_step", "after_step"):
        if label == "after_step":
            acc.step(noise_multiplier=1.0, sample_rate=1.0)
        try:
            out[label] = {"value": repr(acc.get_epsilon(1e-5)), "exception": None}
        except Exception as e:  # recorded, not hidden
            out[label] = {"exception": exc_str(e)}
    e1_, e2_ = out["before_step"]["exception"], out["after_step"]["exception"]
    out["checks"] = [
        chk("after a step raises TypeError", "TypeError", e2_, bool(e2_) and e2_.startswith("TypeError")),
        chk("raises only after a step (pre-registered detail)", "no exception before a step", e1_,
            not (bool(e1_) and e1_.startswith("TypeError"))),
    ]
    return out


def row_ob1(F) -> dict:
    ns = types.SimpleNamespace(enable_dp=True, mechanism_type="gaussian", epsilon=0.1, delta=0.1,
                               sensitivity=1.0, dp_type="ldp")
    inst = F.FedMLDifferentialPrivacy()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            inst.init(ns)
        exc = None
    except Exception as e:  # recorded, not hidden
        exc = exc_str(e)
    has_add_noise = hasattr(F.FedMLDifferentialPrivacy, "add_noise")
    return {"row": "OB1", "init_with_the_test_file_arguments": exc, "class_has_add_noise": has_add_noise,
            "checks": [
                chk("init raises AttributeError for dp_solution_type", "AttributeError", exc,
                    bool(exc) and exc.startswith("AttributeError") and "dp_solution_type" in exc),
                chk("FedMLDifferentialPrivacy has no add_noise", False, has_add_noise, not has_add_noise),
            ]}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="small run for tests")
    args = ap.parse_args()
    runs = 500 if args.quick else 20_000
    trials = 2_000 if args.quick else 200_000
    k_n = 20_000 if args.quick else 200_000

    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    F = load_fedml()
    block_network()
    t0 = time.time()
    rows: list[dict] = []

    def add(r):
        rows.extend(r if isinstance(r, list) else [r])
        print(f"{rows[-1]['row']} done {time.time() - t0:.0f}s", flush=True)

    add(row_g1(F, runs))
    add(row_g2(F))
    add(row_lp1(F, runs))
    add(row_lp2(F, runs, trials))
    add(row_lp3(F))
    add(row_k1(F))
    add(row_k2(F, k_n))
    add(row_k3(F))
    add(row_k4(F))
    add(row_cd1(F, runs))
    add(row_cd2(F))
    add(row_cd3(F))
    add(row_ld(F, runs, trials, False))
    add(row_ld(F, runs, trials, True))
    add(row_ra1(F))
    add(row_ra2(F))
    add(row_ra3(F))
    add(row_ra4(F))
    add(row_ra5(F))
    add(row_ob1(F))

    def all_checks(r):
        yield from r.get("checks", [])

    summary = {
        "checks_total": sum(len(list(all_checks(r))) for r in rows),
        "checks_failed": [
            {"row": r["row"], **c} for r in rows for c in all_checks(r) if not c["held"]
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "feddisclosure.fedml_dp_audit.v1",
        "quick": args.quick,
        "environment": {
            "fedml": F.version,
            "fedml_dp_tree_sha256": F.tree_sha256,
            "fedml_modules_loaded": F.loaded_modules,
            "fedml_init_executed": False,
            "network_blocked_during_rows": True,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "torch": __import__("torch").__version__,
            "python": platform.python_version(),
        },
        "parameters": {"dim": DIM, "delta": DELTA, "noise_runs": runs, "trials_per_world": trials,
                       "compute_noise_samples": k_n, "seed": SEED},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary,
        "rows": rows,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "fedml_dp_audit.json").write_text(json.dumps(doc, indent=2, default=float) + "\n")
    print(f"wrote {out / 'fedml_dp_audit.json'} in {doc['runtime_seconds']}s, "
          f"{len(summary['checks_failed'])} of {summary['checks_total']} checks failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
