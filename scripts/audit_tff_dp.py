#!/usr/bin/env python3
"""Measurement of the differential privacy aggregators of TensorFlow Federated 0.87.0 (v0.16.9).

Written after the pre-registration commit 7a22b12 (docs/preregistration/v0.16-tff-dp.md). Every row, number
and tolerance comes from that file. The script changes none of them. Anything that differs from the file goes in
DEVIATIONS below, is copied to the JSON evidence, and is appended to the "Deviation log" of the pre-registration.

Rows
  F1 to F6  gaussian_fixed (noise against configuration, one client's effect, cohort size, state, weights, randomness)
  A1 to A8  gaussian_adaptive and dp_aggregator (parameters, event, clip update, delivered noise, count noise, cohort, composition)
  V, T      environment and run time

Usage: python scripts/audit_tff_dp.py [--out DIR] [--quick]
Full mode writes reports/results/privacy/v0160/tff_dp/tff-0.87.0/tff_dp_audit.json.
Quick mode (few runs, for tests) requires --out. The noise of TFF is not seedable from Python (row F6), so a rerun
reproduces verdicts and not the noise values.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

PREREG_FILE = "docs/preregistration/v0.16-tff-dp.md"
PREREG_COMMIT = "7a22b12"
BASE_SEED = 20261126
OUT_NAME = "tff_dp_audit.json"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "results" / "privacy" / "v0160" / "tff_dp" / "tff-0.87.0"
EXPECTED_VERSIONS = {"tensorflow-federated": "0.87.0", "tensorflow": "2.14.1", "tensorflow-privacy": "0.9.0",
                     "dp-accounting": "0.4.3", "numpy": "1.25.2", "jax": "0.4.14", "jaxlib": "0.4.14", "scipy": "1.9.3"}
CLIP0 = 0.1
LR = 0.2
D_NOISE = 100_000
Z80 = NormalDist().inv_cdf(0.8)
E1 = 1.0

# Deviations from the pre-registration. Each entry is a dict with "date", "item" and "reason".
DEVIATIONS: list[dict] = [
    {"date": "2026-11-26", "item": "Order of work and test runs",
     "reason": "The script was written after the pre-registration commit 7a22b12. Two quick runs (few calls, output in a temporary directory outside the repository, not evidence) were made before the full run. Their statistical checks failed or held at random because the run counts are small, which is expected. They changed no prediction, tolerance or configuration."},
    {"date": "2026-11-26", "item": "Row F2 case e model structure",
     "reason": "The first quick run raised an exception in case e because a plain StructType of two tensors is not accepted by the TFP clipping code. The script uses StructWithPythonType with the list container for the two tensors. The statement tested is the one pre-registered."},
    {"date": "2026-11-26", "item": "Row F5 third check, arithmetic slip in the pre-registration",
     "reason": "The pre-registration says the output norm for a client of weight 1.0 and update norm 0.5 is 0.5 / (n W) with W 14, which is 0.0035714. When the first weight is 1.0 and the other nine stay 1.0 the weights sent sum to 10, so the arithmetic gives 0.005. The script checks the number as pre-registered, which is expected to fail, and adds a check with W 10. The failure stays in the record."},
    {"date": "2026-11-26", "item": "Checks added beyond the pre-registered statements",
     "reason": "A recomputation of each pre-registered power figure (F1, A4, A5, A7) and the extra F5 check of deviation 3. None replaces a pre-registered check. Case a of A1 does not assert the absence of a warning, because the pre-registration states that only for case b."},
    {"date": "2026-11-26", "item": "Replication rule as implemented",
     "reason": "When a statistical check of a row fails, the row is run once more with the seed offset 5,000,000 added to the seed rule, which only changes the input vectors and the call to tf.random.set_seed. A mismatch is called found when the same check fails in both runs. The seed does not control the noise (row F6), so the replication has fresh noise."},
]


# ---------------------------------------------------------------- helpers
def chk(name, expected, observed, held, kind="exact"):
    return {"check": name, "expected": expected, "observed": observed, "held": bool(held), "kind": kind}


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0 else abs(a)


def zdelta(z: float, sb: float) -> float:
    return (z ** -2 - (2 * sb) ** -2) ** -0.5


def rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(math.sqrt(np.mean(x * x)))


def moments(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    m, s = float(x.mean()), float(x.std())
    zs = (x - m) / s
    return {"mean": m, "std": s, "skew": float(np.mean(zs ** 3)), "excess_kurtosis": float(np.mean(zs ** 4) - 3.0)}


def se_rms(n_values: int) -> float:
    return 1.0 / math.sqrt(2 * n_values)


def mde_rms(n_values: int, tol: float) -> float:
    return tol + Z80 * se_rms(n_values)


def flat(r) -> np.ndarray:
    """Concatenate every array in a nested result into one float64 vector."""
    if isinstance(r, np.ndarray):
        return r.astype(np.float64).ravel()
    if hasattr(r, "_asdict"):
        r = list(r._asdict().values())
    elif isinstance(r, dict):
        r = list(r.values())
    if isinstance(r, (list, tuple)):
        return np.concatenate([flat(v) for v in r])
    return np.asarray(r, dtype=np.float64).ravel()


def find_key(obj, key):
    """First value stored under the name `key` in nested mappings and named tuples."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif hasattr(obj, "_asdict"):
        return find_key(obj._asdict(), key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def clip_of(out) -> float:
    return float(find_key(out.measurements, "clip"))


def recover_fraction(c1: float, c0: float, lr: float = LR, target: float = 0.5) -> float:
    return target - math.log(c1 / c0) / lr


def unit(d: int, k: int = 0) -> np.ndarray:
    v = np.zeros(d, np.float32)
    v[k] = 1.0
    return v


def kill_children() -> None:
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            if ppid == me:
                os.kill(int(p.name), signal.SIGKILL)
        except (OSError, ValueError, IndexError):
            continue


class Ctx:
    """Holds the TFF modules and the sizes for one run."""

    def __init__(self, quick: bool):
        import tensorflow as tf
        import tensorflow_federated as tff
        from tensorflow_federated.python.aggregators import differential_privacy as dpm
        self.tf, self.tff, self.dpm = tf, tff, dpm
        self.quick = quick
        self.D = 5_000 if quick else D_NOISE
        self.calls = 1 if quick else 3
        self.a4_calls = 2 if quick else 10
        self.r_a5a = 60 if quick else 2000
        self.r_a5b = 60 if quick else 2000
        self.r_a5c = 100 if quick else 8000
        self.factory = tff.aggregators.DifferentiallyPrivateFactory

    def vt(self, d: int):
        return self.tff.TensorType(np.float32, [d])

    def fixed(self, z, n, c, d):
        p = self.factory.gaussian_fixed(float(z), float(n), float(c)).create(self.vt(d))
        return p, p.initialize()

    def adaptive(self, z, n, d, **kw):
        p = self.factory.gaussian_adaptive(float(z), float(n), **kw).create(self.vt(d))
        return p, p.initialize()


def zero_clients(m: int, d: int):
    z = np.zeros(d, np.float32)
    return [z] * m


def clip_clients(m: int, u: float, d: int, below: float = 0.05, above: float = 0.2):
    k = int(round(u * m))
    return [below * unit(d)] * k + [above * unit(d)] * (m - k)


def noise_calls(p, state, values, calls: int, extra=None):
    outs, clips = [], []
    for _ in range(calls):
        o = p.next(state, values, extra) if extra is not None else p.next(state, values)
        outs.append(flat(o.result))
        try:
            clips.append(clip_of(o))
        except (TypeError, ValueError):
            clips.append(None)
    return outs, clips


def finish(rid, configuration, measured, checks, **extra):
    d = {"id": rid, "configuration": configuration, "measured": measured, "checks": checks,
         "prediction_held": all(c["held"] for c in checks)}
    d.update(extra)
    return d


# ---------------------------------------------------------------- F rows
def row_f1(c: Ctx, offset=0) -> dict:
    tol = 0.005
    cfgs = {"F1a": (0.5, 1.0, 10.0), "F1b": (1.0, 2.0, 10.0), "F1c": (2.0, 0.5, 20.0)}
    checks, measured = [], {}
    for name, (z, cl, n) in cfgs.items():
        tf_seed(c, 100 + offset)
        p, s = c.fixed(z, n, cl, c.D)
        outs, _ = noise_calls(p, s, zero_clients(int(n), c.D), c.calls)
        x = np.concatenate(outs)
        pred = z * cl / n
        mo = moments(x)
        ratio = rms(x) / pred
        o1 = p.next(s, zero_clients(int(n), c.D))
        ev = c.dpm.extract_dp_event_from_state(o1.state)
        measured[name] = {"z": z, "clip": cl, "n": n, "N": int(x.size), "predicted_rms": pred, "rms": rms(x), "ratio": ratio,
                          "moments": mo, "event": repr(ev), "mde80_relative": mde_rms(x.size, tol)}
        sem = rms(x) / math.sqrt(x.size)
        checks += [
            chk(f"{name}: RMS ratio within {tol} of 1", f"1 +- {tol}", ratio, abs(ratio - 1) <= tol, "stat"),
            chk(f"{name}: absolute skewness at most 0.02", "<= 0.02", mo["skew"], abs(mo["skew"]) <= 0.02, "stat"),
            chk(f"{name}: absolute excess kurtosis at most 0.04", "<= 0.04", mo["excess_kurtosis"], abs(mo["excess_kurtosis"]) <= 0.04, "stat"),
            chk(f"{name}: absolute mean at most 4 standard errors", f"<= {4 * sem}", mo["mean"], abs(mo["mean"]) <= 4 * sem, "stat"),
            chk(f"{name}: reported event is Gaussian with multiplier z", z, getattr(ev, "noise_multiplier", None),
                type(ev).__name__ == "GaussianDpEvent" and rel(float(ev.noise_multiplier), z) <= 1e-6),
        ]
    checks.append(chk("power: smallest deviation found with 80% power at N 300,000 (pre-registered 0.609%)", "0.609% +- 0.002%",
                      mde_rms(300_000, tol), abs(mde_rms(300_000, tol) - 0.00609) <= 0.00002))
    return finish("F1", {"configs": cfgs, "calls": c.calls, "D": c.D, "tolerance": tol}, measured, checks)


def tf_seed(c: Ctx, k: int) -> None:
    c.tf.random.set_seed(BASE_SEED + k)
    np.random.seed(BASE_SEED + k)


def _rand_unit(rng, d):
    v = rng.standard_normal(d)
    return (v / np.linalg.norm(v)).astype(np.float64)


def row_f2(c: Ctx, offset=0) -> dict:
    d, n = 1000, 10.0
    rng = np.random.RandomState(BASE_SEED + 200 + offset)
    others = [(0.5 * _rand_unit(rng, d)).astype(np.float32) for _ in range(9)]
    v = _rand_unit(rng, d)
    p, s = c.fixed(0.0, n, 1.0, d)

    def out_with(target):
        return flat(p.next(s, others + [target]).result)

    def diff(a, b):
        return float(np.linalg.norm(out_with(a) - out_with(b)))

    zero = np.zeros(d, np.float32)
    cases = {
        "a": ("norm 3 against zero", diff((3 * v).astype(np.float32), zero), 0.1),
        "b": ("norm 3 against its opposite (replace-one)", diff((3 * v).astype(np.float32), (-3 * v).astype(np.float32)), 0.2),
        "c": ("norm 0.5 against zero", diff((0.5 * v).astype(np.float32), zero), 0.05),
        "d": ("norm exactly 1 against zero", diff(unit(d), zero), 0.1),
    }
    # case e: two tensors of 300 and 700 coordinates, one global norm
    vt2 = c.tff.types.StructWithPythonType([c.tff.TensorType(np.float32, [300]), c.tff.TensorType(np.float32, [700])], list)
    p2 = c.factory.gaussian_fixed(0.0, n, 1.0).create(vt2)
    s2 = p2.initialize()
    oth2 = [[o[:300], o[300:]] for o in others]
    t3 = (3 * v).astype(np.float32)

    def out2(t):
        return flat(p2.next(s2, oth2 + [t]).result)

    de = float(np.linalg.norm(out2([t3[:300], t3[300:]]) - out2([np.zeros(300, np.float32), np.zeros(700, np.float32)])))
    cases["e"] = ("two tensors, global norm 3 against zero", de, 0.1)
    checks, measured = [], {}
    for k, (label, got, pred) in cases.items():
        measured[k] = {"label": label, "norm_of_difference": got, "predicted": pred, "relative_error": rel(got, pred)}
        checks.append(chk(f"case {k}: {label}", f"{pred} within 1e-5 relative", got, rel(got, pred) <= 1e-5))
    measured["per_tensor_clip_would_give"] = 0.1 * math.sqrt(2)
    return finish("F2", {"z": 0.0, "clip": 1.0, "n": n, "m": 10, "coordinates": d}, measured, checks)


def row_f3(c: Ctx, offset=0) -> dict:
    tol, n, z, cl = 0.005, 10.0, 1.0, 1.0
    checks, measured = [], {}
    for m in (5, 20):
        tf_seed(c, 300 + m + offset)
        p, s = c.fixed(z, n, cl, c.D)
        outs, _ = noise_calls(p, s, zero_clients(m, c.D), c.calls)
        x = np.concatenate(outs)
        ratio = rms(x) / (z * cl / n)
        ratio_m = rms(x) / (z * cl / m)
        # noise free
        d = 1000
        rng = np.random.RandomState(BASE_SEED + 300 + m + offset)
        vv = (0.5 * _rand_unit(rng, d)).astype(np.float32)
        p0, s0 = c.fixed(0.0, n, cl, d)
        o = flat(p0.next(s0, [vv] * m).result)
        nf_pred = 0.5 * m / n
        oth = [(0.5 * _rand_unit(rng, d)).astype(np.float32) for _ in range(m - 1)]
        big = (3 * _rand_unit(rng, d)).astype(np.float32)
        sens = float(np.linalg.norm(flat(p0.next(s0, oth + [big]).result) - flat(p0.next(s0, oth + [np.zeros(d, np.float32)]).result)))
        delivered = rms(x) / sens
        measured[f"m{m}"] = {"N": int(x.size), "rms": rms(x), "ratio_to_z_C_over_n": ratio, "ratio_to_z_C_over_m": ratio_m,
                             "noise_free_norm": float(np.linalg.norm(o)), "noise_free_predicted": nf_pred,
                             "sensitivity": sens, "delivered_multiplier": delivered}
        checks += [
            chk(f"m {m}: noise RMS ratio to z C / n within {tol} of 1", f"1 +- {tol}", ratio, abs(ratio - 1) <= tol, "stat"),
            chk(f"m {m}: positive control, ratio to z C / m separates from 1", "< 0.7 for m 5, > 1.3 for m 20", ratio_m,
                (ratio_m < 0.7) if m == 5 else (ratio_m > 1.3), "stat"),
            chk(f"m {m}: noise-free output norm equals (m / n) * 0.5", f"{nf_pred} within 1e-5 relative", float(np.linalg.norm(o)),
                rel(float(np.linalg.norm(o)), nf_pred) <= 1e-5),
            chk(f"m {m}: sensitivity of the output to one client of norm 3 is C / n", f"{cl / n} within 1e-5 relative", sens, rel(sens, cl / n) <= 1e-5),
            chk(f"m {m}: delivered multiplier within {tol} of z", f"{z} +- {tol}", delivered, abs(delivered - z) <= tol, "stat"),
        ]
    return finish("F3", {"z": z, "clip": cl, "n": n, "m": [5, 20], "calls": c.calls, "D": c.D}, measured, checks)


def epsilon_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        for attr in ("id", "attr", "arg", "name"):
            v = getattr(node, attr, None)
            if isinstance(v, str) and "epsilon" in v.lower():
                names.append(v)
    return names


def event_fields(ev) -> list[str]:
    attrs = getattr(type(ev), "__attrs_attrs__", None)
    if attrs is not None:
        return [a.name for a in attrs]
    import dataclasses
    if dataclasses.is_dataclass(ev):
        return [f.name for f in dataclasses.fields(ev)]
    return sorted(vars(ev).keys())


def row_f4(c: Ctx, offset=0) -> dict:
    n, z, cl, d = 10.0, 1.0, 1.0, 16
    p, s = c.fixed(z, n, cl, d)
    try:
        c.dpm.extract_dp_event_from_state(s)
        raised = "no exception"
    except Exception as e:  # recorded
        raised = type(e).__name__
    o10 = p.next(s, zero_clients(10, d))
    o5 = p.next(s, zero_clients(5, d))
    o5b = p.next(o5.state, zero_clients(5, d))
    ev10, ev5, ev5b = (c.dpm.extract_dp_event_from_state(o.state) for o in (o10, o5, o5b))
    site = Path(c.tff.__file__).resolve().parent
    tfp_dir = Path(importlib.import_module("tensorflow_privacy").__file__).resolve().parent / "privacy" / "dp_query"
    six = [site / "python/aggregators/differential_privacy.py", site / "python/learning/model_update_aggregator.py"] + \
          [tfp_dir / f for f in ("gaussian_query.py", "normalized_query.py", "quantile_estimator_query.py", "quantile_adaptive_clip_sum_query.py")]
    scan = {str(f.relative_to(f.anchor)): epsilon_names(f) for f in six}
    whole = {}
    for f in site.rglob("*.py"):
        try:
            nm = epsilon_names(f)
        except (SyntaxError, UnicodeDecodeError):
            continue
        if nm:
            whole[str(f.relative_to(site))] = sorted(set(nm))
    measured = {"initial_state_extraction": raised, "event_m10": repr(ev10), "event_m5": repr(ev5), "event_round2_m5": repr(ev5b),
                "event_fields": event_fields(ev10), "epsilon_names_six_files": scan,
                "whole_package_epsilon_names_reported_not_judged": whole}
    checks = [
        chk("extract_dp_event_from_state on the initial state raises ExtractingDpEventFromInitialStateError",
            "ExtractingDpEventFromInitialStateError", raised, raised == "ExtractingDpEventFromInitialStateError"),
        chk("event after m 10 is Gaussian with multiplier 1.0", 1.0, repr(ev10),
            type(ev10).__name__ == "GaussianDpEvent" and rel(float(ev10.noise_multiplier), z) <= 1e-6),
        chk("event after m 5 is Gaussian with multiplier 1.0 and equals the m 10 event", 1.0, repr(ev5),
            type(ev5).__name__ == "GaussianDpEvent" and rel(float(ev5.noise_multiplier), z) <= 1e-6 and
            rel(float(ev5.noise_multiplier), float(ev10.noise_multiplier)) <= 1e-12),
        chk("event has only the field noise_multiplier and is not a PoissonSampledDpEvent", ["noise_multiplier"], event_fields(ev10),
            event_fields(ev10) == ["noise_multiplier"] and type(ev10).__name__ != "PoissonSampledDpEvent"),
        chk("after a second round the event is one Gaussian event with the same multiplier", repr(ev5), repr(ev5b),
            type(ev5b).__name__ == "GaussianDpEvent" and rel(float(ev5b.noise_multiplier), z) <= 1e-6),
        chk("no name containing epsilon in the six files outside docstrings and comments", 0, sum(len(v) for v in scan.values()),
            sum(len(v) for v in scan.values()) == 0),
    ]
    return finish("F4", {"z": z, "clip": cl, "n": n}, measured, checks)


def row_f5(c: Ctx, offset=0) -> dict:
    tol, n, z, cl = 0.005, 10.0, 1.0, 1.0
    tff = c.tff

    def build(zz, d):
        fac = tff.aggregators.MeanFactory(value_sum_factory=c.factory.gaussian_fixed(float(zz), n, cl))
        p = fac.create(c.vt(d), tff.TensorType(np.float32))
        return p, p.initialize()

    sets = {"equal": [2.0] * 10, "unequal": [5.0] + [1.0] * 9}
    checks, measured = [], {}
    for name, w in sets.items():
        tf_seed(c, 500 + offset + len(name))
        p, s = build(z, c.D)
        outs = [flat(p.next(s, zero_clients(10, c.D), [np.float32(x) for x in w]).result) for _ in range(c.calls)]
        x = np.concatenate(outs)
        W = sum(w)
        pred = z * cl / (n * W)
        ratio = rms(x) / pred
        measured[name] = {"W": W, "predicted_rms": pred, "rms": rms(x), "ratio": ratio, "N": int(x.size)}
        checks.append(chk(f"{name} weights: noise RMS ratio to z C / (n W) within {tol} of 1", f"1 +- {tol}", ratio, abs(ratio - 1) <= tol, "stat"))
    d = 1000
    p0, s0 = build(0.0, d)
    W = 14.0
    w = [np.float32(x) for x in sets["unequal"]]
    zero = np.zeros(d, np.float32)
    v = (0.5 * unit(d)).astype(np.float32)
    o5 = float(np.linalg.norm(flat(p0.next(s0, [v] + [zero] * 9, w).result)))
    w1 = [np.float32(1.0)] + w[1:]
    o1 = float(np.linalg.norm(flat(p0.next(s0, [v] + [zero] * 9, w1).result)))
    # weight 1 for the first client changes W to 10, so the pre-registered 0.5 / (n W) uses W of the set run
    w1_total = 1.0 + 9.0
    measured["noise_free"] = {"weight5_norm": o5, "weight5_predicted": cl / (n * W), "weight1_norm": o1,
                              "weight1_predicted_with_W_10": 0.5 / (n * w1_total), "weight1_predicted_with_W_14": 0.5 / (n * W)}
    checks += [
        chk("weight 5 and update norm 0.5: output norm is C / (n W) (the weighted value is clipped)", f"{cl / (n * W)} within 1e-5 relative", o5, rel(o5, cl / (n * W)) <= 1e-5),
        chk("weight 1 and update norm 0.5: output norm is 0.5 / (n W) with W 14, 0.0035714 (as pre-registered)", "0.0035714 within 1e-5 relative",
            o1, rel(o1, 0.5 / (n * W)) <= 1e-5),
        chk("beyond the pre-registration: the same output norm against 0.5 / (n W) with W the sum of the weights actually sent (10)",
            f"{0.5 / (n * w1_total)} within 1e-5 relative", o1, rel(o1, 0.5 / (n * w1_total)) <= 1e-5),
    ]
    return finish("F5", {"weights": sets, "z": z, "clip": cl, "n": n, "calls": c.calls, "D": c.D}, measured, checks)


def corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])


def child_seed_main(seed: int) -> int:
    c = Ctx(True)
    c.tf.random.set_seed(seed)
    np.random.seed(seed)
    p, s = c.fixed(1.0, 10.0, 1.0, 10_000)
    x = np.asarray(p.next(s, zero_clients(10, 10_000)).result, np.float32)
    print(json.dumps({"digest": hashlib.sha256(x.tobytes()).hexdigest()}), flush=True)
    kill_children()
    os._exit(0)


def row_f6(c: Ctx, offset=0) -> dict:
    d = c.D
    tf_seed(c, 600 + offset)
    p, s = c.fixed(1.0, 10.0, 1.0, d)
    v = zero_clients(10, d)
    x1 = flat(p.next(s, v).result)
    x2 = flat(p.next(s, v).result)
    thr = 4.0 / math.sqrt(d)
    r12, rlag = corr(x1, x2), corr(x1[:-1], x1[1:])
    y = x1[1:] + x1[:-1]
    rctl = corr(y[:-1], y[1:])
    digests = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--child-seed", "12345"], capture_output=True, text=True, timeout=600)
        try:
            digests.append(json.loads(r.stdout.strip().splitlines()[-1])["digest"])
        except (ValueError, IndexError):
            digests.append(f"error: {(r.stderr or r.stdout)[-200:]}")
    measured = {"correlation_between_calls": r12, "lag1_autocorrelation": rlag, "positive_control_lag1": rctl, "threshold": thr,
                "vectors_equal": bool(np.array_equal(x1, x2)), "seed_digests": digests, "mde80_correlation": (4.0 + Z80) / math.sqrt(d)}
    checks = [
        chk("two calls from the same state give different noise", "not equal", measured["vectors_equal"], not measured["vectors_equal"]),
        chk(f"absolute correlation between the two calls at most {thr:.5f}", f"<= {thr}", r12, abs(r12) <= thr, "stat"),
        chk(f"absolute lag-1 autocorrelation at most {thr:.5f}", f"<= {thr}", rlag, abs(rlag) <= thr, "stat"),
        chk("positive control: lag-1 autocorrelation of neighbouring sums above 0.4", "> 0.4", rctl, rctl > 0.4, "stat"),
        chk("two fresh processes with the same global seed give different digests (prediction: the seed does not control the noise)",
            "different", digests, len(set(digests)) == 2 and not any(str(x).startswith("error") for x in digests)),
    ]
    return finish("F6", {"z": 1.0, "clip": 1.0, "n": 10.0, "D": d}, measured, checks)


# ---------------------------------------------------------------- A rows
A1_CASES = [
    ("a", 1.0, 100.0, None, ("value", 1.005037815259212, 5.0, None)),
    ("b", 1.0, 100.0, 10.0, ("value", 1.0012523486435176, 10.0, False)),
    ("c", 0.5, 10.0, None, ("value", 0.5773502691896257, 0.5, None)),
    ("d", 0.3, 200.0, 2.0, ("value", 0.3008473263383401, 2.0, None)),
    ("e", 1.0, 12.0, None, ("value", 1.809068067466581, 0.6000000000000001, None)),
    ("f", 10.0, 100.0, None, ("error", "ValueError")),
    ("g", 9.99, 100.0, None, ("value", 223.43905770087412, 5.0, True)),
    ("h", 1.0, 100.0, 0.5, ("error", "ValueError")),
    ("i", 0.0, 100.0, None, ("value", 0.0, 0.0, None)),
    ("j", 0.0, 100.0, 3.0, ("value", 0.0, 3.0, None)),
    ("k", 1.0, 100.0, 0.0, ("error", "ValueError")),
    ("l", 1.0, 100.0, 0.55, ("value", 2.4003967925959158, 0.55, True)),
]


def st_get(state, *path):
    cur = state
    for p in path:
        cur = getattr(cur, p) if not isinstance(cur, dict) else cur[p]
    return cur


def row_a1(c: Ctx, offset=0) -> dict:
    checks, measured = [], {"cases": {}}
    for cid, z, n, sb, pred in A1_CASES:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                v, s = c.dpm.adaptive_clip_noise_params(z, n, sb)
                got = ("value", float(v), float(s), len(w) > 0)
            except Exception as e:
                got = ("error", type(e).__name__)
        measured["cases"][cid] = {"input": [z, n, sb], "output": list(got)}
        if pred[0] == "error":
            checks.append(chk(f"case {cid} ({z}, {n}, {sb}) raises {pred[1]}", pred[1], got[1] if got[0] == "error" else got, got == pred))
        else:
            ok = got[0] == "value" and rel(got[1], pred[1]) <= 1e-12 and rel(got[2], pred[2]) <= 1e-12
            if pred[3] is not None:
                ok = ok and got[3] == pred[3]
            checks.append(chk(f"case {cid} ({z}, {n}, {sb})", list(pred), list(got), ok))
    p, s = c.adaptive(1.0, 100.0, 8)
    qn = s.query_state.numerator_state
    vals = {"noise_multiplier": float(qn.noise_multiplier), "sum_stddev": float(qn.sum_state.stddev), "clip": float(qn.sum_state.l2_norm_clip),
            "count_stddev": float(qn.quantile_estimator_state.below_estimate_state.numerator_state.stddev),
            "denominator_count": float(qn.quantile_estimator_state.below_estimate_state.denominator),
            "denominator_sum": float(s.query_state.denominator)}
    zv = zdelta(1.0, 5.0)
    preds = {"noise_multiplier": zv, "sum_stddev": zv * 0.1, "clip": 0.1, "count_stddev": 5.0, "denominator_count": 100.0, "denominator_sum": 100.0}
    measured["initial_state"] = {"read": vals, "predicted": preds}
    checks.append(chk("initial state of gaussian_adaptive(1.0, 100.0)", preds, vals, all(rel(vals[k], preds[k]) <= 1e-6 for k in preds)))
    sig = inspect.signature(c.factory.gaussian_adaptive)
    defaults = {k: sig.parameters[k].default for k in ("initial_l2_norm_clip", "target_unclipped_quantile", "learning_rate", "clipped_count_stddev")}
    measured["signature_defaults"] = defaults
    checks.append(chk("signature defaults", {"initial_l2_norm_clip": 0.1, "target_unclipped_quantile": 0.5, "learning_rate": 0.2, "clipped_count_stddev": None},
                      defaults, defaults == {"initial_l2_norm_clip": 0.1, "target_unclipped_quantile": 0.5, "learning_rate": 0.2, "clipped_count_stddev": None}))
    return finish("A1", {"cases": [[a, b, c_, d] for a, b, c_, d, _ in A1_CASES]}, measured, checks)


def row_a2(c: Ctx, offset=0) -> dict:
    grid = [(1.0, 100.0, None), (1.0, 100.0, 10.0), (0.5, 10.0, None), (0.3, 200.0, 2.0), (1.0, 12.0, None)]
    checks, measured = [], {}
    for z, n, sb in grid:
        kw = {} if sb is None else {"clipped_count_stddev": sb}
        tf_seed(c, 700 + offset)
        p, s = c.adaptive(z, n, 8, **kw)
        o = p.next(s, zero_clients(int(n), 8))
        ev = c.dpm.extract_dp_event_from_state(o.state)
        sbv = 0.05 * n if sb is None else sb
        zv = zdelta(z, sbv)
        ok_struct = type(ev).__name__ == "ComposedDpEvent" and len(ev.events) == 2 and all(type(e).__name__ == "GaussianDpEvent" for e in ev.events)
        if ok_struct:
            m1, m2 = float(ev.events[0].noise_multiplier), float(ev.events[1].noise_multiplier)
            comp = (m1 ** -2 + m2 ** -2) ** -0.5
        else:
            m1 = m2 = comp = float("nan")
        key = f"z{z}_n{n}_sb{sb}"
        measured[key] = {"event": repr(ev), "first": m1, "second": m2, "composed": comp, "predicted_first": zv, "predicted_second": 2 * sbv}
        checks += [
            chk(f"{key}: composed event of two Gaussian events", "ComposedDpEvent of 2", repr(ev), ok_struct),
            chk(f"{key}: first multiplier equals the value multiplier", zv, m1, ok_struct and rel(m1, zv) <= 1e-6),
            chk(f"{key}: second multiplier equals 2 times the count stddev", 2 * sbv, m2, ok_struct and rel(m2, 2 * sbv) <= 1e-6),
            chk(f"{key}: composed multiplier equals z", z, comp, ok_struct and rel(comp, z) <= 1e-5),
        ]
    return finish("A2", {"grid": grid}, measured, checks)


def row_a3(c: Ctx, offset=0) -> dict:
    d, n = 8, 100.0
    checks, measured = [], {"u": {}}
    p, s = c.adaptive(0.0, n, d)
    c0 = float(s.query_state.numerator_state.sum_state.l2_norm_clip)
    for u in (0.0, 0.3, 0.5, 0.7, 1.0):
        o = p.next(s, clip_clients(100, u, d))
        c1 = clip_of(o)
        pred = 0.1 * math.exp(-LR * (u - 0.5))
        norm = float(np.linalg.norm(flat(o.result)))
        npred = u * 0.05 + (1 - u) * 0.1
        measured["u"][str(u)] = {"next_clip": c1, "predicted": pred, "recovered_fraction": recover_fraction(c1, c0), "output_norm": norm, "output_predicted": npred}
        checks += [chk(f"u {u}: next clip equals 0.1 exp(-0.2 (u - 0.5))", f"{pred} within 1e-5 relative", c1, rel(c1, pred) <= 1e-5),
                   chk(f"u {u}: output norm equals the sum clipped at the old clip", f"{npred} within 1e-5 relative", norm, rel(norm, npred) <= 1e-5)]
    pb, sb_ = c.adaptive(0.0, 10.0, d, initial_l2_norm_clip=0.125)
    at = unit(d) * np.float32(0.125)
    above = unit(d) * np.nextafter(np.float32(0.125), np.float32(1.0))
    c_at = clip_of(pb.next(sb_, [at] * 10))
    c_above = clip_of(pb.next(sb_, [above] * 10))
    measured["boundary"] = {"norm_equal_to_clip": c_at, "norm_next_float32_above": c_above}
    checks += [chk("boundary: norm equal to the clip counts as below it", f"{0.125 * math.exp(-0.1)} within 1e-5 relative", c_at, rel(c_at, 0.125 * math.exp(-0.1)) <= 1e-5),
               chk("boundary: next float32 above counts as above it", f"{0.125 * math.exp(0.1)} within 1e-5 relative", c_above, rel(c_above, 0.125 * math.exp(0.1)) <= 1e-5)]
    return finish("A3", {"z": 0.0, "n": n, "m": 100, "initial_clip": 0.1}, measured, checks)


def row_a4(c: Ctx, offset=0) -> dict:
    tol = 0.005
    checks, measured = [], {}
    for name, (z, n) in (("A4a", (1.0, 100.0)), ("A4b", (1.0, 12.0))):
        tf_seed(c, 800 + int(n) + offset)
        p, s = c.adaptive(z, n, c.D)
        c0 = float(s.query_state.numerator_state.sum_state.l2_norm_clip)
        values = zero_clients(int(n), c.D)
        outs, c1s = noise_calls(p, s, values, c.a4_calls)
        zv = zdelta(z, 0.05 * n)
        x = np.concatenate(outs)
        pred = zv * c0 / n
        ratio_old = rms(x) / pred
        r_new = [rms(o) / (zv * c1 / n) for o, c1 in zip(outs, c1s)]
        measured[name] = {"z": z, "n": n, "N": int(x.size), "z_value": zv, "C0": c0, "rms": rms(x), "predicted_rms": pred, "ratio_to_old_clip": ratio_old,
                          "mean_ratio_to_new_clip": float(np.mean(r_new)), "mean_C1_over_C0": float(np.mean([c1 / c0 for c1 in c1s])),
                          "C1": c1s, "zs_delivered": rms(x) * n / c0, "mde80_relative": mde_rms(x.size, tol)}
        checks += [chk(f"{name}: RMS ratio to z_value C0 / n within {tol} of 1", f"1 +- {tol}", ratio_old, abs(ratio_old - 1) <= tol, "stat"),
                   chk(f"{name}: mean ratio to z_value C1 / n at least 1.05 (positive control)", ">= 1.05", float(np.mean(r_new)), float(np.mean(r_new)) >= 1.05, "stat")]
    checks.append(chk("power: smallest deviation found with 80% power at N 1,000,000 (pre-registered 0.560%)", "0.560% +- 0.002%",
                      mde_rms(1_000_000, tol), abs(mde_rms(1_000_000, tol) - 0.00560) <= 0.00002))
    return finish("A4", {"calls": c.a4_calls, "D": c.D, "tolerance": tol}, measured, checks)


def row_a5(c: Ctx, offset=0) -> dict:
    checks, measured = [], {}
    specs = [("A5a", 1.0, 100.0, {}, c.r_a5a), ("A5b", 1.0, 10.0, {"clipped_count_stddev": 5.0}, c.r_a5b), ("A5c", 1.0, 12.0, {}, c.r_a5c)]
    for name, z, n, kw, r in specs:
        tf_seed(c, 900 + int(n) + offset)
        p, s = c.adaptive(z, n, 8, **kw)
        c0 = float(s.query_state.numerator_state.sum_state.l2_norm_clip)
        values = clip_clients(int(n), 0.5, 8)
        ratios = np.array([clip_of(p.next(s, values)) / c0 for _ in range(r)])
        f = np.array([recover_fraction(x * c0, c0) for x in ratios])
        S = n * (f - 0.5)
        lo, hi = math.exp(-0.1), math.exp(0.1)
        at_bound = float(np.mean((np.abs(ratios / lo - 1) <= 1e-5) | (np.abs(ratios / hi - 1) <= 1e-5)))
        inside = bool(np.all((ratios >= lo * (1 - 1e-5)) & (ratios <= hi * (1 + 1e-5))))
        measured[name] = {"z": z, "n": n, "R": r, "S_mean": float(S.mean()), "S_std": float(S.std(ddof=1)), "share_at_bound": at_bound, "all_inside_bounds": inside}
        if name == "A5a":
            checks += [chk("A5a: absolute mean of S at most 0.45", "<= 0.45", float(S.mean()), abs(S.mean()) <= 0.45, "stat"),
                       chk("A5a: standard deviation of S between 4.70 and 5.30", "[4.70, 5.30]", float(S.std(ddof=1)), 4.70 <= S.std(ddof=1) <= 5.30, "stat")]
        elif name == "A5b":
            checks += [chk("A5b: share of calls at a bound is 0.3173 within 0.04", "0.3173 +- 0.04", at_bound, abs(at_bound - 0.3173) <= 0.04, "stat"),
                       chk("A5b: every call has C1 over C0 inside [exp(-0.1), exp(0.1)]", True, inside, inside)]
        else:
            checks += [chk("A5c: absolute mean of S at most 0.0268", "<= 0.0268", float(S.mean()), abs(S.mean()) <= 0.0268, "stat"),
                       chk("A5c: standard deviation of S between 0.576 and 0.624", "[0.576, 0.624]", float(S.std(ddof=1)), 0.576 <= S.std(ddof=1) <= 0.624, "stat")]
    pa = 0.06 + Z80 / math.sqrt(2 * 1999)
    pc = 0.04 + Z80 / math.sqrt(2 * 7999)
    checks += [chk("power: A5a smallest deviation found with 80% power (pre-registered 7.3%)", "7.3% +- 0.1%", pa, abs(pa - 0.073) <= 0.001),
               chk("power: A5c smallest deviation found with 80% power (pre-registered 4.7%)", "4.7% +- 0.1%", pc, abs(pc - 0.047) <= 0.001)]
    return finish("A5", {"specs": [[a, b, c_, d, e] for a, b, c_, d, e in specs]}, measured, checks)


def row_a6(c: Ctx, offset=0) -> dict:
    d, n, tol = 8, 100.0, 0.005
    checks, measured = [], {"noise_free": {}}
    p, s = c.adaptive(0.0, n, d)
    c0 = float(s.query_state.numerator_state.sum_state.l2_norm_clip)
    cases = [(50, 1.0, 0.1 * math.exp(-0.05)), (50, 0.0, 0.1 * math.exp(0.05)), (50, 0.5, 0.1), (200, 1.0, 0.1 * math.exp(-0.1))]
    for m, u, pred in cases:
        o = p.next(s, clip_clients(m, u, d))
        c1 = clip_of(o)
        measured["noise_free"][f"m{m}_u{u}"] = {"next_clip": c1, "predicted": pred, "recovered_fraction": recover_fraction(c1, c0),
                                                "unclamped_fraction": 0.5 + (m / n) * (u - 0.5), "output_norm": float(np.linalg.norm(flat(o.result)))}
        checks.append(chk(f"m {m}, u {u}: next clip", f"{pred} within 1e-5 relative", c1, rel(c1, pred) <= 1e-5))
    on = measured["noise_free"]["m50_u1.0"]["output_norm"]
    checks.append(chk("m 50, u 1.0: output norm is 0.025", "0.025 within 1e-5 relative", on, rel(on, 0.025) <= 1e-5))
    tf_seed(c, 1000 + offset)
    pn, sn = c.adaptive(1.0, n, c.D)
    zv = zdelta(1.0, 5.0)
    outs, _ = noise_calls(pn, sn, zero_clients(50, c.D), c.calls)
    x = np.concatenate(outs)
    pred = zv * 0.1 / n
    ratio, ratio_m = rms(x) / pred, rms(x) / (zv * 0.1 / 50)
    measured["noisy_m50"] = {"N": int(x.size), "rms": rms(x), "predicted_rms": pred, "ratio_to_n": ratio, "ratio_to_m": ratio_m}
    checks += [chk(f"noisy m 50: RMS ratio to z_value C0 / n within {tol} of 1", f"1 +- {tol}", ratio, abs(ratio - 1) <= tol, "stat"),
               chk("noisy m 50: positive control, ratio to z_value C0 / m is below 0.6", "< 0.6", ratio_m, ratio_m < 0.6, "stat")]
    return finish("A6", {"n": n, "m": [50, 200], "calls": c.calls, "D": c.D}, measured, checks)


def row_a7(a4: dict, a5: dict) -> dict:
    checks, measured = [], {}
    for name, a4k, a5k, tol, n in (("n100", "A4a", "A5a", 0.01, 100), ("n12", "A4b", "A5c", 0.025, 12)):
        zs = a4["measured"][a4k]["zs_delivered"]
        sb = a5["measured"][a5k]["S_std"]
        comp = (zs ** -2 + (2 * sb) ** -2) ** -0.5
        zv = zdelta(1.0, 0.05 * n)
        a, b = zv ** -2, (2 * 0.05 * n) ** -2
        se_zs = se_rms(a4["measured"][a4k]["N"])
        se_sb = 1 / math.sqrt(2 * (a5["measured"][a5k]["R"] - 1))
        se = math.sqrt((a / (a + b) * se_zs) ** 2 + (b / (a + b) * se_sb) ** 2)
        measured[name] = {"zs_delivered": zs, "count_stddev_measured": sb, "composed": comp, "se": se, "mde80": tol + Z80 * se}
        checks.append(chk(f"{name}: composed delivered multiplier within {tol} of 1", f"1 +- {tol}", comp, abs(comp - 1) <= tol, "stat"))
        pre = {"n100": 0.0106, "n12": 0.0296}[name]
        checks.append(chk(f"{name}: power, smallest deviation found with 80% power (pre-registered {pre})", f"{pre} +- 0.0005", tol + Z80 * se, abs(tol + Z80 * se - pre) <= 0.0005))
    return finish("A7", "derived from A4 and A5, no new runs", measured, checks)


def row_a8(c: Ctx, offset=0) -> dict:
    tol, n, d = 0.005, 100.0, 8
    tff = c.tff
    checks, measured = [], {}

    def build(z, dd, zeroing=True):
        p = tff.learning.dp_aggregator(float(z), n, zeroing=zeroing).create(c.vt(dd))
        return p, p.initialize()

    tf_seed(c, 1100 + offset)
    p, s = build(1.0, c.D)
    outs = [flat(p.next(s, zero_clients(100, c.D)).result) for _ in range(c.calls)]
    x = np.concatenate(outs)
    zv = zdelta(1.0, 5.0)
    pred = zv * 0.1 / n
    measured["noise"] = {"N": int(x.size), "rms": rms(x), "predicted": pred, "ratio": rms(x) / pred}
    checks.append(chk(f"dp_aggregator noise RMS ratio to z_value C0 / n within {tol} of 1", f"1 +- {tol}", rms(x) / pred, abs(rms(x) / pred - 1) <= tol, "stat"))
    p0, s0 = build(0.0, d)
    for u, pr in ((0.5, 0.1), (1.0, 0.1 * math.exp(-0.1))):
        c1 = clip_of(p0.next(s0, clip_clients(100, u, d)))
        measured[f"noise_free_u{u}"] = {"next_clip": c1, "predicted": pr}
        checks.append(chk(f"dp_aggregator noise-free u {u}: next clip", f"{pr} within 1e-5 relative", c1, rel(c1, pr) <= 1e-5))
    vals = [0.2 * unit(d)] * 50 + [1000.0 * unit(d)] * 50
    o_on = p0.next(s0, vals)
    zc = find_key(o_on.measurements, "zeroed_count")
    c_on, n_on = clip_of(o_on), float(np.linalg.norm(flat(o_on.result)))
    pf, sf = build(0.0, d, zeroing=False)
    o_off = pf.next(sf, vals)
    c_off, n_off = clip_of(o_off), float(np.linalg.norm(flat(o_off.result)))
    measured["zeroing"] = {"on": {"next_clip": c_on, "output_norm": n_on, "zeroed_count": None if zc is None else float(zc)},
                           "off": {"next_clip": c_off, "output_norm": n_off}}
    checks += [chk("zeroing on: next clip is 0.1", "0.1 within 1e-5 relative", c_on, rel(c_on, 0.1) <= 1e-5),
               chk("zeroing on: output norm is 0.05", "0.05 within 1e-5 relative", n_on, rel(n_on, 0.05) <= 1e-5),
               chk("zeroing on: zeroed count is 50", 50, None if zc is None else float(zc), zc is not None and abs(float(zc) - 50) < 1e-6),
               chk("zeroing off: next clip is 0.1 exp(0.1)", f"{0.1 * math.exp(0.1)} within 1e-5 relative", c_off, rel(c_off, 0.1 * math.exp(0.1)) <= 1e-5),
               chk("zeroing off: output norm is 0.1", "0.1 within 1e-5 relative", n_off, rel(n_off, 0.1) <= 1e-5)]
    return finish("A8", {"n": n, "calls": c.calls, "D": c.D}, measured, checks)


def row_v() -> dict:
    got = {k: importlib.metadata.version(k) for k in EXPECTED_VERSIONS}
    return finish("V", {"expected": EXPECTED_VERSIONS}, {"installed": got, "python": platform.python_version(), "platform": platform.platform()},
                  [chk("installed versions equal the pre-registered ones", EXPECTED_VERSIONS, got, got == EXPECTED_VERSIONS)])


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
    return repr(o)


def run_row(fn, *args, **kw) -> dict:
    try:
        return fn(*args, **kw)
    except Exception as e:  # a failed row stays in the record
        return {"id": fn.__name__.replace("row_", "").upper(), "measured": {"exception": f"{type(e).__name__}: {e}"[:400]},
                "checks": [chk("row ran", "no exception", f"{type(e).__name__}: {e}"[:300], False)], "prediction_held": False}


def main() -> int:
    if "--child-seed" in sys.argv:
        return child_seed_main(int(sys.argv[sys.argv.index("--child-seed") + 1]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true", help="fewer runs, for tests, requires --out")
    args = ap.parse_args()
    if args.quick and not args.out:
        ap.error("--quick requires --out so that the committed evidence is not overwritten")
    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    c = Ctx(args.quick)
    doc = {"schema": "feddisclosure.tff_dp_audit.v1", "quick": args.quick, "preregistration": PREREG_FILE, "preregistration_commit": PREREG_COMMIT,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "environment": {k: importlib.metadata.version(k) for k in EXPECTED_VERSIONS} | {"python": platform.python_version(), "platform": platform.platform()},
           "parameters": {"base_seed": BASE_SEED, "D": c.D, "calls": c.calls, "a4_calls": c.a4_calls, "R_a5": [c.r_a5a, c.r_a5b, c.r_a5c]},
           "deviations": DEVIATIONS, "rows": [], "replication": []}

    def save():
        rows = doc["rows"]
        checks = [(r["id"], ch) for r in rows for ch in r.get("checks", [])]
        doc["summary"] = {"rows_total": len(rows), "rows_held": sum(1 for r in rows if r.get("prediction_held")),
                          "rows_failed": [r["id"] for r in rows if r.get("prediction_held") is False],
                          "checks_total": len(checks), "checks_held": sum(1 for _, ch in checks if ch["held"]),
                          "checks_failed": [{"row": rid, **ch} for rid, ch in checks if not ch["held"]],
                          "mismatch_found_rows": [r["id"] for r in rows if r.get("mismatch_found")]}
        doc["runtime_seconds"] = round(time.time() - t0, 1)
        (out_dir / OUT_NAME).write_text(json.dumps(doc, indent=2, default=enc) + "\n")

    def add(row, fn=None, *a):
        # one replication when a statistical check fails, a mismatch is found only when the same check fails twice
        if row.get("prediction_held") is False and fn is not None:
            failed_stat = {ch["check"] for ch in row["checks"] if not ch["held"] and ch.get("kind") == "stat"}
            if failed_stat:
                rep = run_row(fn, *a, offset=5_000_000)
                rep["id"] = row["id"] + "-replication"
                doc["replication"].append(rep)
                again = {ch["check"] for ch in rep["checks"] if not ch["held"]}
                row["mismatch_found"] = bool(failed_stat & again)
        doc["rows"].append(row)
        print(f"{row['id']} done {time.time() - t0:.0f}s held={row.get('prediction_held')}", flush=True)
        save()

    add(row_v())
    rows = {}
    for fn in (row_f1, row_f2, row_f3, row_f4, row_f5, row_f6, row_a1, row_a2, row_a3, row_a4, row_a5, row_a6):
        r = run_row(fn, c)
        rows[r["id"]] = r
        add(r, fn, c)
    if "A4" in rows and "A5" in rows and "exception" not in rows["A4"]["measured"] and "exception" not in rows["A5"]["measured"]:
        add(row_a7(rows["A4"], rows["A5"]))
    add(run_row(row_a8, c), row_a8, c)
    total = time.time() - t0
    add({"id": "T", "measured": {"total_seconds": round(total, 1)}, "checks": [chk("whole script under 1500 seconds", "< 1500", round(total, 1), total < 1500 or args.quick)],
         "prediction_held": total < 1500 or args.quick})
    s = doc["summary"]
    print(f"wrote {out_dir / OUT_NAME} in {doc['runtime_seconds']}s, {s['rows_held']} of {s['rows_total']} rows held, "
          f"{s['checks_held']} of {s['checks_total']} checks held, mismatch rows {s['mismatch_found_rows']}", flush=True)
    kill_children()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
