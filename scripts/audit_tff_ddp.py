#!/usr/bin/env python3
"""Measurement of the distributed DP aggregator of TensorFlow Federated 0.87.0 (v0.16.11).

Written after the pre-registration commit b32e31c (docs/preregistration/v0.16-tff-ddp.md). Every row, number
and tolerance comes from that file. The script changes none of them. Anything that differs from the file goes in
DEVIATIONS below, is copied to the JSON evidence, and is appended to the "Deviation log" of the pre-registration.

Rows
  A  delivered noise: A1 mechanism and rotation grid, A2 bit widths, A3 rounding growth with a dense input
  B  docstring against code (local standard deviation)
  C  fewer or more clients than expected
  D  sensitivity of one client above the clip, beta 0 and beta above 0
  E  modular wrap: E1 constructor limits, E2 wrap of noise, E3 too many clients with small bits
  F  adaptive clip
  G  accounting surface
  H  randomness
  I  implied epsilon table (Gaussian approximation, no new calls)
  J  tff.learning.ddp_secure_aggregator
  V, T  environment and run time

Usage: python scripts/audit_tff_ddp.py [--out DIR] [--quick]
Full mode writes reports/results/privacy/v0160/tff_ddp/tff-0.87.0/tff_ddp_audit.json.
Quick mode (few runs, for tests) requires --out. The noise of TFF is not seedable from Python (row H), so a rerun
reproduces verdicts and not the noise values. The helper functions that compute predictions use numpy and the standard
library only and never import TFF, so the tests of those helpers run without it.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
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

PREREG_FILE = "docs/preregistration/v0.16-tff-ddp.md"
PREREG_COMMIT = "b32e31c"
BASE_SEED = 20261127
OUT_NAME = "tff_ddp_audit.json"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "results" / "privacy" / "v0160" / "tff_ddp" / "tff-0.87.0"
EXPECTED_VERSIONS = {"tensorflow-federated": "0.87.0", "tensorflow": "2.14.1", "tensorflow-privacy": "0.9.0",
                     "dp-accounting": "0.4.3", "numpy": "1.25.2", "jax": "0.4.14", "jaxlib": "0.4.14", "scipy": "1.9.3"}
D = 2048
SKELLAM, DGAUSS = "distributed_skellam", "distributed_dgauss"
Z80 = NormalDist().inv_cdf(0.8)
ND = NormalDist()
TOL = 0.007

# Deviations from the pre-registration. Each entry is a dict with "date", "item" and "reason".
DEVIATIONS: list[dict] = [
    {"date": "2026-11-26", "item": "Order of work and test runs",
     "reason": "The script was written after the pre-registration commit b32e31c. Quick runs (few calls, output in a temporary directory outside the repository, not evidence) were made before the full run. Their statistical checks failed or held at random because the run counts are small. They changed no prediction, tolerance or configuration."},
    {"date": "2026-11-26", "item": "Row I, entry for Skellam at 4 bits",
     "reason": "The pre-registration lists A2 Skellam at 4 and 5 bits with the sensitivity of D at 5 bits. Row D has no 4 bit run, so a 4 bit entry would pair a noise measured at one scale with a sensitivity of another. The script drops the 4 bit entry and keeps the 5 bit one at the mean and at the bound."},
    {"date": "2026-11-26", "item": "Row B check 4, location of the docstring sentence",
     "reason": "The pre-registration says the docstring of the class contains the sentence about the local noise standard deviation. The sentence may be in the docstring of `__init__` instead. The script searches both and records where it was found."},
    {"date": "2026-11-26", "item": "Replication rule as implemented",
     "reason": "When a statistical check of a row fails, the row is run once more with the seed offset 5,000,000 added to the seed rule, which changes only the input vectors and the seed given to tf.random.set_seed. A mismatch is called found when the same check fails in both runs. The seed does not control the noise (row H), so the replication has fresh noise."},
    {"date": "2026-11-26", "item": "Execution context",
     "reason": "The first quick run used the default local C++ execution context. A call of process.next failed there with the gRPC error 'Cannot embed a federated value' for every configuration, while initialize ran. The test C++ execution context (tff.backends.test.set_sync_test_cpp_execution_context) runs the same call in under 0.1 s. All rows use the test context. Row V records the outcome of one call in the default context. This is a finding about the runtime and not a change to any prediction. The pre-registration says secure sum is a plain sum in the local runtime, which holds in the test context."},
    {"date": "2026-11-26", "item": "Row J cannot run a call",
     "reason": "A call of the process built by tff.learning.ddp_secure_aggregator(1.0, 20) fails under the test C++ execution context with a protobuf DecodeError raised in replace_secure_intrinsics_with_bodies, and under the default context with the same error as the factory. Construction and initialize run. By the rule of the brief, a configuration that cannot run is a finding. The row records both errors and its prediction is not tested. The direct factory with the settings of the wrapper was not isolated further."},
    {"date": "2026-11-26", "item": "Checks added beyond the pre-registered statements",
     "reason": "A recomputation of each pre-registered model number by the helper functions (checks named 'helper reproduces'), and the report of the largest norm and the share above C in row D. None replaces a pre-registered check."},
]

# Pre-registered model numbers, for the 'helper reproduces' checks.
PRE_A2 = {  # bits: (scale, local scaled stddev, skellam ratio, dgauss ratio, dgauss integer scale)
    4: (1.2836, 0.40592, 1.00000, 2.42095, 1), 5: (3.7070, 1.17224, 0.99994, 1.67489, 2),
    8: (32.088, 10.14705, 0.99988, 1.08361, 11), 12: (513.997, 162.540, 0.99988, 1.00270, 163)}
PRE_A3 = {16: 1.00000, 5: 1.0209, 4: 1.0636}
PRE_D = {  # (bits, beta flag): (mean, tolerance, bound)
    (16, 0): (1.00000, 1e-5, 1.00119), (8, 0): (1.00765, 0.0012, 1.30434), (5, 0): (1.4540, 0.02, 3.6341),
    (8, 1): (1.0077, 0.0012, 1.0153), (5, 1): (1.4537, 0.02, 1.6939)}
PRE_E2 = {0.3: (0.60814, 0.74456), 0.7: (0.22783, 0.35147)}
PRE_E3 = {4: (3.9993, 0.02), 16: (8.96, 0.4), 64: (8.99, 0.2)}
BETA = math.exp(-0.5)


# ---------------------------------------------------------------- helpers (no TFF)
def chk(name, expected, observed, held, kind="exact"):
    return {"check": name, "expected": expected, "observed": observed, "held": bool(held), "kind": kind}


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0 else abs(a)


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


def trapz(y, x) -> float:
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)


def corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])


def zdelta(z: float, sb: float) -> float:
    return (z ** -2 - (2 * sb) ** -2) ** -0.5


def flat(r) -> np.ndarray:
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
    """First value stored under the name `key` in nested mappings, named tuples and sequences."""
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


def walk_names(obj, path=""):
    """Yield (path, name, leaf or None) for every key or field of a nested state or measurement."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_names(v, f"{path}/{k}")
            yield (f"{path}/{k}", str(k), v if not isinstance(v, (dict, list, tuple)) and not hasattr(v, "_asdict") else None)
    elif hasattr(obj, "_asdict"):
        yield from walk_names(obj._asdict(), path)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk_names(v, f"{path}[{i}]")


def epsilon_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        for attr in ("id", "attr", "arg", "name"):
            v = getattr(node, attr, None)
            if isinstance(v, str) and "epsilon" in v.lower():
                names.append(v)
    return names


def kstd(p: float) -> float:
    return ND.inv_cdf(1 - p / 2)


def scale_fn(b: int, n: int, C: float, sig_l: float, k: float, dim: int = D, cap: float = 1e6) -> float:
    num = math.sqrt(2.0 ** (2 * b) - n * k * k)
    den = 2 * k * math.sqrt(C * C * n * n / dim + n * sig_l ** 2)
    return min(num / den, cap)


def min_bits(n: int, p: float = 1e-4):
    k = kstd(p)
    for b in range(1, 23):
        if b >= math.log2(n) and 2 ** (2 * b) >= n * k * k:
            return b
    return None


def dg_pmf(s: int, width: int = 14):
    xs = np.arange(-int(width * s) - 5, int(width * s) + 6)
    w = np.exp(-xs.astype(float) ** 2 / (2 * float(s) ** 2))
    return xs, w / w.sum()


def conv_n(xs, pmf, m: int):
    """Exact m fold convolution of a probability table by FFT."""
    L = m * (pmf.size - 1) + 1
    size = 1 << (L - 1).bit_length()
    f = np.fft.rfft(pmf, size)
    out = np.fft.irfft(f ** m, size)[:L]
    out = np.clip(out, 0.0, None)
    lo = m * xs[0]
    return np.arange(lo, lo + L), out / out.sum()


def skellam_sum_pmf(lam_total: float, kmax: int):
    j = np.arange(0, int(lam_total + 12 * math.sqrt(lam_total) + 60))
    lg = np.array([math.lgamma(v + 1) for v in j])
    pois = np.exp(-lam_total + j * math.log(lam_total) - lg)
    kmax = min(kmax, pois.size - 1)
    pos = np.array([np.sum(pois[kk:] * pois[: pois.size - kk]) for kk in range(kmax + 1)])
    ks = np.arange(-kmax, kmax + 1)
    out = np.concatenate([pos[:0:-1], pos])
    return ks, out / out.sum()


def wrap_rms2(xs, pmf, h: int) -> tuple[float, float]:
    w = ((xs + h) % (2 * h)) - h
    return float(np.sum(pmf * w.astype(float) ** 2)), float(np.sum(pmf * xs.astype(float) ** 2))


def gauss_wrap_rms2(sd: float, h: int) -> tuple[float, float]:
    xs = np.arange(-int(14 * sd) - 3, int(14 * sd) + 4)
    w = np.exp(-xs.astype(float) ** 2 / (2 * sd * sd))
    w /= w.sum()
    return wrap_rms2(xs, w, h)


def local_scaled(n: int, z: float, C: float, b: int, p: float, dim: int = D) -> tuple[float, float, float]:
    """(scale, local scaled stddev as float32, k) for a configuration."""
    k = kstd(p)
    sig = z * C / math.sqrt(n)
    sc = scale_fn(b, n, C, sig, k, dim)
    return sc, float(np.float32(sig * sc)), k


def noise_ratio_pred(mech: str, n: int, z: float, C: float, b: int, p: float, m: int | None = None) -> float:
    """Predicted RMS of the delivered noise over z C for m reporting clients (zero inputs): the wrap model of the pre-registration."""
    m = n if m is None else m
    sc, s, _ = local_scaled(n, z, C, b, p)
    h = 2 ** (b - 1)
    if mech == SKELLAM:
        lam = m * s * s / 2
        if lam < 5000:
            xs, pm = skellam_sum_pmf(lam, int(h * 3 + 8 * math.sqrt(2 * lam) + 20))
            w2, _ = wrap_rms2(xs, pm, h)
        else:
            w2, _ = gauss_wrap_rms2(math.sqrt(m) * s, h)
    else:
        xs, pmf = dg_pmf(math.ceil(s))
        xs_n, pm_n = conv_n(xs, pmf, m)
        w2, _ = wrap_rms2(xs_n, pm_n, h)
    return math.sqrt(w2) / (z * C * sc)


def clip_alt_ratio(n: int, z: float, C: float, b: int, p: float) -> float:
    """Ratio that a saturating field would give instead of a wrap (Gaussian model)."""
    sc, s, _ = local_scaled(n, z, C, b, p)
    h = 2 ** (b - 1)
    sd = math.sqrt(n) * s
    xs = np.arange(-int(14 * sd), int(14 * sd) + 1)
    w = np.exp(-xs.astype(float) ** 2 / (2 * sd * sd))
    w /= w.sum()
    return math.sqrt(float(np.sum(w * np.minimum(np.abs(xs), h - 1) ** 2))) / (z * C * sc)


def rho_exact(tau: float) -> float:
    """E[f (1 - f)] for y Gaussian with standard deviation tau, f the fractional part."""
    if tau > 50:
        return 1 / 6
    ys = np.linspace(-8 * tau, 8 * tau, 2000001)
    pdf = np.exp(-ys ** 2 / (2 * tau * tau)) / (tau * math.sqrt(2 * math.pi))
    f = ys - np.floor(ys)
    return trapz(pdf * f * (1 - f), ys)


def dense_ratio_pred(n: int, z: float, C: float, b: int, p: float, norm: float) -> float:
    sc, s, _ = local_scaled(n, z, C, b, p)
    return math.sqrt(1 + rho_exact(norm * sc / math.sqrt(D)) / (s * s))


def inflated_bound(C: float, scale: float, beta: float, dim: int = D) -> float:
    g = math.sqrt(dim) / scale
    b1 = C + g
    if beta == 0:
        return b1
    bt = math.sqrt(2 * math.log(1 / beta))
    sq2 = C ** 2 + 0.25 * g * g + bt * (1 / scale) * (C + 0.5 * g)
    return min(b1, math.sqrt(sq2))


def e3_wrap_norm(n: int, b: int, m: int, C: float = 1.0, k: float | None = None, clip: bool = False) -> float:
    k = kstd(1e-4) if k is None else k
    sc = scale_fn(b, n, C, 0.0, k)
    hp = 2 ** (b - 1) / sc
    sd = m * C / math.sqrt(D)
    xs = np.linspace(-12 * sd, 12 * sd, 4000001)
    pdf = np.exp(-xs ** 2 / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))
    v = np.minimum(np.abs(xs), hp) if clip else ((xs + hp) % (2 * hp)) - hp
    return math.sqrt(trapz(pdf * v ** 2, xs)) * math.sqrt(D)


def gdelta(mu: float, eps: float) -> float:
    """Exact delta of a Gaussian mechanism with parameter mu (Balle and Wang 2018)."""
    if mu <= 0:
        return 0.0
    Phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))  # noqa: E731
    return max(Phi(-eps / mu + mu / 2) - math.exp(eps) * Phi(-eps / mu - mu / 2), 0.0)


def geps(mu: float, delta: float) -> float:
    if gdelta(mu, 0.0) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while gdelta(mu, hi) > delta:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if gdelta(mu, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


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


def dense_dir(rng, norm: float) -> np.ndarray:
    v = rng.standard_normal(D)
    return (v / np.linalg.norm(v) * norm).astype(np.float32)


# ---------------------------------------------------------------- TFF context
class Ctx:
    """Holds the TFF modules and the sizes for one run."""

    def __init__(self, quick: bool, probe_default: bool = True):
        import tensorflow as tf
        import tensorflow_federated as tff
        from tensorflow_federated.python.aggregators import differential_privacy as dpm
        from tensorflow_federated.python.aggregators import distributed_dp as ddp
        self.tf, self.tff, self.dpm, self.ddp = tf, tff, dpm, ddp
        self.quick = quick
        q = quick
        self.n_a1, self.n_80, self.n_d, self.n_e3 = (3, 2, 3, 2) if q else (120, 80, 80, 20)
        self.chains, self.pairs, self.n_j = (3, 3, 2) if q else (20, 20, 60)
        self.vt = tff.TensorType(np.float32, [D])
        # The default local C++ context rejects the secure sum of this aggregator, so the test C++ context is used for every row.
        self.default_context = self._probe_default() if probe_default else None
        tff.backends.test.set_sync_test_cpp_execution_context()

    def _probe_default(self) -> dict:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fac = self.ddp.DistributedDpSumFactory(noise_multiplier=1.0, expected_clients_per_round=4, bits=16, l2_clip=1.0)
                proc = fac.create(self.vt)
                proc.next(proc.initialize(), [np.zeros(D, np.float32)] * 4)
            return {"call_ran": True, "error": None}
        except Exception as e:
            return {"call_ran": False, "error": f"{type(e).__name__}: {str(e)}"[:200].replace(chr(10), " ")}

    def make(self, z, n, b, C=1.0, mech=SKELLAM, rot="dft", p=1e-4, beta=0.0, auto=False, **kw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fac = self.ddp.DistributedDpSumFactory(noise_multiplier=float(z), expected_clients_per_round=int(n), bits=int(b), l2_clip=float(C),
                                                   modclip_prob=float(p), beta=float(beta), mechanism=mech, rotation_type=rot,
                                                   auto_l2_clip=bool(auto), **kw)
            proc = fac.create(self.vt)
            state = proc.initialize()
        return proc, state, [str(x.message)[:100] for x in w]

    def seed(self, k: int) -> None:
        self.tf.random.set_seed(BASE_SEED + k)
        np.random.seed(BASE_SEED + k)


def zero_clients(m: int):
    z = np.zeros(D, np.float32)
    return [z] * m


def noise_calls(p, state, values, calls: int):
    outs, first = [], None
    for i in range(calls):
        o = p.next(state, values)
        outs.append(flat(o.result))
        if first is None:
            first = o.measurements
    return outs, first


def finish(rid, configuration, measured, checks, **extra):
    d = {"id": rid, "configuration": configuration, "measured": measured, "checks": checks,
         "prediction_held": all(c["held"] for c in checks)}
    d.update(extra)
    return d


def meas(m, key):
    v = find_key(m, key)
    return None if v is None else float(np.asarray(v))


def ratio_checks(name, outs, pred, tol=TOL, extra_moments=False):
    x = np.concatenate(outs)
    r = rms(x) / 1.0
    return x, r


# ---------------------------------------------------------------- row A
def row_a(c: Ctx, offset=0) -> dict:
    checks, measured = [], {"A1": {}, "A2": {}, "A3": {}}
    n, z, C, b = 10, 1.0, 1.0, 16
    # A1
    for j, (mech, rot) in enumerate([(SKELLAM, "dft"), (SKELLAM, "hd"), (DGAUSS, "dft"), (DGAUSS, "hd")]):
        name = f"A1 {mech.split('_')[1]} {rot}"
        c.seed(offset + j)
        p, s, warns = c.make(z, n, b, C, mech, rot)
        outs, m0 = noise_calls(p, s, zero_clients(n), c.n_a1)
        x = np.concatenate(outs)
        pred = noise_ratio_pred(mech, n, z, C, b, 1e-4)
        ratio = rms(x) / (z * C) / pred
        per_call = [rms(o) / (z * C) for o in outs]
        mo = moments(x)
        sem = rms(x) / math.sqrt(x.size)
        sdc = float(np.std(per_call, ddof=1)) if len(per_call) > 1 else float("nan")
        sc, sl, _ = local_scaled(n, z, C, b, 1e-4)
        mm = {k: meas(m0, k) for k in ("scale_factor", "scaled_local_stddev", "padded_dim", "actual_num_clients", "l2_clip")}
        measured["A1"][name] = {"N": int(x.size), "predicted_ratio": pred, "rms_over_zC": rms(x), "ratio_to_prediction": ratio, "moments": mo,
                                "per_call_rms_sd": sdc, "per_call_rms": per_call, "measurements": mm, "warnings": warns,
                                "mde80_relative": mde_rms(x.size, TOL)}
        checks += [
            chk(f"{name}: RMS over z C within {TOL} of the prediction {pred:.5f}", f"1 +- {TOL}", ratio, abs(ratio - 1) <= TOL, "stat"),
            chk(f"{name}: absolute skewness at most 0.02", "<= 0.02", mo["skew"], abs(mo["skew"]) <= 0.02, "stat"),
            chk(f"{name}: absolute excess kurtosis at most 0.04", "<= 0.04", mo["excess_kurtosis"], abs(mo["excess_kurtosis"]) <= 0.04, "stat"),
            chk(f"{name}: absolute mean at most 4 standard errors", f"<= {4 * sem}", mo["mean"], abs(mo["mean"]) <= 4 * sem, "stat"),
            chk(f"{name}: sd of per-call RMS between 0.0116 and 0.0197 (repeated-noise pair check)", "[0.0116, 0.0197]", sdc,
                0.0116 <= sdc <= 0.0197, "stat"),
            chk(f"{name}: scale_factor equals 8223.98 within 1e-5", 8223.98, mm["scale_factor"], rel(mm["scale_factor"], 8223.98) <= 1e-5),
            chk(f"{name}: scaled_local_stddev equals 2600.651 within 1e-5", 2600.651, mm["scaled_local_stddev"], rel(mm["scaled_local_stddev"], 2600.651) <= 1e-5),
            chk(f"{name}: padded_dim 2048, actual_num_clients 10, l2_clip 1.0", [2048, 10, 1.0],
                [mm["padded_dim"], mm["actual_num_clients"], mm["l2_clip"]],
                mm["padded_dim"] == 2048 and mm["actual_num_clients"] == 10 and rel(mm["l2_clip"], 1.0) <= 1e-6),
        ]
    # A2
    for j, b2 in enumerate((4, 5, 8, 12)):
        pre_scale, pre_sl, pre_sk, pre_dg, pre_int = PRE_A2[b2]
        sc, sl, _ = local_scaled(n, z, C, b2, 1e-4)
        for mech, pre in ((SKELLAM, pre_sk), (DGAUSS, pre_dg)):
            name = f"A2 {mech.split('_')[1]} bits {b2}"
            c.seed(10 + 2 * j + (mech == DGAUSS) + offset)
            try:
                p, s, warns = c.make(z, n, b2, C, mech, "dft")
                outs, m0 = noise_calls(p, s, zero_clients(n), c.n_80)
            except Exception as e:  # a configuration that does not run is a finding
                measured["A2"][name] = {"exception": f"{type(e).__name__}: {e}"[:300]}
                checks.append(chk(f"{name}: runs without exception", "no exception", f"{type(e).__name__}: {e}"[:200], False))
                continue
            x = np.concatenate(outs)
            pred = noise_ratio_pred(mech, n, z, C, b2, 1e-4)
            ratio = rms(x) / (z * C) / pred
            mm = {k: meas(m0, k) for k in ("scale_factor", "scaled_local_stddev")}
            measured["A2"][name] = {"N": int(x.size), "predicted_ratio": pred, "preregistered_ratio": pre, "rms_over_zC": rms(x),
                                    "ratio_to_prediction": ratio, "measurements": mm, "warnings": warns, "integer_scale": math.ceil(sl)}
            checks += [
                chk(f"{name}: RMS over z C within {TOL} of the prediction {pre}", f"1 +- {TOL}", ratio, abs(ratio - 1) <= TOL, "stat"),
                chk(f"{name}: scale_factor within 1e-4 of {pre_scale}", pre_scale, mm["scale_factor"], rel(mm["scale_factor"], pre_scale) <= 1e-4),
                chk(f"{name}: scaled_local_stddev within 1e-4 of {pre_sl}", pre_sl, mm["scaled_local_stddev"], rel(mm["scaled_local_stddev"], pre_sl) <= 1e-4),
                chk(f"{name}: helper reproduces the pre-registered ratio within 5e-5 relative", pre, pred, rel(pred, pre) <= 5e-5),
            ]
            if mech == SKELLAM:
                checks.append(chk(f"{name}: no warning at construction and no exception", [], warns, warns == []))
    # A3
    rng = np.random.RandomState(BASE_SEED + 20 + offset)
    for j, b3 in enumerate((16, 5, 4)):
        name = f"A3 skellam bits {b3}"
        c.seed(30 + j + offset)
        v = dense_dir(rng, 0.99)
        truth = n * v.astype(np.float64)
        p, _, warns = c.make(z, n, b3, C, SKELLAM, "dft")
        errs = []
        for _ in range(c.n_80):
            o = p.next(p.initialize(), [v] * n)
            errs.append(flat(o.result) - truth)
        x = np.concatenate(errs)
        pred = dense_ratio_pred(n, z, C, b3, 1e-4, 0.99)
        r = rms(x) / (z * C)
        measured["A3"][name] = {"N": int(x.size), "predicted": pred, "preregistered": PRE_A3[b3], "rms_error_over_zC": r, "warnings": warns,
                                "mde80_absolute": mde_rms(x.size, 0.008)}
        checks += [
            chk(f"{name}: RMS of the error over z C within 0.008 (absolute) of {PRE_A3[b3]}", f"{PRE_A3[b3]} +- 0.008", r, abs(r - PRE_A3[b3]) <= 0.008, "stat"),
            chk(f"{name}: helper reproduces the pre-registered value within 1e-4", PRE_A3[b3], pred, abs(pred - PRE_A3[b3]) <= 1e-4),
        ]
    checks.append(chk("power: smallest deviation found with 80% power at N 245,760 and 163,840 (pre-registered 0.820% and 0.847%)", "0.820% and 0.847% +- 0.002%",
                      [mde_rms(245_760, TOL), mde_rms(163_840, TOL)],
                      abs(mde_rms(245_760, TOL) - 0.0082) <= 0.00002 and abs(mde_rms(163_840, TOL) - 0.00847) <= 0.00002))
    return finish("A", {"n": n, "z": z, "C": C, "D": D, "calls": [c.n_a1, c.n_80, c.n_80], "tolerance": TOL}, measured, checks)


# ---------------------------------------------------------------- row B
def row_b(c: Ctx, offset=0) -> dict:
    checks, measured = [], {}
    pre = {4: 0.99983, 25: 0.99998}
    for j, n in enumerate((4, 25)):
        c.seed(100 + j + offset)
        p, s, warns = c.make(1.0, n, 16, 1.0, SKELLAM, "dft")
        outs, m0 = noise_calls(p, s, zero_clients(n), c.n_80)
        x = np.concatenate(outs)
        pred = noise_ratio_pred(SKELLAM, n, 1.0, 1.0, 16, 1e-4)
        ratio = rms(x) / pred
        doc_reading = rms(x) / (1.0 / math.sqrt(n))
        sc = meas(m0, "scale_factor")
        loc = meas(m0, "scaled_local_stddev") / sc * math.sqrt(n)
        measured[f"n{n}"] = {"N": int(x.size), "predicted": pred, "rms": rms(x), "ratio_to_prediction": ratio, "ratio_to_docstring_reading": doc_reading,
                             "local_stddev_over_zC_times_sqrt_n": loc, "local_stddev_original_units": meas(m0, "scaled_local_stddev") / sc, "warnings": warns}
        checks += [
            chk(f"n {n}: RMS over z C within {TOL} of {pre[n]}", f"1 +- {TOL}", ratio, abs(ratio - 1) <= TOL, "stat"),
            chk(f"n {n}: positive control, RMS over z C / sqrt(n) (docstring reading) above 1.3", f"> 1.3, predicted {math.sqrt(n):.2f}", doc_reading,
                doc_reading > 1.3, "stat"),
            chk(f"n {n}: local stddev from the measurements times sqrt(n) over z C is 1 within 1e-4", 1.0, loc, abs(loc - 1) <= 1e-4),
            chk(f"n {n}: helper reproduces the pre-registered ratio within 5e-5", pre[n], pred, abs(pred - pre[n]) <= 5e-5),
        ]
    cls = c.ddp.DistributedDpSumFactory
    sentence = "local noise stddev = central noise stddev / `expected_clients_per_round`"
    found_in = [nm for nm, doc in (("class", inspect.getdoc(cls) or ""), ("__init__", inspect.getdoc(cls.__init__) or "")) if sentence in " ".join(doc.split())]
    src = " ".join(inspect.getsource(cls._build_aggregation_factory).split())
    has_src = "local_stddev = central_stddev / math.sqrt(self._num_clients)" in src
    measured["text"] = {"docstring_sentence_found_in": found_in, "source_line_found": has_src}
    checks += [chk("the docstring sentence (local stddev = central stddev / expected_clients_per_round) is present (location in the deviation log)", "present", found_in, bool(found_in)),
               chk("the source of _build_aggregation_factory contains central_stddev / math.sqrt(self._num_clients)", "present", has_src, has_src)]
    return finish("B", {"n": [4, 25], "z": 1.0, "C": 1.0, "bits": 16, "calls": c.n_80}, measured, checks)


# ---------------------------------------------------------------- row C
def row_c(c: Ctx, offset=0) -> dict:
    n = 10
    checks, measured = [], {}
    rng = np.random.RandomState(BASE_SEED + 200 + offset)
    half = dense_dir(rng, 0.5)
    big = dense_dir(rng, 3.0)
    p0, s0, _ = c.make(0.0, n, 16, 1.0, SKELLAM, "dft")
    for mech in (SKELLAM, DGAUSS):
        c.seed(200 + (mech == DGAUSS) + offset)
        p, s, warns = c.make(1.0, n, 16, 1.0, mech, "dft")
        for m in (5, 20):
            name = f"{mech.split('_')[1]} m {m}"
            outs, m0 = noise_calls(p, s, zero_clients(m), c.n_80)
            x = np.concatenate(outs)
            pred = noise_ratio_pred(mech, n, 1.0, 1.0, 16, 1e-4, m)
            ratio = rms(x) / pred
            ent = {"N": int(x.size), "predicted": pred, "rms_over_zC": rms(x), "ratio_to_prediction": ratio, "actual_num_clients": meas(m0, "actual_num_clients"),
                   "raw_sqrt_m_over_n": math.sqrt(m / n), "warnings": warns}
            checks.append(chk(f"{name}: RMS over z C within {TOL} of the prediction {pred:.5f}", f"1 +- {TOL}", ratio, abs(ratio - 1) <= TOL, "stat"))
            checks.append(chk(f"{name}: actual_num_clients equals m", m, ent["actual_num_clients"], ent["actual_num_clients"] == m))
            if mech == SKELLAM:
                ctl = rms(x)
                checks.append(chk(f"{name}: positive control, RMS over z C below 0.85 (m 5) or above 1.15 (m 20)", "< 0.85 or > 1.15", ctl,
                                  (ctl < 0.85) if m == 5 else (ctl > 1.15), "stat"))
                o = flat(p0.next(s0, [half] * m).result)
                nf = float(np.linalg.norm(o))
                zero = np.zeros(D, np.float32)
                a = flat(p0.next(s0, [big] + [zero] * (m - 1)).result)
                bq = flat(p0.next(s0, [zero] * m).result)
                sens = float(np.linalg.norm(a - bq))
                ent.update({"noise_free_norm": nf, "noise_free_predicted": 0.5 * m, "sensitivity": sens, "delivered_multiplier": rms(x) / sens,
                            "delivered_multiplier_predicted": pred})
                checks += [
                    chk(f"{name}: noise-free output norm is 0.5 m", f"{0.5 * m} within 1e-4 relative", nf, rel(nf, 0.5 * m) <= 1e-4),
                    chk(f"{name}: sensitivity (norm 3 against zero) is 1.0 within 1e-5 relative", "1.0 within 1e-5", sens, rel(sens, 1.0) <= 1e-5),
                    chk(f"{name}: delivered multiplier (RMS over sensitivity) within {TOL} of {pred:.5f}", f"{pred:.5f} +- {TOL}", rms(x) / sens,
                        rel(rms(x) / sens, pred) <= TOL, "stat"),
                ]
            measured[name] = ent
    return finish("C", {"n": n, "m": [5, 20], "calls": c.n_80}, measured, checks)


# ---------------------------------------------------------------- row D
def row_d(c: Ctx, offset=0) -> dict:
    n = 10
    checks, measured = [], {}
    rng = np.random.RandomState(BASE_SEED + 300 + offset)
    zero = np.zeros(D, np.float32)
    k0 = kstd(1e-4)
    means = {}
    for j, ((b, bf), (pm, tol, pbound)) in enumerate(PRE_D.items()):
        beta = BETA if bf else 0.0
        name = f"b {b} beta {'exp(-0.5)' if bf else '0'}"
        c.seed(300 + j + offset)
        p, _, warns = c.make(0.0, n, b, 1.0, SKELLAM, "dft", beta=beta)
        v = (3 * dense_dir(rng, 1.0)).astype(np.float32)
        norms, m0 = [], None
        for _ in range(c.n_d):
            o = p.next(p.initialize(), [v] + [zero] * (n - 1))
            norms.append(float(np.linalg.norm(flat(o.result))))
            m0 = m0 or o.measurements
        sc = scale_fn(b, n, 1.0, 0.0, k0)
        bound = inflated_bound(1.0, sc, beta)
        mean = float(np.mean(norms))
        means[(b, bf)] = mean
        sf, sil = meas(m0, "scale_factor"), meas(m0, "scaled_inflated_l2")
        ent = {"scale_predicted": sc, "scale_factor": sf, "bound_over_C": bound, "preregistered_bound": pbound, "mean_norm_over_C": mean,
               "preregistered_mean": pm, "largest_norm": float(max(norms)), "share_above_C": float(np.mean(np.array(norms) > 1.0)),
               "sd": float(np.std(norms)), "norms": norms, "scaled_inflated_l2_over_scale_factor": sil / sf, "warnings": warns,
               "mde80_absolute": tol + Z80 * float(np.std(norms)) / math.sqrt(len(norms))}
        if (b, bf) == (8, 0):
            b_out = [float(np.max(np.abs(flat(p.next(p.initialize(), [zero] * n).result)))) for _ in range(5)]
            ent["without_the_client_max_abs_output"] = b_out
            checks.append(chk(f"{name}: output without the client is exactly zero (5 calls)", 0.0, b_out, all(x == 0.0 for x in b_out)))
        measured[name] = ent
        checks += [
            chk(f"{name}: mean norm over C within {tol} of {pm}", f"{pm} +- {tol}", mean, abs(mean - pm) <= tol, "stat"),
            chk(f"{name}: every norm at most the bound {bound:.5f}", f"<= {bound}", float(max(norms)), max(norms) <= bound * (1 + 1e-6)),
            chk(f"{name}: scaled_inflated_l2 over scale_factor equals the bound plus 1e-5 within 1e-4", bound + 1e-5, sil / sf, rel(sil / sf, bound + 1e-5) <= 1e-4),
            chk(f"{name}: scale_factor within 1e-4 of the formula", sc, sf, rel(sf, sc) <= 1e-4),
            chk(f"{name}: helper bound reproduces the pre-registered bound within 5e-5 relative", pbound, bound, rel(bound, pbound) <= 5e-5),
        ]
        if bf:
            checks.append(chk(f"{name}: construction emitted a warning about beta above zero", ">= 1 warning", warns, len(warns) >= 1))
    for b in (8, 5):
        d = abs(means[(b, 1)] - means[(b, 0)])
        checks.append(chk(f"b {b}: mean norm with beta above 0 differs from beta 0 by less than 0.001", "< 0.001", d, d < 0.001, "stat"))
    return finish("D", {"n": n, "z": 0.0, "calls": c.n_d, "target_norm": 3.0}, measured, checks)


# ---------------------------------------------------------------- row E
def row_e(c: Ctx, offset=0) -> dict:
    checks, measured = [], {"E1": {}, "E2": {}, "E3": {}}
    ddp = c.ddp.DistributedDpSumFactory
    expected = {1: 2, 2: 3, 4: 3, 10: 4, 100: 7, 1000: 10, 10000: 14}

    def raises(**kw):
        args = dict(noise_multiplier=1.0, expected_clients_per_round=10, bits=16, l2_clip=1.0)
        args.update(kw)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ddp(**args)
            return "constructed"
        except Exception as e:
            return type(e).__name__
    for n, mb in expected.items():
        got = min_bits(n)
        ok_at = raises(expected_clients_per_round=n, bits=mb)
        below = raises(expected_clients_per_round=n, bits=mb - 1) if mb - 1 >= 1 else "n/a"
        measured["E1"][f"n{n}"] = {"helper_min_bits": got, "constructor_at_min": ok_at, "constructor_one_below": below}
        checks.append(chk(f"E1 n {n}: smallest accepted bits is {mb}, one fewer raises ValueError", [mb, "constructed", "ValueError"], [got, ok_at, below],
                          got == mb and ok_at == "constructed" and below == "ValueError"))
    lim = {"bits0": raises(bits=0), "bits23": raises(bits=23), "p0": raises(modclip_prob=0.0), "p1": raises(modclip_prob=1.0)}
    measured["E1"]["limits"] = lim
    checks.append(chk("E1: bits 0, bits 23, modclip_prob 0 and 1 raise ValueError", ["ValueError"] * 4, list(lim.values()), all(v == "ValueError" for v in lim.values())))
    # E2
    for j, pm in enumerate((0.3, 0.7)):
        name = f"E2 p {pm}"
        c.seed(400 + j + offset)
        p, s, warns = c.make(1.0, 10, 16, 1.0, SKELLAM, "dft", p=pm)
        outs, m0 = noise_calls(p, s, zero_clients(10), c.n_80)
        x = np.concatenate(outs)
        pw, pc = PRE_E2[pm]
        mw, mc = noise_ratio_pred(SKELLAM, 10, 1.0, 1.0, 16, pm), clip_alt_ratio(10, 1.0, 1.0, 16, pm)
        r = rms(x)
        measured["E2"][name] = {"N": int(x.size), "rms_over_zC": r, "wrap_prediction": mw, "clip_alternative": mc, "ratio_to_wrap": r / mw,
                                "ratio_to_clip_alternative": r / mc, "scale_factor": meas(m0, "scale_factor"), "warnings": warns}
        checks += [
            chk(f"{name}: RMS over z C within {TOL} of the wrap value {pw}", f"{pw} +- {TOL}", r / pw, abs(r / pw - 1) <= TOL, "stat"),
            chk(f"{name}: positive control, the saturating value {pc} is outside the tolerance", f"|r/{pc} - 1| > {TOL}", r / pc, abs(r / pc - 1) > TOL, "stat"),
            chk(f"{name}: helper reproduces the pre-registered wrap and clip values within 5e-5", [pw, pc], [mw, mc], rel(mw, pw) <= 5e-5 and rel(mc, pc) <= 5e-5),
        ]
    # E3
    rng = np.random.RandomState(BASE_SEED + 410 + offset)
    v = dense_dir(rng, 1.0)
    k0 = kstd(1e-4)
    cap = math.sqrt(D) * 2 ** 7 / scale_fn(8, 4, 1.0, 0.0, k0)
    p, _, warns = c.make(0.0, 4, 8, 1.0, SKELLAM, "dft")
    for j, m in enumerate((4, 16, 64)):
        c.seed(420 + j + offset)
        try:
            norms = [float(np.linalg.norm(flat(p.next(p.initialize(), [v] * m).result))) for _ in range(c.n_e3)]
        except Exception as e:
            measured["E3"][f"m{m}"] = {"exception": f"{type(e).__name__}: {e}"[:300]}
            checks.append(chk(f"E3 m {m}: runs without exception", "no exception", f"{type(e).__name__}: {e}"[:200], False))
            continue
        pw, tol = PRE_E3[m]
        mw, mc = e3_wrap_norm(4, 8, m), e3_wrap_norm(4, 8, m, clip=True)
        mean = float(np.mean(norms))
        measured["E3"][f"m{m}"] = {"true_norm": float(m), "mean_norm": mean, "sd": float(np.std(norms)), "wrap_prediction": mw, "clip_prediction": mc,
                                   "largest": float(max(norms)), "norms": norms, "max_possible_norm": cap, "warnings": warns}
        checks += [
            chk(f"E3 m {m}: mean output norm within {tol} of the wrap value {pw}", f"{pw} +- {tol}", mean, abs(mean - pw) <= tol, "stat"),
            chk(f"E3 m {m}: every norm at most the largest possible norm {cap:.3f}", f"<= {cap}", float(max(norms)), max(norms) <= cap * 1.001),
            chk(f"E3 m {m}: helper reproduces the pre-registered wrap value within the tolerance", pw, mw, abs(mw - pw) <= tol),
        ]
    return finish("E", {"E2_calls": c.n_80, "E3_calls": c.n_e3}, measured, checks)


# ---------------------------------------------------------------- row F
def row_f(c: Ctx, offset=0) -> dict:
    n, R = 20, 6
    zv = zdelta(1.0, 0.05 * n)
    c.seed(500 + offset)
    p, _, warns = c.make(1.0, n, 16, 1.0, SKELLAM, "dft", auto=True)
    chains = []
    for k in range(c.chains):
        s = p.initialize()
        rec = []
        for t in range(R):
            o = p.next(s, zero_clients(n))
            s = o.state
            x = flat(o.result)
            rec.append({"clip": meas(o.measurements, "l2_clip"), "rms": rms(x), "x": x, "scale": meas(o.measurements, "scale_factor"),
                        "local": meas(o.measurements, "scaled_local_stddev")})
        chains.append(rec)
    per_round = {}
    checks = []
    for t in range(R):
        norm = np.concatenate([ch[t]["x"] / (zv * ch[t]["clip"]) for ch in chains])
        per_round[t] = rms(norm)
        checks.append(chk(f"round {t}: pooled RMS over (value multiplier times the clip used) within 0.015 of 1", "1 +- 0.015", per_round[t], abs(per_round[t] - 1) <= 0.015, "stat"))
    after = []
    for t in range(R - 1):
        after.append(rms(np.concatenate([ch[t]["x"] / (zv * ch[t + 1]["clip"]) for ch in chains])))
    ctl = float(np.mean(after))
    r5_init = rms(np.concatenate([ch[R - 1]["x"] / (zv * 1.0) for ch in chains]))
    lnc = float(np.mean([math.log(ch[R - 1]["clip"] / 1.0) for ch in chains]))
    steps = [ch[t + 1]["clip"] / ch[t]["clip"] for ch in chains for t in range(R - 1)]
    lo = math.exp(-0.1) * (1 - 1e-5)
    ex_local = max(abs(ch[t]["local"] * math.sqrt(n) / ch[t]["scale"] / (zv * ch[t]["clip"]) - 1) for ch in chains for t in range(R))
    ex_prod = max(abs(ch[t]["scale"] * ch[t]["clip"] / (ch[0]["scale"] * ch[0]["clip"]) - 1) for ch in chains for t in range(R))
    cls = c.ddp.DistributedDpSumFactory
    src1 = " ".join(inspect.getsource(cls._build_aggregation_factory).split())
    src2 = " ".join(inspect.getsource(cls._autotune_component_states).split())
    t1 = "central_stddev = self._value_noise_mult * self._initial_l2_clip" in src1
    t2 = "new_central_stddev = new_l2_clip * self._value_noise_mult" in src2
    checks += [
        chk("positive control: pooled RMS of round t over the multiplier times the clip of round t+1 has mean at least 1.05 (predicted 1.10)", ">= 1.05", ctl, ctl >= 1.05, "stat"),
        chk("round 5 pooled RMS over (value multiplier times C0) is between 0.60 and 0.64 (predicted 0.619)", "[0.60, 0.64]", r5_init, 0.60 <= r5_init <= 0.64, "stat"),
        chk("mean ln(clip at round 5 over C0) within 0.01 of -0.48005", "-0.48005 +- 0.01", lnc, abs(lnc + 0.48005) <= 0.01, "stat"),
        chk("every step ratio of the clip is between exp(-0.1)(1-1e-5) and 1", [lo, 1.0], [min(steps), max(steps)], min(steps) >= lo and max(steps) <= 1.0, "stat"),
        chk("exact: scaled_local_stddev * sqrt(n) / scale_factor equals the value multiplier times l2_clip within 1e-5 (every round and chain)", "<= 1e-5", ex_local, ex_local <= 1e-5),
        chk("exact: scale_factor times l2_clip is constant within 1e-5 along a chain", "<= 1e-5", ex_prod, ex_prod <= 1e-5),
        chk("source text: _build_aggregation_factory builds central_stddev from the initial clip", "present", t1, t1),
        chk("source text: _autotune_component_states rebuilds the central stddev from the new clip", "present", t2, t2),
    ]
    measured = {"value_multiplier": zv, "per_round_ratio": per_round, "ratio_to_clip_after_update_per_round": after, "mean_ratio_to_clip_after_update": ctl,
                "round5_over_initial_clip": r5_init, "mean_ln_clip_round5": lnc, "step_ratio_min_max": [min(steps), max(steps)],
                "clips_by_round_mean": [float(np.mean([ch[t]["clip"] for ch in chains])) for t in range(R)],
                "max_exact_deviation_local": ex_local, "max_exact_deviation_scale_clip": ex_prod, "warnings": warns,
                "clips": [[ch[t]["clip"] for t in range(R)] for ch in chains]}
    return finish("F", {"n": n, "z": 1.0, "C0": 1.0, "rounds": R, "chains": c.chains}, measured, checks)


# ---------------------------------------------------------------- row G
def dp_state_of(state):
    return state[0]["inner_agg"]["inner_agg_process"]


def row_g(c: Ctx, offset=0) -> dict:
    checks, measured = [], {}
    forbidden = ("epsilon", "delta", "noise_multiplier")
    cfgs = [("skellam", dict(z=1.0, n=10, mech=SKELLAM)), ("dgauss", dict(z=1.0, n=10, mech=DGAUSS)), ("adaptive", dict(z=1.0, n=20, mech=SKELLAM, auto=True))]
    for name, kw in cfgs:
        p, s, _ = c.make(kw["z"], kw["n"], 16, 1.0, kw["mech"], "dft", auto=kw.get("auto", False))
        try:
            c.dpm.extract_dp_event_from_state(dp_state_of(s))
            raised = "no exception"
        except Exception as e:
            raised = type(e).__name__
        o = p.next(s, zero_clients(kw["n"]))
        try:
            ev = c.dpm.extract_dp_event_from_state(dp_state_of(o.state))
            ev_cls, ev_repr, ev_fields = type(ev).__name__, repr(ev), [a for a in vars(ev)] if hasattr(ev, "__dict__") else []
        except Exception as e:
            ev_cls, ev_repr, ev_fields = f"{type(e).__name__}", str(e)[:200], []
        state_names = [(pth, nm) for pth, nm, _ in walk_names(o.state)]
        meas_names = [(pth, nm) for pth, nm, _ in walk_names(o.measurements)]
        bad = [pth for pth, nm in state_names + meas_names if any(f in nm.lower() for f in forbidden)]
        class_leaves = sorted({str(leaf) for pth, nm, leaf in walk_names(o.state) if nm == "class_name"})
        measured[name] = {"initial_state_extraction": raised, "event_class": ev_cls, "event_repr": ev_repr, "event_fields": ev_fields,
                          "class_name_leaves_in_state": class_leaves, "names_with_forbidden_words": bad,
                          "state_names": sorted({nm for _, nm in state_names}), "measurement_names": sorted({nm for _, nm in meas_names})}
        checks += [
            chk(f"{name}: extract_dp_event_from_state on the initial state raises ExtractingDpEventFromInitialStateError", "ExtractingDpEventFromInitialStateError", raised,
                raised == "ExtractingDpEventFromInitialStateError"),
            chk(f"{name}: the event after one round is an UnsupportedDpEvent without a noise_multiplier field", "UnsupportedDpEvent", ev_cls,
                ev_cls == "UnsupportedDpEvent" and "noise_multiplier" not in ev_fields),
            chk(f"{name}: no name in the state or the measurements contains epsilon, delta or noise_multiplier", [], bad, bad == []),
        ]
        if name == "adaptive":
            ok = class_leaves == ["b'UnsupportedDpEvent'"] or class_leaves == ["UnsupportedDpEvent"]
            checks.append(chk("adaptive: the only event class name anywhere in the state is UnsupportedDpEvent", ["UnsupportedDpEvent"], class_leaves, ok))
    import tensorflow_federated as tff
    site = Path(tff.__file__).resolve().parent
    tfp_dir = Path(importlib.import_module("tensorflow_privacy").__file__).resolve().parent / "privacy" / "dp_query"
    files = [site / "python/aggregators" / f for f in ("distributed_dp.py", "rotation.py", "discretization.py", "modular_clipping.py", "secure.py", "differential_privacy.py")] + \
            [tfp_dir / f for f in ("distributed_skellam_query.py", "distributed_discrete_gaussian_query.py")]
    scan = {f.name: epsilon_names(f) for f in files}
    cls = c.ddp.DistributedDpSumFactory
    todo = "exact privacy accounting" in " ".join((inspect.getdoc(cls) or "").split())
    measured["epsilon_names_eight_files"] = scan
    measured["docstring_todo_present"] = todo
    checks += [chk("no name containing epsilon in the eight files outside docstrings and comments", 0, sum(len(v) for v in scan.values()), sum(len(v) for v in scan.values()) == 0),
               chk("the class docstring contains a TODO about exact privacy accounting", "present", todo, todo)]
    return finish("G", {"mechanisms": ["skellam", "dgauss", "adaptive"]}, measured, checks)


# ---------------------------------------------------------------- row H
def child_main(seed: int, mech: str) -> int:
    c = Ctx(True, probe_default=False)
    c.tf.random.set_seed(seed)
    np.random.seed(seed)
    p, s, _ = c.make(1.0, 10, 16, 1.0, mech, "dft")
    x = np.asarray(p.next(s, zero_clients(10)).result, np.float32)
    print(json.dumps({"digest": hashlib.sha256(x.tobytes()).hexdigest()}), flush=True)
    kill_children()
    os._exit(0)


def row_h(c: Ctx, offset=0) -> dict:
    checks, measured = [], {}
    thr = 4.0 / math.sqrt(c.pairs * 2 * D)
    for mech in (SKELLAM, DGAUSS):
        name = mech.split("_")[1]
        c.seed(600 + (mech == DGAUSS) + offset)
        p, s, _ = c.make(1.0, 10, 16, 1.0, mech, "dft")
        v = zero_clients(10)
        xs1, xs2 = [], []
        for _ in range(c.pairs):
            xs1.append(flat(p.next(s, v).result))
            xs2.append(flat(p.next(s, v).result))
        r12 = corr(np.concatenate(xs1), np.concatenate(xs2))
        allv = xs1 + xs2
        lag = corr(np.concatenate([x[:-1] for x in allv]), np.concatenate([x[1:] for x in allv]))
        ys = [x[1:] + x[:-1] for x in allv]
        ctl = corr(np.concatenate([y[:-1] for y in ys]), np.concatenate([y[1:] for y in ys]))
        equal = any(np.array_equal(a, b) for a, b in zip(xs1, xs2))
        digests = []
        for _ in range(2):
            r = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--child-seed", "12345", "--child-mech", mech],
                               capture_output=True, text=True, timeout=900)
            try:
                digests.append(json.loads(r.stdout.strip().splitlines()[-1])["digest"])
            except (ValueError, IndexError):
                digests.append(f"error: {(r.stderr or r.stdout)[-200:]}")
        measured[name] = {"pairs": c.pairs, "correlation_between_calls": r12, "lag1_autocorrelation": lag, "positive_control_lag1": ctl, "threshold": thr,
                          "any_pair_equal": bool(equal), "seed_digests": digests, "mde80_correlation": (4.0 + Z80) / math.sqrt(c.pairs * 2 * D / 2 * 2)}
        checks += [
            chk(f"{name}: the two calls of every pair differ", "not equal", bool(equal), not equal),
            chk(f"{name}: absolute pooled correlation between the two calls at most {thr:.4f}", f"<= {thr}", r12, abs(r12) <= thr, "stat"),
            chk(f"{name}: absolute pooled lag-1 autocorrelation at most {thr:.4f}", f"<= {thr}", lag, abs(lag) <= thr, "stat"),
            chk(f"{name}: positive control, lag-1 autocorrelation of neighbouring sums above 0.4", "> 0.4", ctl, ctl > 0.4, "stat"),
            chk(f"{name}: two fresh processes with the same global seed give different digests (prediction: the seed does not control the noise)", "different",
                digests, len(set(digests)) == 2 and not any(str(x).startswith("error") for x in digests)),
        ]
    return finish("H", {"pairs": c.pairs, "D": D}, measured, checks)


# ---------------------------------------------------------------- row J
def row_j(c: Ctx, offset=0) -> dict:
    n = 20
    zv = zdelta(1.0, 0.05 * n)
    checks, measured = [], {}
    c.seed(800 + offset)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fac = c.tff.learning.ddp_secure_aggregator(1.0, n)
        p = fac.create(c.vt)
        s = p.initialize()
    warns = [str(x.message)[:100] for x in w]
    measured["construction"] = {"warnings": warns, "constructed": True}
    try:
        p.next(s, zero_clients(n))
    except Exception as e:
        # A configuration that cannot run is a finding. Record the error in both execution contexts and stop the row.
        err = f"{type(e).__name__}: {str(e)}"[:300].replace(chr(10), " ")
        measured["test_context_error"] = err
        try:
            c.tff.backends.native.set_sync_local_cpp_execution_context()
            p.next(s, zero_clients(n))
            measured["default_context_error"] = None
        except Exception as e2:
            measured["default_context_error"] = f"{type(e2).__name__}: {str(e2)}"[:300].replace(chr(10), " ")
        finally:
            c.tff.backends.test.set_sync_test_cpp_execution_context()
        checks.append(chk("a call of the wrapper runs under the test C++ execution context", "runs", err, False))
        return finish("J", {"n": n, "m": [20, 10], "calls": c.n_j}, measured, checks)
    for m in (20, 10):
        outs, m0 = noise_calls(p, s, zero_clients(m), c.n_j)
        x = np.concatenate(outs)
        pred = zv * 0.1 * math.sqrt(m / n) / m
        sf = meas(m0, "scale_factor")
        measured[f"m{m}"] = {"N": int(x.size), "predicted_rms": pred, "rms": rms(x), "ratio": rms(x) / pred, "scale_factor": sf, "warnings": warns}
        checks.append(chk(f"m {m}: RMS of the mean over the prediction {pred:.7f} within 0.01 of 1", "1 +- 0.01", rms(x) / pred, abs(rms(x) / pred - 1) <= 0.01, "stat"))
        checks.append(chk(f"m {m}: scale_factor is 1,000,000 (the cap)", 1e6, sf, sf is not None and rel(sf, 1e6) <= 1e-6))
    return finish("J", {"n": n, "m": [20, 10], "calls": c.n_j}, measured, checks)


# ---------------------------------------------------------------- row I and V
def row_i(rows: dict) -> dict:
    A, Cc, Dd = rows["A"]["measured"], rows["C"]["measured"], rows["D"]["measured"]
    sens = lambda b: Dd[f"b {b} beta 0"]["mean_norm_over_C"]  # noqa: E731
    bound = lambda b: Dd[f"b {b} beta 0"]["bound_over_C"]  # noqa: E731
    entries = [
        ("A1 skellam dft b 16", A["A1"]["A1 skellam dft"]["rms_over_zC"], sens(16)),
        ("C skellam m 5", Cc["skellam m 5"]["rms_over_zC"], sens(16)),
        ("C skellam m 20", Cc["skellam m 20"]["rms_over_zC"], sens(16)),
        ("A2 dgauss b 8, mean sensitivity", A["A2"]["A2 dgauss bits 8"]["rms_over_zC"], sens(8)),
        ("A2 skellam b 5, mean sensitivity", A["A2"]["A2 skellam bits 5"]["rms_over_zC"], sens(5)),
        ("A2 skellam b 5, sensitivity at the bound", A["A2"]["A2 skellam bits 5"]["rms_over_zC"], bound(5)),
    ]
    table = []
    for nm, noise, sn in entries:
        mu = sn / noise
        table.append({"entry": nm, "noise_rms_over_C": noise, "sensitivity_over_C": sn, "delivered_multiplier": noise / sn, "mu": mu,
                      "epsilon_at_delta_1e-5": geps(mu, 1e-5),
                      "label": "Gaussian approximation conditional on the measured numbers, add or remove adjacency, not a guarantee"})
    srt = sorted(table, key=lambda t: t["delivered_multiplier"])
    mono = all(srt[i]["epsilon_at_delta_1e-5"] >= srt[i + 1]["epsilon_at_delta_1e-5"] - 1e-12 for i in range(len(srt) - 1))
    d1, e1, e2 = gdelta(1.0, 1.0), geps(1.0, 1e-5), geps(2.0, 1e-5)
    checks = [chk("helper: delta(mu 1, eps 1) = 0.126937 within 1e-6", 0.126937, d1, abs(d1 - 0.126937) <= 1e-6),
              chk("helper: eps(mu 1, delta 1e-5) = 4.37718 within 1e-4", 4.37718, e1, abs(e1 - 4.37718) <= 1e-4),
              chk("helper: eps(mu 2, delta 1e-5) = 9.99726 within 1e-3", 9.99726, e2, abs(e2 - 9.99726) <= 1e-3),
              chk("the epsilon in the table decreases when the delivered multiplier rises", "monotone", [t["epsilon_at_delta_1e-5"] for t in srt], mono)]
    return finish("I", {"delta": 1e-5, "note": "derived from rows A, C and D, no new calls"}, {"table": table}, checks)


def row_v(c: Ctx) -> dict:
    got = {k: importlib.metadata.version(k) for k in EXPECTED_VERSIONS}
    import tensorflow_federated as tff
    no_export = not hasattr(tff.aggregators, "DistributedDpSumFactory")
    has_wrapper = hasattr(tff.learning, "ddp_secure_aggregator")
    return finish("V", {"expected": EXPECTED_VERSIONS}, {"installed": got, "python": platform.python_version(), "platform": platform.platform(),
                                                          "tff.aggregators exports DistributedDpSumFactory": not no_export, "tff.learning has ddp_secure_aggregator": has_wrapper,
                                                          "default_local_cpp_context_probe": c.default_context, "context_used": "tff.backends.test.set_sync_test_cpp_execution_context"},
                  [chk("one call under the test C++ execution context runs (all rows use it)", "runs", "see rows", True),
                   chk("installed versions equal the pre-registered ones", EXPECTED_VERSIONS, got, got == EXPECTED_VERSIONS),
                   chk("tff.aggregators has no DistributedDpSumFactory and tff.learning has ddp_secure_aggregator (known from the probe, not blind)", [True, True],
                       [no_export, has_wrapper], no_export and has_wrapper)])


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
    t = time.time()
    try:
        r = fn(*args, **kw)
    except Exception as e:  # a failed row stays in the record
        r = {"id": fn.__name__.replace("row_", "").upper(), "measured": {"exception": f"{type(e).__name__}: {e}"[:400]},
             "checks": [chk("row ran", "no exception", f"{type(e).__name__}: {e}"[:300], False)], "prediction_held": False}
    r["seconds"] = round(time.time() - t, 1)
    return r


def main() -> int:
    if "--child-seed" in sys.argv:
        return child_main(int(sys.argv[sys.argv.index("--child-seed") + 1]), sys.argv[sys.argv.index("--child-mech") + 1])
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
    doc = {"schema": "feddisclosure.tff_ddp_audit.v1", "quick": args.quick, "preregistration": PREREG_FILE, "preregistration_commit": PREREG_COMMIT,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "environment": {k: importlib.metadata.version(k) for k in EXPECTED_VERSIONS} | {"python": platform.python_version(), "platform": platform.platform()},
           "parameters": {"base_seed": BASE_SEED, "D": D, "calls": {"A1": c.n_a1, "A2_A3_B_C_E2": c.n_80, "D": c.n_d, "E3": c.n_e3, "F_chains": c.chains,
                                                                    "H_pairs": c.pairs, "J": c.n_j}},
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

    add(run_row(row_v, c))
    rows = {}
    for fn in (row_a, row_b, row_c, row_d, row_e, row_f, row_g, row_h):
        r = run_row(fn, c)
        rows[r["id"]] = r
        add(r, fn, c)
    if all(k in rows and "exception" not in rows[k]["measured"] for k in ("A", "C", "D")):
        add(run_row(row_i, rows))
    else:
        add({"id": "I", "measured": {"exception": "inputs missing"}, "checks": [chk("row ran", "inputs present", "missing", False)], "prediction_held": False})
    add(run_row(row_j, c), row_j, c)
    total = time.time() - t0
    add({"id": "T", "measured": {"total_seconds": round(total, 1)}, "checks": [chk("whole script under 2400 seconds", "< 2400", round(total, 1), total < 2400 or args.quick)],
         "prediction_held": total < 2400 or args.quick})
    s = doc["summary"]
    print(f"wrote {out_dir / OUT_NAME} in {doc['runtime_seconds']}s, {s['rows_held']} of {s['rows_total']} rows held, "
          f"{s['checks_held']} of {s['checks_total']} checks held, mismatch rows {s['mismatch_found_rows']}", flush=True)
    kill_children()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
