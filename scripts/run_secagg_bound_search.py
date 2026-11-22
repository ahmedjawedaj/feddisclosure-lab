"""v0.15 WP6: search for counterexamples to the one-round SecAgg+ quantisation error bound.

Everything here runs the REAL upstream flwr quantize and dequantize through
feddisclosure.attacks.view_study.secagg_aggregate. Statements and proofs are in
docs/proofs/one-round-secagg-error.md and docs/proofs/multi-round-note.md.
Predictions are in docs/preregistration/v0.15.md (E1, E2, E3, M1). Thresholds are copied from
that file and are not changed here.

Scope. The integer sum of the quantised client vectors is taken as recovered exactly modulo 2**32.
Only the quantisation arithmetic is analysed.

Usage:
    PYTHONPATH=src:. python scripts/run_secagg_bound_search.py
Writes reports/results/privacy/v0150/secagg_bound_search.json and prereg_check_proofs.json.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import flwr  # noqa: E402
import flwr.common.secure_aggregation.quantization as qz  # noqa: E402
from flwr.common.secure_aggregation.ndarrays_arithmetic import parameters_mod  # noqa: E402

from feddisclosure.attacks.view_study import CLIP_RANGE, QUANT_RANGE, secagg_aggregate  # noqa: E402

C = CLIP_RANGE            # 8.0
Q = QUANT_RANGE           # 2**22
MOD = 2 ** 32             # upstream default modulus_range
MAX_N = (MOD - 1) // Q    # 1023, largest n with n * Q < 2**32
OUT_DIR = ROOT / "reports/results/privacy/v0150"
LD = np.longdouble


# ---------------------------------------------------------------------------------------
# Quantities from the proof note
# ---------------------------------------------------------------------------------------
def quantised_weights(sizes, max_weight):
    """Same expression as upstream secaggplus_mod: round((n / max_weight) * target_range)."""
    return [round(m / max_weight * Q) for m in sizes]


def exact_means(vectors, sizes, max_weight):
    """Return (mu_q, mu_true, T). mu_q uses the integer weights q_i, mu_true uses the sizes."""
    qs = quantised_weights(sizes, max_weight)
    total_q = sum(qs)
    stack = np.stack([np.asarray(v, dtype=LD) for v in vectors])
    mu_q = (np.asarray(qs, dtype=LD)[:, None] * stack).sum(0) / LD(total_q)
    mu_true = (np.asarray(sizes, dtype=LD)[:, None] * stack).sum(0) / LD(sum(sizes))
    return mu_q, mu_true, total_q


def no_clipping(vectors, sizes, max_weight):
    """True when no weighted coordinate leaves [-C, C], computed in each vector's own dtype."""
    for v, q in zip(vectors, quantised_weights(sizes, max_weight)):
        if float(np.max(np.abs(v * (q / Q)))) > C:
            return False
    return True


def worst_case_bound(n, total_q):
    """Theorem part (a): strict bound 2 c n / T on one coordinate (exact arithmetic)."""
    return 2.0 * C * n / total_q


def server_float64_term(n, total_q):
    """Lemma 4: rounding of the float64 server stage, at most 2**-47 * n * Q / T."""
    return n * Q / total_q * 2.0 ** -47


def stated_bound(n, total_q):
    """The bound tested for E1 against the aggregate of the integer-weighted mean."""
    return worst_case_bound(n, total_q) + server_float64_term(n, total_q)


def weight_term(vectors, total_q):
    """Lemma 5: |mu_q - mu_true| <= n * R_j * (1 + 2**-30) / (2 T) with R_j the range over clients."""
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    rng_j = stack.max(0) - stack.min(0)
    return len(vectors) * rng_j * (1 + 2.0 ** -30) / (2.0 * total_q)


def hoeffding_term(n, total_q, d, delta):
    """Theorem part (b): all d coordinates within this with probability at least 1 - delta."""
    return 2.0 * C / total_q * math.sqrt(n / 2.0 * math.log(2.0 * d / delta))


# ---------------------------------------------------------------------------------------
# Spy on the real upstream path
# ---------------------------------------------------------------------------------------
def pre_quantised(v, dq):
    """The array that upstream _stochastic_round receives for weighted input v * dq."""
    captured = []
    original = qz._stochastic_round

    def spy(arr):
        captured.append(arr.copy())
        return original(arr)

    with mock.patch.object(qz, "_stochastic_round", spy):
        qz.quantize([v * dq], C, Q)
    return captured[0]


def exact_pre_quantised(v, dq):
    """x = (w v + c) Q / (2 c) in extended precision from the exact product of the stored v."""
    wv = np.asarray(v, dtype=LD) * LD(dq)
    return (wv + LD(C)) * LD(Q) / LD(2 * C)


# ---------------------------------------------------------------------------------------
# E1: random counterexample search through secagg_aggregate
# ---------------------------------------------------------------------------------------
FAMILIES = ("uniform", "boundary", "high_binade", "gauss_small", "identical")


def limit_for(dq, dtype):
    """Largest magnitude L in dtype with |fl(L * dq)| <= C."""
    lim = dtype(C / dq)
    while float(np.abs(np.asarray([lim], dtype=dtype) * dq)[0]) > C:
        lim = np.nextafter(lim, dtype(0))
    return lim


def make_case(rng, family, dtype=np.float32):
    n = int(rng.integers(2, 65)) if rng.random() < 0.8 else int(rng.choice([2, 3, 63, 64]))
    d = int(rng.integers(1, 49))
    max_weight = float(rng.choice([16.0, 200.0, 1000.0, 4096.0, 5000.0]))
    kind = rng.random()
    if kind < 0.2:
        sizes = [int(rng.integers(1, int(max_weight) + 1))] * n
    elif kind < 0.35:
        sizes = [int(max_weight) if rng.random() < 0.4 else int(rng.integers(1, int(max_weight) + 1)) for _ in range(n)]
    else:
        sizes = [int(rng.integers(1, int(max_weight) + 1)) for _ in range(n)]
    dqs = [q / Q for q in quantised_weights(sizes, max_weight)]
    lims = [limit_for(dq, dtype) for dq in dqs]
    vectors = []
    base = None
    for i in range(n):
        lim = float(lims[i]) if family != "identical" else float(min(lims))
        if family == "uniform":
            u = rng.uniform(-1, 1, d)
        elif family == "high_binade":
            u = rng.choice([-1.0, 1.0], d) * rng.uniform(0.5, 1.0, d)
        elif family == "gauss_small":
            u = np.clip(rng.normal(0, 0.3, d), -1, 1) * (0.3 / 0.3) / max(lim, 1e-12) * min(lim, 1.0)
        elif family == "boundary":
            pick = rng.integers(0, 6, d)
            table = np.array([1.0, -1.0, 0.0, 2.0 ** -24, -2.0 ** -24, 0.5])
            u = table[pick]
        else:   # identical
            if base is None:
                base = rng.uniform(-1, 1, d) if rng.random() < 0.5 else rng.choice([-1.0, 1.0], d) * rng.uniform(0.9, 1.0, d)
            u = base
        v = (np.asarray(u) * lim).astype(dtype)
        if family == "boundary":                       # step to the neighbours of the limit
            move = rng.random(d) < 0.5
            v = np.where(move, np.nextafter(v, dtype(0)), v).astype(dtype)
        # enforce no clipping exactly as upstream computes it
        for _ in range(8):
            bad = np.abs(v * dqs[i]) > C
            if not bad.any():
                break
            v = np.where(bad, np.nextafter(v, dtype(0)), v).astype(dtype)
        vectors.append(v)
    if family == "identical":
        sizes = [sizes[0]] * n
    return vectors, sizes, max_weight


def run_e1(cases, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    per_family = {f: 0 for f in FAMILIES}
    violations, viol_true, exceed_float64_only = [], [], 0
    max_ratio_q = max_ratio_true = 0.0
    worst_over_f64 = 0.0
    n_hist = {}
    for case in range(cases):
        family = FAMILIES[case % len(FAMILIES)]
        vectors, sizes, mw = make_case(rng, family, dtype)
        assert no_clipping(vectors, sizes, mw), "generator produced a clipped input"
        n = len(vectors)
        n_hist[n] = n_hist.get(n, 0) + 1
        agg = secagg_aggregate(vectors, sizes, max_weight=mw, rng_seed=int(rng.integers(2 ** 31)))
        mu_q, mu_true, total_q = exact_means(vectors, sizes, mw)
        err_q = np.abs(agg.astype(LD) - mu_q).astype(np.float64)
        err_t = np.abs(agg.astype(LD) - mu_true).astype(np.float64)
        b_q = stated_bound(n, total_q)
        b_t = b_q + weight_term(vectors, total_q)
        per_family[family] += 1
        max_ratio_q = max(max_ratio_q, float(err_q.max() / b_q))
        max_ratio_true = max(max_ratio_true, float((err_t / b_t).max()))
        if err_q.max() > worst_case_bound(n, total_q):
            exceed_float64_only += 1
        worst_over_f64 = max(worst_over_f64, float(err_q.max() / worst_case_bound(n, total_q)))
        if err_q.max() > b_q:
            violations.append({"case": case, "family": family, "n": n, "err": float(err_q.max()), "bound": b_q})
        if np.any(err_t > b_t):
            viol_true.append({"case": case, "family": family, "n": n, "err": float(err_t.max()), "bound": float(b_t.max())})
    return {
        "dtype": np.dtype(dtype).name, "cases": cases, "seed": seed, "per_family": per_family,
        "n_min": min(n_hist), "n_max": max(n_hist),
        "violations_integer_weight_mean": len(violations), "violations_true_mean": len(viol_true),
        "cases_above_float64_only_bound_2cn_over_T": exceed_float64_only,
        "max_error_over_stated_bound": max_ratio_q, "max_error_over_2cn_over_T": worst_over_f64,
        "max_true_mean_error_over_bound_with_weight_term": max_ratio_true,
        "first_violations": violations[:5] + viol_true[:5],
    }


# ---------------------------------------------------------------------------------------
# E2: float32 client path, per element bias and the cell lemma
# ---------------------------------------------------------------------------------------
def float32_path_study(batches, batch_size, seed):
    rng = np.random.default_rng(seed)
    classes = {"abs_wv_lt_1": [0, 0.0, 0.0, 0.0, 0.0], "abs_wv_1_to_4": [0, 0.0, 0.0, 0.0, 0.0],
               "abs_wv_4_to_8": [0, 0.0, 0.0, 0.0, 0.0]}     # count, sum, sum abs, sum sq, max abs
    cell_violations = grid_violations = integer_over = 0
    total = 0
    worst_single_client = 0.0          # max over both roundings of |k - x_exact|, in units
    worst_single_client_f64 = 0.0
    for b in range(batches):
        m = int(rng.integers(1, 1001)) if b % 3 else 1000
        dq = round(m / 1000.0 * Q) / Q
        mode = b % 4
        if mode == 0:
            v = rng.uniform(-8, 8, batch_size)
        elif mode == 1:
            v = rng.uniform(-1, 1, batch_size)
        elif mode == 2:
            v = rng.choice([-1.0, 1.0], batch_size) * 2.0 ** rng.uniform(-12, 3, batch_size)
        else:
            v = rng.choice([-1.0, 1.0], batch_size) * rng.uniform(3.5, 8.0, batch_size)
        v = (v / dq).astype(np.float32)
        lim = limit_for(dq, np.float32)
        v = np.clip(v, -lim, lim)
        for _ in range(8):
            bad = np.abs(v * dq) > C
            if not bad.any():
                break
            v = np.where(bad, np.nextafter(v, np.float32(0)), v).astype(np.float32)
        x32 = pre_quantised(v, dq).astype(LD)
        xe = exact_pre_quantised(v, dq)
        bias = (x32 - xe).astype(np.float64)
        floor, ceil = np.floor(x32), np.ceil(x32)
        nonint = x32 != floor
        cell_violations += int((~((floor < xe) & (xe < ceil)))[nonint].sum())
        integer_over += int((np.abs(bias[~nonint]) > 3.0 / 16.0 + 1e-9).sum())
        grid_violations += int((np.mod(x32 * 8, 1) != 0).sum())
        worst_single_client = max(worst_single_client,
                                  float(np.maximum(np.abs(floor - xe), np.abs(ceil - xe)).astype(np.float64).max()))
        mag = np.abs(v.astype(np.float64) * dq)
        for name, mask in (("abs_wv_lt_1", mag < 1), ("abs_wv_1_to_4", (mag >= 1) & (mag < 4)), ("abs_wv_4_to_8", mag >= 4)):
            if mask.any():
                bb = bias[mask]
                acc = classes[name]
                acc[0] += bb.size; acc[1] += float(bb.sum()); acc[2] += float(np.abs(bb).sum())
                acc[3] += float((bb ** 2).sum()); acc[4] = max(acc[4], float(np.abs(bb).max()))
        total += v.size
    table = {}
    all_n = sum(a[0] for a in classes.values())
    for name, (cnt, s, sa, s2, mx) in list(classes.items()) + [("all", [all_n, sum(a[1] for a in classes.values()),
                                                                    sum(a[2] for a in classes.values()),
                                                                    sum(a[3] for a in classes.values()),
                                                                    max(a[4] for a in classes.values())])]:
        if cnt:
            table[name] = {"count": cnt, "mean_bias_units": s / cnt, "mean_abs_bias_units": sa / cnt,
                           "rms_bias_units": math.sqrt(s2 / cnt), "max_abs_bias_units": mx}
    return {"elements": total, "bias_table_units": table, "theoretical_sup_abs_bias_units": 3.0 / 16.0,
            "cell_lemma_violations": cell_violations, "x32_not_multiple_of_one_eighth": grid_violations,
            "integer_cells_with_bias_above_3_16": integer_over,
            "worst_single_client_error_units_either_rounding": worst_single_client,
            "elements_with_single_client_error_at_least_1": 0 if worst_single_client < 1 else None}


def adversarial_rounding_tightness():
    """Force the rounding draws to the worst case and report error over 2 c n / T (float64 inputs)."""
    out = {}
    n, d = 8, 16
    sizes = [500] * n
    qs = quantised_weights(sizes, 1000.0)
    dq = qs[0] / Q
    total_q = sum(qs)
    cases = {"down": (1.0 - 2.0 ** -20, 0.0), "up": (2.0 ** -20, float(np.nextafter(1.0, 0.0)))}
    for name, (frac, draw) in cases.items():
        x_target = 1000000.0 + frac
        wv = x_target * 2 * C / Q - C
        v = np.full(d, wv / dq)
        with mock.patch("numpy.random.rand", lambda *s, draw=draw: np.full(s, draw)):
            agg = secagg_aggregate([v] * n, sizes, max_weight=1000.0, rng_seed=0)
        mu_q, _, _ = exact_means([v] * n, sizes, 1000.0)
        out[f"float64_always_{name}"] = float(np.abs(agg.astype(LD) - mu_q).max() / worst_case_bound(n, total_q))
    return out


def run_e2(batches, batch_size, seed):
    study = float32_path_study(batches, batch_size, seed)
    # Deterministic example: the float32 path moves the stored value by -1/8 unit and lands on an integer.
    dq, n = 1.0, 256
    v = np.float32(0.999 * 2.0 ** -21)
    b = float((pre_quantised(np.array([v]), dq).astype(LD) - exact_pre_quantised(np.array([v]), dq))[0])
    sizes = [1000] * n
    total_q = sum(quantised_weights(sizes, 1000.0))
    agg = secagg_aggregate([np.full(4, v, dtype=np.float32)] * n, sizes, max_weight=1000.0, rng_seed=1)
    mu_q, _, _ = exact_means([np.full(4, v, dtype=np.float32)] * n, sizes, 1000.0)
    err_units = float(np.abs(agg.astype(LD) - mu_q).max() * total_q / (2 * C))
    study["identical_clients_example"] = {
        "n": n, "per_element_bias_units": b, "aggregate_error_units": err_units,
        "hoeffding_term_units_d1_delta_0.01": math.sqrt(n / 2.0 * math.log(2.0 / 0.01)),
        "worst_case_units": float(n),
        "reading": "the pure Hoeffding term fails here by a deterministic float32 effect with no randomness left, "
                   "the worst case bound 2cn/T still holds"}
    study["adversarial_rounding_tightness"] = adversarial_rounding_tightness()
    return study


# ---------------------------------------------------------------------------------------
# E3: Hoeffding bound with independent rounding, and a shared stream
# ---------------------------------------------------------------------------------------
def aggregate_with_client_streams(vectors, sizes, max_weight, seeds):
    """Mirror of secagg_aggregate that reseeds numpy before each client.

    seeds=None seeds once (this reproduces secagg_aggregate bit for bit when seed0 is given).
    A list gives one seed per client. Equal entries make every client draw the same uniforms.
    Uses the upstream quantize and dequantize.
    """
    total_q, summed = 0, None
    for i, (v, m) in enumerate(zip(vectors, sizes)):
        q_ratio = round(m / max_weight * Q)
        np.random.seed(seeds[i])
        q = qz.quantize([v * (q_ratio / Q)], C, Q)[0].astype(np.int64)
        summed = q if summed is None else summed + q
        total_q += q_ratio
    agg = qz.dequantize([summed], C, Q)[0]
    return (agg - (len(vectors) - 1) * C) * (Q / total_q)


def run_e3(ns, trials, d, seed, delta=0.01):
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        sizes = [450] * n
        total_q = sum(quantised_weights(sizes, 1000.0))
        h = hoeffding_term(n, total_q, d, delta)
        slack = server_float64_term(n, total_q) + 2 * C / total_q * n * 2.0 ** -31      # float64 stage and float64 client path
        bound = h + slack
        bound_single = hoeffding_term(n, total_q, 1, delta) + slack
        row = {"n": n, "d": d, "trials": trials, "total_q": total_q, "delta": delta,
               "hoeffding_bound_all_coordinates": bound, "hoeffding_bound_single_coordinate": bound_single,
               "worst_case_bound": worst_case_bound(n, total_q), "bound_is_tighter_than_worst_case": bound < worst_case_bound(n, total_q)}
        unit = total_q / (2 * C)           # error in rounding units = error * unit
        for mode in ("independent", "shared_seed_identical_data", "shared_seed_different_data"):
            maxima, pooled = [], []
            for t in range(trials):
                if mode == "shared_seed_identical_data":
                    base = rng.uniform(-1, 1, d)
                    vecs = [base] * n
                else:
                    vecs = [rng.uniform(-1, 1, d) for _ in range(n)]
                mu_q, _, _ = exact_means(vecs, sizes, 1000.0)
                s0 = int(rng.integers(2 ** 31))
                if mode == "independent":
                    agg = secagg_aggregate(vecs, sizes, max_weight=1000.0, rng_seed=s0)
                else:
                    agg = aggregate_with_client_streams(vecs, sizes, 1000.0, [s0] * n)
                err = np.abs(agg.astype(LD) - mu_q).astype(np.float64)
                maxima.append(err.max()); pooled.extend(err.tolist())
            maxima = np.asarray(maxima); pooled = np.asarray(pooled)
            p99_max, p99_pooled = float(np.quantile(maxima, 0.99)), float(np.quantile(pooled, 0.99))
            row[mode] = {"p99_max_over_coordinates": p99_max, "p99_pooled_coordinates": p99_pooled,
                         "p99_max_over_bound": p99_max / bound, "p99_pooled_over_single_bound": p99_pooled / bound_single,
                         "fraction_trials_above_bound": float((maxima > bound).mean()),
                         "p99_max_error_units": p99_max * unit,
                         "p99_max_error_units_over_sqrt_n": p99_max * unit / math.sqrt(n),
                         "p99_max_error_units_over_n": p99_max * unit / n,
                         "below_bound": bool(p99_max < bound), "below_single_bound": bool(p99_pooled < bound_single)}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------
# Upstream limit on n
# ---------------------------------------------------------------------------------------
def overflow_limit_check():
    def total(n):
        k = qz.quantize([np.array([C])], C, Q)[0].astype(np.int64)
        return int(parameters_mod([k * n], MOD)[0][0]), int(k[0]) * n
    wrapped_1023, raw_1023 = total(1023)
    wrapped_1024, raw_1024 = total(1024)
    return {"quantised_value_at_plus_c": int(qz.quantize([np.array([C])], C, Q)[0][0]),
            "largest_n_without_possible_wrap": MAX_N,
            "n_1023_all_at_plus_c": {"raw_sum": raw_1023, "after_mod": wrapped_1023, "wrapped": wrapped_1023 != raw_1023},
            "n_1024_all_at_plus_c": {"raw_sum": raw_1024, "after_mod": wrapped_1024, "wrapped": wrapped_1024 != raw_1024}}


# ---------------------------------------------------------------------------------------
# M1: convex trials and the two-round ReLU counterexample
# ---------------------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _make_clients(rng, kind, n, d, lam, shared=False):
    clients = []
    for _ in range(n):
        if shared and clients:
            clients.append(clients[0])
            continue
        m = int(rng.integers(8, 65))
        X = rng.normal(0, 0.6, (m, d))
        if kind == "ls":
            y = X @ rng.normal(0, 0.5, d) + rng.normal(0, 0.1, m)
            beta = float(np.linalg.eigvalsh(X.T @ X / m)[-1])
        else:
            y = np.where(rng.random(m) < 0.5, -1.0, 1.0)
            beta = float(np.linalg.eigvalsh(X.T @ X / m)[-1]) / 4.0 + lam
        clients.append((X, y, m, beta))
    return clients


def _grad(kind, X, y, m, theta, lam):
    if kind == "ls":
        return X.T @ (X @ theta - y) / m
    return -(X.T @ (y * _sigmoid(-y * (X @ theta)))) / m + lam * theta


def m1_trial(rng, kappa=None, kind=None, rounds=None, local_steps=None, shared=False):
    """One random convex FedAvg trial. Returns dict with the checks. Step is kappa * 2 / beta_max."""
    kind = kind or ("ls" if rng.random() < 0.5 else "logreg")
    n = int(rng.integers(2, 9)); d = int(rng.integers(2, 25))
    lam = float(rng.choice([0.0, 0.01, 0.1])) if kind == "logreg" else 0.0
    rounds = rounds or int(rng.integers(2, 7)); steps = local_steps or int(rng.integers(1, 4))
    clients = _make_clients(rng, kind, n, d, lam, shared)
    sizes = [c[2] for c in clients]
    max_weight = float(rng.choice([64.0, 100.0, 1000.0]))
    beta_max = max(c[3] for c in clients)
    if kappa is None:
        kappa = 1.0 if rng.random() < 0.4 else float(rng.uniform(0.05, 1.0))
    eta = kappa * 2.0 / beta_max
    p = np.asarray(sizes, dtype=np.float64) / sum(sizes)

    def local(theta, c):
        X, y, m, _ = c
        t = theta.copy()
        for _ in range(steps):
            t = t - eta * _grad(kind, X, y, m, t, lam)
        return t

    theta0 = rng.normal(0, 0.5, d)
    ex, sec = theta0.copy(), theta0.copy()
    e_norms, b_coord_max, b_l2, one_round_viol = [], 0.0, [], 0
    for r in range(rounds):
        ex = sum(pi * local(ex, c) for pi, c in zip(p, clients))
        states = [local(sec, c) for c in clients]
        if not no_clipping(states, sizes, max_weight):
            return None
        agg = secagg_aggregate(states, sizes, max_weight=max_weight, rng_seed=int(rng.integers(2 ** 31)))
        g_true = sum(pi * s for pi, s in zip(p, states))
        _, _, total_q = exact_means(states, sizes, max_weight)
        b = stated_bound(n, total_q) + weight_term(states, total_q)
        e = agg - g_true
        one_round_viol += int(np.any(np.abs(e) > b))
        e_norms.append(float(np.linalg.norm(e)))
        b_coord_max = max(b_coord_max, float(np.max(b)))
        b_l2.append(float(np.linalg.norm(b)))
        sec = agg
    delta = float(np.linalg.norm(sec - ex))
    return {"kind": kind, "kappa": kappa, "n": n, "d": d, "rounds": rounds, "local_steps": steps,
            "delta_l2": delta, "sum_realised_error_l2": sum(e_norms),
            "stated_l2_bound": rounds * math.sqrt(d) * b_coord_max, "sum_per_round_l2_bounds": sum(b_l2),
            "one_round_violations": one_round_viol,
            "telescoping_violated": delta > sum(e_norms) * (1 + 1e-9) + 1e-12,
            "stated_bound_violated": delta > rounds * math.sqrt(d) * b_coord_max,
            "step_exactly_two_over_beta": kappa == 1.0}


def run_m1_convex(trials, seed):
    rng = np.random.default_rng(seed)
    done, skipped = [], 0
    while len(done) < trials:
        res = m1_trial(rng)
        if res is None:
            skipped += 1
            continue
        done.append(res)
    ratios = [r["delta_l2"] / r["stated_l2_bound"] for r in done]
    return {"trials": len(done), "seed": seed, "skipped_for_clipping": skipped,
            "trials_with_step_exactly_two_over_beta": sum(r["step_exactly_two_over_beta"] for r in done),
            "kinds": {k: sum(r["kind"] == k for r in done) for k in ("ls", "logreg")},
            "l2_bound_violations": sum(r["stated_bound_violated"] for r in done),
            "telescoping_violations": sum(r["telescoping_violated"] for r in done),
            "one_round_bound_violations_inside_trajectories": sum(r["one_round_violations"] for r in done),
            "max_delta_over_stated_bound": max(ratios), "median_delta_over_stated_bound": float(np.median(ratios))}


def run_m1_negative_control(trials, seed, kappa=1.3):
    """Step above 2/beta. The bound is expected to fail, which shows the check can fail."""
    rng = np.random.default_rng(seed)
    out, attempts = [], 0
    while len(out) < trials and attempts < 20 * trials:
        attempts += 1
        r = m1_trial(rng, kappa=kappa, kind="ls", rounds=6, local_steps=2, shared=True)
        if r is not None:
            out.append(r)
    return {"trials": len(out), "kappa": kappa, "l2_bound_violations": sum(r["stated_bound_violated"] for r in out)}


def relu_counterexample(seeds):
    """Two clients (sizes 300 and 600), one sample each, model u * relu(w x), local step size 1.

    Exact round 1 gives w = 0 exactly, which is the kink of the ReLU. The SecAgg+ round 1 error
    moves w to one side of the kink, a different client becomes active in round 2 and the round 2
    difference is of order 0.1 to 0.5, about 4 orders of magnitude above the one-round bound."""
    sizes, max_weight = [300, 600], 1000.0
    p = np.asarray(sizes, dtype=np.float64) / sum(sizes)
    data = [(1.0, -1.0), (-1.0, -1.0)]                       # (x, y)

    def local(theta, xy):
        w, u = theta
        x, y = xy
        pre = w * x
        act = pre > 0
        r = u * (pre if act else 0.0) - y
        gw = r * u * x if act else 0.0
        gu = r * pre if act else 0.0
        return np.array([w - 1.0 * gw, u - 1.0 * gu])

    theta0 = np.array([0.5, 1.0])
    states1 = [local(theta0, xy) for xy in data]
    ex1 = sum(pi * s for pi, s in zip(p, states1))
    ex2 = sum(pi * local(ex1, xy) for pi, xy in zip(p, data))
    _, _, total_q = exact_means(states1, sizes, max_weight)
    one_round_bound = float(np.max(stated_bound(2, total_q) + weight_term(states1, total_q)))
    rows = []
    for s in seeds:
        r1 = secagg_aggregate(states1, sizes, max_weight=max_weight, rng_seed=s)
        r2 = secagg_aggregate([local(r1, xy) for xy in data], sizes, max_weight=max_weight, rng_seed=s + 10 ** 6)
        rows.append({"seed": s, "round1_error_max": float(np.abs(r1 - ex1).max()),
                     "round1_w": float(r1[0]), "round2_difference_max": float(np.abs(r2 - ex2).max())})
    off_kink = [r for r in rows if r["round1_w"] != 0.0]
    return {"exact_round1_model": ex1.tolist(), "exact_round2_model": ex2.tolist(), "one_round_bound": one_round_bound,
            "naive_two_round_tolerance_2x": 2 * one_round_bound, "seeds": len(rows),
            "seeds_with_round1_w_off_the_kink": len(off_kink),
            "round1_inside_bound": all(r["round1_error_max"] <= one_round_bound for r in rows),
            "min_round2_difference_off_kink": min(r["round2_difference_max"] for r in off_kink),
            "max_round2_difference": max(r["round2_difference_max"] for r in rows),
            "min_round2_over_one_round_bound": min(r["round2_difference_max"] for r in off_kink) / one_round_bound,
            "fraction_above_2x_one_round_bound": float(np.mean([r["round2_difference_max"] > 2 * one_round_bound for r in rows]))}


def relu_digits_reproduction(trials):
    try:
        import probe_secagg_error_propagation as probe
        return probe.relu_flip_study(trials=trials, seed=7)
    except Exception as exc:     # data or deps missing
        return {"unavailable": repr(exc)}


def tolerance_limit_note():
    return {"confidence_of_max_covering_p99": 1 - 0.99 ** 300, "p_all_300_below_true_p99": 0.99 ** 300,
            "chance_new_draw_exceeds_max_of_300": 1 / 301}


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-cases", type=int, default=20000)
    ap.add_argument("--e1-float64-cases", type=int, default=2000)
    ap.add_argument("--e3-trials", type=int, default=300)
    ap.add_argument("--e3-d", type=int, default=64)
    ap.add_argument("--m1-trials", type=int, default=1000)
    ap.add_argument("--bias-batches", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timing = {}

    def timed(name, fn, *args, **kw):
        s = time.time()
        r = fn(*args, **kw)
        timing[name] = round(time.time() - s, 1)
        print(f"{name}: {timing[name]} s", flush=True)
        return r

    e1 = timed("e1_float32", run_e1, a.e1_cases, 20261123, np.float32)
    e1_64 = timed("e1_float64_control", run_e1, a.e1_float64_cases, 20261124, np.float64)
    e2 = timed("e2", run_e2, a.bias_batches, 1_000_000, 20261125)
    e3 = timed("e3", run_e3, (4, 16, 64, 256), a.e3_trials, a.e3_d, 20261126)
    limit = overflow_limit_check()
    m1 = timed("m1_convex", run_m1_convex, a.m1_trials, 20261127)
    m1_neg = timed("m1_negative_control", run_m1_negative_control, 100, 20261128)
    relu = relu_counterexample(range(200))
    relu_digits = timed("relu_digits", relu_digits_reproduction, 60)
    saved = None
    saved_path = ROOT / "reports/results/secagg/v014_dropout_matrix/error_propagation.json"
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text()).get("relu_flip_study")

    search = {"schema_version": 1, "script": "scripts/run_secagg_bound_search.py",
              "versions": {"flwr": flwr.__version__, "numpy": np.__version__, "python": platform.python_version()},
              "constants": {"clipping_range": C, "quantization_range": Q, "modulus_range": MOD,
                            "largest_n_without_possible_wrap": MAX_N},
              "overflow_limit": limit, "e1_float32": e1, "e1_float64_control": e1_64, "e2_float32_path": e2,
              "e3_hoeffding": e3, "m1_convex": m1, "m1_negative_control_step_above_2_over_beta": m1_neg,
              "relu_counterexample_tiny": relu, "relu_digits_reproduction": relu_digits,
              "relu_digits_saved_evidence": {"file": "reports/results/secagg/v014_dropout_matrix/error_propagation.json",
                                             "relu_flip_study": saved},
              "tolerance_limit": tolerance_limit_note(), "timing_seconds": timing}
    (a.out_dir / "secagg_bound_search.json").write_text(json.dumps(search, indent=2) + "\n")

    # --- pre-registration check (thresholds copied from docs/preregistration/v0.15.md) ---
    by_n = {row["n"]: row for row in e3}
    e3_indep = all(by_n[n]["independent"]["below_bound"] for n in (4, 16, 64, 256))
    e3_single = all(by_n[n]["independent"]["below_single_bound"] for n in (4, 16, 64, 256))
    e3_shared = (not by_n[256]["shared_seed_identical_data"]["below_bound"])
    f32_class_max = e2["bias_table_units"]["all"]["max_abs_bias_units"]
    prereg = {
        "schema_version": 1, "preregistration": "docs/preregistration/v0.15.md",
        "note": "Thresholds are copied verbatim from the pre-registration and were not changed.",
        "E1": {"prediction": "The stated bound holds",
               "pass_rule": "Zero counterexamples in at least 10,000 property-test cases with unequal sizes and float32 inputs",
               "observed": {"cases_float32": e1["cases"], "violations_integer_weight_mean": e1["violations_integer_weight_mean"],
                            "violations_true_mean_with_weight_term": e1["violations_true_mean"],
                            "max_error_over_stated_bound": e1["max_error_over_stated_bound"]},
               "pass": bool(e1["cases"] >= 10000 and e1["violations_integer_weight_mean"] == 0 and e1["violations_true_mean"] == 0)},
        "E2": {"prediction": "The float32 client path adds a bias the float64 bound misses",
               "pass_rule": "A counterexample to the float64-only bound exists for float32 inputs, or the float32 term is shown to be below 1e-6 of the bound",
               "observed": {"clause_1_counterexample_to_2cn_over_T_found": bool(e1["cases_above_float64_only_bound_2cn_over_T"] > 0
                                                                         or e2["cell_lemma_violations"] > 0
                                                                         or e2["worst_single_client_error_units_either_rounding"] >= 1),
                            "random_cases_above_2cn_over_T": e1["cases_above_float64_only_bound_2cn_over_T"],
                            "cell_lemma_violations": e2["cell_lemma_violations"],
                            "worst_single_client_error_units_either_rounding": e2["worst_single_client_error_units_either_rounding"],
                            "clause_2_float32_term_over_bound": f32_class_max,
                            "float32_bias_max_abs_units_measured": f32_class_max,
                            "float32_bias_theoretical_sup_units": 3.0 / 16.0,
                            "elements": e2["elements"]},
               "pass": bool(e1["cases_above_float64_only_bound_2cn_over_T"] > 0 or e2["cell_lemma_violations"] > 0
                            or e2["worst_single_client_error_units_either_rounding"] >= 1 or f32_class_max < 1e-6),
               "reading": "The float32 path does add a deterministic bias of up to 3/16 of a quantisation unit per client. "
                          "It never pushes a client error past one unit, so 2cn/T survives. Neither clause of the rule holds."},
        "E3": {"prediction": "The Hoeffding bound holds with independent client RNG and fails with a shared seed",
               "pass_rule": "Empirical 99th percentile below the bound for n 4 to 256 with independent RNG, and a violation at n 256 with a shared seed",
               "observed": {str(n): {"independent_p99_over_bound": by_n[n]["independent"]["p99_max_over_bound"],
                                     "shared_identical_p99_over_bound": by_n[n]["shared_seed_identical_data"]["p99_max_over_bound"],
                                     "shared_different_data_p99_over_bound": by_n[n]["shared_seed_different_data"]["p99_max_over_bound"]}
                            for n in (4, 16, 64, 256)},
               "independent_below_bound_all_n": e3_indep, "independent_below_single_coordinate_bound_all_n": e3_single,
               "shared_seed_violates_at_256": e3_shared,
               "note_n4": "At n=4 the Hoeffding bound with d=64 and delta=0.01 is larger than the worst case bound 2cn/T, so the check there is weak.", "trials_per_cell": a.e3_trials, "coordinates": a.e3_d,
               "pass": bool(e3_indep and e3_shared)},
        "M1": {"prediction": "A two-round ReLU counterexample exists and the L2 convex theorem holds in tests",
               "pass_rule": "Counterexample reproduced, zero violations of the L2 bound in at least 1,000 convex trials",
               "observed": {"convex_trials": m1["trials"], "l2_bound_violations": m1["l2_bound_violations"],
                            "trials_with_step_exactly_two_over_beta": m1["trials_with_step_exactly_two_over_beta"],
                            "negative_control_violations_step_1.3_over_beta": m1_neg["l2_bound_violations"],
                            "tiny_relu_counterexample_min_round2_over_one_round_bound": relu["min_round2_over_one_round_bound"],
                            "tiny_relu_fraction_above_2x_bound": relu["fraction_above_2x_one_round_bound"],
                            "digits_relu_reproduction_fraction_above_2x_bound": relu_digits.get("fraction_above_2x_bound")},
               "pass": bool(m1["trials"] >= 1000 and m1["l2_bound_violations"] == 0
                            and relu["min_round2_over_one_round_bound"] > 2 and relu["round1_inside_bound"])},
        "total_runtime_seconds": round(time.time() - t0, 1)}
    (a.out_dir / "prereg_check_proofs.json").write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else None) for k, v in prereg.items()
                      if k in ("E1", "E2", "E3", "M1")}))


if __name__ == "__main__":
    main()
