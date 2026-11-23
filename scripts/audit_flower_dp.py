#!/usr/bin/env python3
"""Audit of Flower's differential privacy mechanisms (v0.16 P1).

Runs the real Flower classes in process on crafted client results and compares
the Gaussian mechanism that actually runs with the one implied by the
configured parameters. See docs/preregistration/v0.16-p1.md for the rows.

Usage: python scripts/audit_flower_dp.py --out DIR [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from scipy import optimize, stats

import flwr
from flwr.common import (
    Code,
    Context,
    FitIns,
    FitRes,
    Message,
    RecordDict,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

try:  # flwr 1.27
    from flwr.common import recorddict_compat as compat
except ImportError:  # newer releases use ArrayRecord messages in LocalDpMod
    compat = None
flwr_logger = logging.getLogger("flwr")
try:  # newer releases need an identity before a Message can be built
    from flwr.supercore.task_identity import TaskIdentity

    TaskIdentity.task_id = TaskIdentity.run_id = TaskIdentity.node_id = 1
except ImportError:
    pass
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    DifferentialPrivacyServerSideAdaptiveClipping,
    DifferentialPrivacyServerSideFixedClipping,
    FedAvg,
)

DIM = 16
C = 1.0
Z = 1.0
N = 10
DELTA = 1e-5
ALPHA = 0.05
OK = Status(code=Code.OK, message="")


# ---------------------------------------------------------------- analytic
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


# ---------------------------------------------------------------- old API
class StubProxy(ClientProxy):
    def get_properties(self, ins, timeout, group_id):  # pragma: no cover
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):  # pragma: no cover
        raise NotImplementedError

    def fit(self, ins, timeout, group_id):  # pragma: no cover
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):  # pragma: no cover
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):  # pragma: no cover
        raise NotImplementedError


def e1() -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def null_update() -> np.ndarray:
    """Stand-in for a zero update.

    An exactly zero update raises ZeroDivisionError inside Flower's clipping
    helper (recorded as row C0), so the zero-out world uses an update of norm
    1e-6 on the last coordinate. Its effect on the measured sensitivity is
    below 1e-5 relative.
    """
    v = np.zeros(DIM, dtype=np.float32)
    v[-1] = 1e-6
    return v


def other_updates(count: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Fixed non-target updates, each with norm 3C so that they are clipped."""
    out = []
    for _ in range(count):
        v = rng.normal(size=DIM)
        out.append((3 * C * v / np.linalg.norm(v)).astype(np.float32))
    return out


class OldApi:
    """Drive flwr.server.strategy wrappers through configure_fit/aggregate_fit."""

    name = "flwr.server.strategy"

    def __init__(self, wrapper_cls, n_cfg: int, z: float, inplace=True, **kw) -> None:
        inner = FedAvg(
            inplace=inplace,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=n_cfg,
            min_available_clients=n_cfg,
        )
        if wrapper_cls is DifferentialPrivacyServerSideAdaptiveClipping:
            self.strategy = wrapper_cls(inner, z, n_cfg, initial_clipping_norm=C, **kw)
        else:
            self.strategy = wrapper_cls(inner, z, C, n_cfg)
        self.manager = SimpleClientManager()
        self.proxies = [StubProxy(str(i)) for i in range(n_cfg)]
        for p in self.proxies:
            self.manager.register(p)
        self.g = [np.zeros(DIM, dtype=np.float32)]
        self.extra: list[np.ndarray] = []

    def run(self, updates: list[np.ndarray], weights: list[int]) -> np.ndarray:
        glob = self.g + self.extra
        self.strategy.configure_fit(1, ndarrays_to_parameters(glob), self.manager)
        results = []
        for i, (u, w) in enumerate(zip(updates, weights)):
            arrs = [glob[0] + u] + [a.copy() for a in self.extra]
            results.append(
                (self.proxies[i], FitRes(OK, ndarrays_to_parameters(arrs), w, {}))
            )
        params, _ = self.strategy.aggregate_fit(1, results, [])
        self.last = parameters_to_ndarrays(params)
        return self.last[0].astype(np.float64)


class NewApi:
    """Drive flwr.serverapp.strategy wrappers through aggregate_train."""

    name = "flwr.serverapp.strategy"

    def __init__(self, n_cfg: int, z: float) -> None:
        from flwr.serverapp import strategy as s

        self.ArrayRecord = __import__("flwr.app", fromlist=["ArrayRecord"]).ArrayRecord
        self.MetricRecord = __import__(
            "flwr.app", fromlist=["MetricRecord"]
        ).MetricRecord
        inner = s.FedAvg(fraction_train=1.0, fraction_evaluate=0.0)
        self.strategy = s.DifferentialPrivacyServerSideFixedClipping(
            inner, z, C, n_cfg
        )
        self.g = np.zeros(DIM, dtype=np.float32)

    def run(self, updates: list[np.ndarray], weights: list[int]) -> np.ndarray:
        # configure_train only stores the arrays before delegating to the grid.
        self.strategy.current_arrays = self.ArrayRecord([self.g])
        replies = []
        for i, (u, w) in enumerate(zip(updates, weights)):
            ask = Message(RecordDict(), dst_node_id=i + 1, message_type="train")
            content = RecordDict(
                {
                    "arrays": self.ArrayRecord([self.g + u]),
                    "metrics": self.MetricRecord({"num-examples": w}),
                }
            )
            replies.append(Message(content, reply_to=ask))
        arrays, _ = self.strategy.aggregate_train(1, replies)
        return arrays.to_numpy_ndarrays()[0].astype(np.float64)


def noise_stats(samples: np.ndarray) -> dict:
    """Per-coordinate noise scale and a normality check on one coordinate."""
    sd = samples.std(axis=0, ddof=1)
    col = samples[:, 1]
    ks = stats.kstest((col - col.mean()) / col.std(ddof=1), "norm")
    return {
        "runs": int(samples.shape[0]),
        "sigma": float(np.sqrt((sd**2).mean())),
        "sigma_min_coord": float(sd.min()),
        "sigma_max_coord": float(sd.max()),
        "ks_pvalue": float(ks.pvalue),
    }


def central_row(
    row, api_factory, n_cfg, weights, n_results, runs, trials, note, do_cp=True
) -> dict:
    rng = np.random.default_rng(1600 + n_results)
    others = other_updates(n_results - 1, rng)
    target = 5 * C * e1()
    zero = null_update()

    quiet = api_factory(0.0)
    out1 = quiet.run([target] + others, weights)
    out0 = quiet.run([zero] + others, weights)
    outm = quiet.run([-target] + others, weights)
    sens_zero = float(np.linalg.norm(out1 - out0))
    sens_repl = float(np.linalg.norm(out1 - outm))

    api = api_factory(Z)
    np.random.seed(20261123)
    samples = np.stack([api.run([zero] + others, weights) for _ in range(runs)])
    ns = noise_stats(samples - out0)
    sigma = ns["sigma"]
    mu = sens_zero / sigma
    claimed_mu = 1.0 / Z
    res = {
        "row": row,
        "api": api.name,
        "note": note,
        "configured": {
            "noise_multiplier": Z,
            "clipping_norm": C,
            "num_sampled_clients": n_cfg,
            "results_received": n_results,
            "weights": weights,
            "target_weight_share": weights[0] / sum(weights),
        },
        "claimed": {
            "sigma": Z * C / n_cfg,
            "mu": claimed_mu,
            "eps_at_delta": gauss_eps(claimed_mu, DELTA),
        },
        "measured": {
            "sensitivity_zero_out": sens_zero,
            "sensitivity_replace_one": sens_repl,
            "noise": ns,
            "mu_zero_out": mu,
            "mu_replace_one": sens_repl / sigma,
            "effective_noise_multiplier": 1.0 / mu,
            "eps_at_delta_zero_out": gauss_eps(mu, DELTA),
            "eps_at_delta_replace_one": gauss_eps(sens_repl / sigma, DELTA),
        },
        "delta": DELTA,
    }
    res["mu_ratio_measured_over_claimed"] = mu / claimed_mu
    res["matches_claim_within_2pct"] = bool(abs(mu / claimed_mu - 1) <= 0.02)

    if do_cp and trials:
        predicted_mu = (weights[0] / sum(weights)) * n_cfg / Z
        thr_sigma = planned_threshold(predicted_mu, trials)
        thr = thr_sigma * (Z * C / n_cfg)
        d = e1().astype(np.float64)
        np.random.seed(20261124)
        p0 = np.array([api.run([zero] + others, weights) @ d for _ in range(trials)])
        p1 = np.array([api.run([target] + others, weights) @ d for _ in range(trials)])
        emp = empirical_eps_lower(p0 - out0 @ d, p1 - out0 @ d, thr)
        emp["threshold_in_claimed_sigma"] = thr_sigma
        emp["exceeds_claimed_eps"] = bool(
            emp["eps_lower_bound"] > res["claimed"]["eps_at_delta"]
        )
        res["empirical"] = emp
    return res


# ---------------------------------------------------------------- rows
def row_c0() -> dict:
    """Exactly zero client update."""
    api = OldApi(DifferentialPrivacyServerSideFixedClipping, N, Z)
    rng = np.random.default_rng(1600)
    ups = [np.zeros(DIM, dtype=np.float32)] + other_updates(N - 1, rng)
    try:
        api.run(ups, [1] * N)
        exc = None
    except Exception as e:  # recorded, not hidden
        exc = f"{type(e).__name__}: {e}"
    return {
        "row": "C0",
        "api": api.name,
        "note": "not pre-registered. One client returns parameters identical "
        "to the global model. Found while building the zero-out world.",
        "exception": exc,
    }


def row_c5(runs: int) -> dict:
    """Client-side clipping variant with one client that does not clip."""
    rng = np.random.default_rng(1605)
    others = [u / 3 for u in other_updates(N - 1, rng)]  # honest, norm C
    target = 5 * C * e1()
    zero = null_update()
    w = [1] * N
    quiet = OldApi(DifferentialPrivacyClientSideFixedClipping, N, 0.0)
    sens = float(
        np.linalg.norm(quiet.run([target] + others, w) - quiet.run([zero] + others, w))
    )
    api = OldApi(DifferentialPrivacyClientSideFixedClipping, N, Z)
    np.random.seed(5)
    base = quiet.run([zero] + others, w)
    ns = noise_stats(
        np.stack([api.run([zero] + others, w) for _ in range(runs)]) - base
    )
    return {
        "row": "C5",
        "api": api.name,
        "note": "client-side clipping wrapper, target client sends an update of "
        "norm 5C without clipping. The server does not clip in this variant.",
        "claimed": {"mu": 1.0 / Z},
        "measured": {
            "sensitivity_zero_out": sens,
            "noise": ns,
            "mu_zero_out": sens / ns["sigma"],
            "sensitivity_scales_with_update_norm": True,
        },
    }


def row_c6(runs: int) -> dict:
    """Model that contains an integer-typed array."""
    out: dict = {"row": "C6", "api": "flwr.server.strategy"}
    w = [1] * N
    rng = np.random.default_rng(1606)
    zero = null_update()

    def make(z, inplace=True):
        api = OldApi(DifferentialPrivacyServerSideFixedClipping, N, z, inplace=inplace)
        api.extra = [np.array([7, 7], dtype=np.int64)]
        return api

    # Unclipped case: all updates have norm below C.
    small = [u / 30 for u in other_updates(N, rng)]
    api = make(Z)
    np.random.seed(6)
    ints = []
    try:
        for _ in range(runs):
            api.run(small, w)
            ints.append(api.last[1].astype(np.float64))
        arr = np.stack(ints)
        out["unclipped_case"] = {
            "runs": runs,
            "float_array_sigma_claimed": Z * C / N,
            "integer_array_dtype_after_aggregation": str(api.last[1].dtype),
            "integer_array_noise_sigma": float(arr.std(axis=0, ddof=1).mean()),
            "integer_array_fraction_of_runs_changed": float(
                (arr != arr[0]).any(axis=1).mean()
            ),
            "integer_array_value": arr[0].tolist(),
        }
    except Exception as exc:  # recorded, not hidden
        out["unclipped_case"] = {"exception": f"{type(exc).__name__}: {exc}"}

    # Same input with FedAvg(inplace=False), which averages in floating point.
    api = make(Z, inplace=False)
    np.random.seed(6)
    vals = []
    for _ in range(runs):
        api.run(small, w)
        vals.append(api.last[1].astype(np.float64))
    arr = np.stack(vals)
    out["unclipped_case_fedavg_inplace_false"] = {
        "runs": runs,
        "integer_array_dtype_after_aggregation": str(api.last[1].dtype),
        "integer_array_noise_sigma": float(arr.std(axis=0, ddof=1).mean()),
        "integer_array_mean": float(arr.mean()),
    }
    out["note"] = (
        "every client sends the integer array [7, 7]. With the default "
        "FedAvg(inplace=True) the integer array is aggregated to 0 before any "
        "noise step, so it carries no client information. With inplace=False "
        "it is averaged as float64 and receives the configured noise."
    )

    # Clipped case: one update has norm above C.
    try:
        make(Z).run([5 * C * e1()] + small[1:], w)
        out["clipped_case"] = {"exception": None}
    except Exception as exc:
        out["clipped_case"] = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def row_adaptive(row: str, lr: float, runs: int) -> dict:
    z_cfg = 0.5
    rng = np.random.default_rng(1610)
    updates = other_updates(N, rng)  # all norm 3C, all clipped
    w = [1] * N

    quiet = OldApi(
        DifferentialPrivacyServerSideAdaptiveClipping, N, 0.0, clip_norm_lr=lr
    )
    zero = null_update()
    quiet.strategy.clipping_norm = C
    o1 = quiet.run([5 * C * e1()] + updates[1:], w)
    quiet.strategy.clipping_norm = C
    o0 = quiet.run([zero] + updates[1:], w)
    quiet.strategy.clipping_norm = C
    base = quiet.run(updates, w)
    sens = float(np.linalg.norm(o1 - o0))

    api = OldApi(DifferentialPrivacyServerSideAdaptiveClipping, N, z_cfg, clip_norm_lr=lr)
    z_delta = float(api.strategy.noise_multiplier)
    np.random.seed(10)
    outs, norms = [], []
    for _ in range(runs):
        api.strategy.clipping_norm = C
        outs.append(api.run(updates, w))
        norms.append(api.strategy.clipping_norm)
    noise = np.stack(outs) - base
    norms = np.array(norms)
    ref_sigma = z_delta * C / N
    sigma = float(np.sqrt((noise**2).mean()))
    standardised = noise / (z_delta * norms[:, None] / N)
    return {
        "row": row,
        "api": api.name,
        "note": "all N updates are clipped with the round's norm C_t = C. The "
        "reference algorithm (Andrew et al. 2021) scales noise with C_t.",
        "configured": {
            "noise_multiplier": z_cfg,
            "internal_update_noise_multiplier": z_delta,
            "clipped_count_stddev": float(api.strategy.clipped_count_stddev),
            "clip_norm_lr": lr,
            "target_clipped_quantile": 0.5,
            "num_sampled_clients": N,
            "clipping_norm_used_for_clipping": C,
        },
        "predicted_ratio": math.exp(-lr * 0.5),
        "measured": {
            "runs": runs,
            "sensitivity_zero_out": sens,
            "reference_sigma": ref_sigma,
            "sigma": sigma,
            "sigma_ratio_to_reference": sigma / ref_sigma,
            "mean_updated_norm_over_clip_norm": float(norms.mean() / C),
            "min_updated_norm_over_clip_norm": float(norms.min() / C),
            "noise_std_in_units_of_updated_norm_sigma": float(standardised.std()),
            "mu_zero_out": sens / sigma,
            "mu_reference": sens / ref_sigma,
        },
    }


def local_dp(runs: int) -> list[dict]:
    g = [np.zeros(DIM, dtype=np.float32)]
    ctxt = Context(run_id=0, node_id=0, node_config={}, state=RecordDict(), run_config={})
    if compat is not None:
        from flwr.client.mod import LocalDpMod

        api_name = "flwr.client.mod.LocalDpMod"
        ask = Message(
            compat.fitins_to_recorddict(FitIns(ndarrays_to_parameters(g), {}), True),
            dst_node_id=1,
            message_type="train",
        )

        def call(mod, update: np.ndarray) -> np.ndarray:
            def app(msg, _ctxt):
                res = FitRes(OK, ndarrays_to_parameters([g[0] + update]), 1, {})
                return Message(compat.fitres_to_recorddict(res, True), reply_to=msg)

            out = mod(ask, ctxt, app)
            res = compat.recorddict_to_fitres(out.content, True)
            return parameters_to_ndarrays(res.parameters)[0].astype(np.float64)

    else:
        from flwr.app import ArrayRecord
        from flwr.clientapp.mod import LocalDpMod

        api_name = "flwr.clientapp.mod.LocalDpMod"
        ask = Message(
            RecordDict({"arrays": ArrayRecord(g)}), dst_node_id=1, message_type="train"
        )

        def call(mod, update: np.ndarray) -> np.ndarray:
            def app(msg, _ctxt):
                content = RecordDict({"arrays": ArrayRecord([g[0] + update])})
                return Message(content, reply_to=msg)

            out = mod(ask, ctxt, app)
            rec = next(iter(out.content.array_records.values()))
            return rec.to_numpy_ndarrays()[0].astype(np.float64)

    up, down = 5 * C * e1(), -5 * C * e1()
    quiet = LocalDpMod(C, 0.0, 1.0, DELTA)
    sens = float(np.linalg.norm(call(quiet, up) - call(quiet, down)))
    base = call(quiet, up)

    rows = []
    for label, s_param in (("L1", 2 * C), ("L2", C)):
        for eps in (0.5, 1, 2, 5, 10, 20):
            mod = LocalDpMod(C, s_param, eps, DELTA)
            np.random.seed(int(eps * 10) + 7)
            ns = noise_stats(np.stack([call(mod, up) for _ in range(runs)]) - base)
            claimed_sigma = s_param * math.sqrt(2 * math.log(1.25 / DELTA)) / eps
            mu = sens / ns["sigma"]
            mu_formula = sens / claimed_sigma
            rows.append(
                {
                    "row": label,
                    "api": api_name,
                    "configured": {
                        "clipping_norm": C,
                        "sensitivity": s_param,
                        "epsilon": eps,
                        "delta": DELTA,
                    },
                    "measured": {
                        "sensitivity_replace_one": sens,
                        "noise": ns,
                        "sigma_formula": claimed_sigma,
                        "sigma_ratio_measured_over_formula": ns["sigma"] / claimed_sigma,
                        "mu": mu,
                    },
                    "exact_from_formula_sigma": {
                        "delta_at_configured_eps": gauss_delta(mu_formula, eps),
                        "delta_ratio_to_configured": gauss_delta(mu_formula, eps) / DELTA,
                        "eps_at_configured_delta": gauss_eps(mu_formula, DELTA),
                    },
                    "exact_from_measured_sigma": {
                        "delta_at_configured_eps": gauss_delta(mu, eps),
                        "eps_at_configured_delta": gauss_eps(mu, DELTA),
                    },
                    "meets_configured_guarantee": bool(
                        gauss_delta(mu_formula, eps) <= DELTA
                    ),
                }
            )
    return rows


def accounted_rows() -> list[dict]:
    """Message API fixed clipping with the optional accountant, when present."""
    try:
        from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
        from flwr.serverapp import strategy as s
        from flwr.serverapp.grid import Grid
        from flwr.supercore.privacy_accounting import (
            NeighboringRelation,
            PrivacyConfig,
            SamplingMethod,
        )
        from flwr.supercore.privacy_accounting.rdp_accountant import RdpAccountant
    except ImportError as exc:
        return [
            {
                "row": "ACC",
                "available": False,
                "note": "this flwr version has no accountant option on the "
                f"fixed clipping strategy ({type(exc).__name__}).",
            }
        ]
    g = np.zeros(DIM, dtype=np.float32)
    rng = np.random.default_rng(1620)
    others = other_updates(N - 1, rng)
    cases = {
        "C1-msg-acc": ([1] * N, N, "equal weights, all sampled clients respond"),
        "C3-msg-acc": ([91] + [1] * (N - 1), N, "target reports num_examples 91"),
        "C4-msg-acc": ([1, 1], 2, "2 of 10 sampled clients respond"),
    }
    rows = []
    for row, (weights, n_results, note) in cases.items():
        rec: dict = {"row": row, "api": "flwr.serverapp.strategy", "note": note,
                     "available": True}
        try:
            cfg = PrivacyConfig(
                target_delta=DELTA,
                population_size=N,
                neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
                sampling_method=SamplingMethod.NO_AMPLIFICATION,
            )
            acc = RdpAccountant(cfg)
            inner = s.FedAvg(
                fraction_train=1.0, fraction_evaluate=0.0,
                min_train_nodes=N, min_available_nodes=N,
            )
            wrap = s.DifferentialPrivacyServerSideFixedClipping(
                inner, Z, C, N, accountant=acc
            )
            grid = Mock(spec=Grid)
            grid.get_node_ids.return_value = list(range(1, N + 1))
            asks = list(wrap.configure_train(1, ArrayRecord([g]), ConfigRecord(), grid))
            rec["messages_configured"] = len(asks)
            ups = ([5 * C * e1()] + others)[:n_results]
            replies = []
            for i, (u, w) in enumerate(zip(ups, weights)):
                ask = Message(RecordDict(), dst_node_id=i + 1, message_type="train")
                content = RecordDict(
                    {
                        "arrays": ArrayRecord([g + u]),
                        "metrics": MetricRecord({"num-examples": w}),
                    }
                )
                replies.append(Message(content, reply_to=ask))
            arrays, _ = wrap.aggregate_train(1, replies)
            rec["aggregate_released"] = arrays is not None
            spent = wrap.privacy_spent()
            rec["privacy_spent"] = {
                "epsilon": spent.epsilon, "delta": spent.delta,
                "num_releases": spent.num_releases,
                "accounting_method": spent.accounting_method,
            }
            rec["exact_eps_for_one_release_mu_1_over_z"] = gauss_eps(1.0 / Z, DELTA)
            rec["exception"] = None
        except Exception as exc:  # recorded, not hidden
            rec["aggregate_released"] = False
            rec["exception"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(rec)
    return rows


def row_r1() -> dict:
    root = Path(flwr.__file__).parent
    hits = []
    for p in sorted(root.rglob("*.py")):
        text = p.read_text(errors="ignore").lower()
        if p.name.endswith("_test.py"):
            continue
        for term in ("accountant", "renyi", "privacy_spent", "get_epsilon", "rdp_"):
            if term in text:
                hits.append({"file": str(p.relative_to(root)), "term": term})
    return {
        "row": "R1",
        "statement": "the installed flwr package ships no privacy accountant",
        "search_terms": ["accountant", "renyi", "privacy_spent", "get_epsilon", "rdp_"],
        "hit_count": len(hits),
        "hit_files": sorted({h["file"] for h in hits})[:20],
        "true": len(hits) == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="small run for tests")
    args = ap.parse_args()
    runs = 500 if args.quick else 20_000
    trials = 2_000 if args.quick else 200_000

    flwr_logger.setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    rows: list[dict] = []

    def old(n_cfg):
        return lambda z: OldApi(DifferentialPrivacyServerSideFixedClipping, n_cfg, z)

    def new(n_cfg):
        return lambda z: NewApi(n_cfg, z)

    for tag, fac in (("", old), ("-msg", new)):
        rows.append(
            central_row(
                "C1" + tag, fac(N), N, [1] * N, N, runs, 0,
                "equal weights, all sampled clients respond. C2 is the "
                "replace-one column of this row.", do_cp=False,
            )
        )
        rows.append(
            central_row(
                "C3" + tag, fac(N), N, [91] + [1] * (N - 1), N, runs, trials,
                "target client reports num_examples 91, the other nine report 1.",
            )
        )
        rows.append(
            central_row(
                "C4" + tag, fac(N), N, [1, 1], 2, runs, trials,
                "2 of the 10 configured clients return results. Flower logs a "
                "warning in this configuration.",
            )
        )
        print(f"central rows {tag or 'old'} done {time.time() - t0:.0f}s", flush=True)
    rows.append(row_c0())
    rows.append(row_c5(runs))
    rows.append(row_c6(min(runs, 2000)))
    rows.append(row_adaptive("A1", 0.2, runs))
    rows.append(row_adaptive("A2", 1.0, runs))
    print(f"adaptive done {time.time() - t0:.0f}s", flush=True)
    rows.extend(local_dp(runs))
    rows.extend(accounted_rows())
    rows.append(row_r1())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "feddisclosure.flower_dp_audit.v1",
        "quick": args.quick,
        "environment": {
            "flwr": flwr.__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "python": platform.python_version(),
        },
        "parameters": {
            "dim": DIM, "clipping_norm": C, "noise_multiplier": Z,
            "num_sampled_clients": N, "delta": DELTA,
            "noise_runs": runs, "trials_per_world": trials,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "flower_dp_audit.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out / 'flower_dp_audit.json'} in {doc['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
