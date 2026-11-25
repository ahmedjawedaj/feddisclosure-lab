#!/usr/bin/env python3
"""Measurement of the effective noise multiplier of Flower's whole adaptive clipping mechanism (v0.16 T8c).

Runs the four adaptive and the four fixed clipping strategy classes in process with a FedAvg inner
strategy and fake client replies, over simulated trajectories of the clip. The rows, predictions and
tolerances are fixed in docs/preregistration/v0.16-adaptive-guarantee.md, which was committed before
this script existed. No epsilon or delta is computed.

Usage:
  python scripts/audit_flower_adaptive_guarantee.py --out DIR [--quick]
  python scripts/audit_flower_adaptive_guarantee.py --compare DIR_A DIR_B --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import logging
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import flwr
from flwr.common import (
    Code,
    Context,
    FitRes,
    Message,
    RecordDict,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

try:  # newer releases need an identity before a Message can be built
    from flwr.supercore.task_identity import TaskIdentity

    TaskIdentity.task_id = TaskIdentity.run_id = TaskIdentity.node_id = 1
except ImportError:
    pass

try:
    from flwr.supercore.differential_privacy import (
        KEY_CLIPPING_NORM,
        KEY_NORM_BIT,
        adaptive_clip_inputs_inplace,
        compute_stdv,
    )
except ImportError:
    from flwr.common.differential_privacy import adaptive_clip_inputs_inplace, compute_stdv
    from flwr.common.differential_privacy_constants import KEY_CLIPPING_NORM, KEY_NORM_BIT
try:
    from flwr.common import recorddict_compat as compat
except ImportError:
    from flwr.compat.common import recorddict_compat as compat

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from flwr.client.mod import adaptiveclipping_mod as legacy_adaptive_mod
from flwr.client.mod import fixedclipping_mod as legacy_fixed_mod
from flwr.clientapp.mod import adaptiveclipping_mod as message_adaptive_mod
from flwr.clientapp.mod import fixedclipping_mod as message_fixed_mod
from flwr.server import strategy as legacy_strategy
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.serverapp import strategy as message_strategy

DIM = 256
N = 100
Q = 0.5
Z = 1.0
OK = Status(code=Code.OK, message="")
CTX = Context(run_id=0, node_id=0, node_config={}, state=RecordDict(), run_config={})
ADAPTIVE = [
    ("V1", "legacy", "server", False),
    ("V2", "legacy", "client", False),
    ("V3", "message", "server", False),
    ("V4", "message", "client", False),
]
FIXED = [
    ("F1", "legacy", "server", True),
    ("F2", "legacy", "client", True),
    ("F3", "message", "server", True),
    ("F4", "message", "client", True),
]
SIGMAS = [0.6, 2.5, 5.0, 10.0]
LRS = [0.05, 0.2, 1.0]
STARTS = [0.1, 1.0]
FIXED_Z = [0.5, 1.0, 2.0]
FIXED_CLIPS = [0.1, 1.0]
NORMS = [float(x) for x in np.linspace(0.5, 1.5, N)]
REL = 1e-9
LABEL_LO, LABEL_HI = 0.98, 1.02


# ---------------------------------------------------------------- helpers
def close(a: float, b: float, rel: float = REL) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


def e1() -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def label(x: float) -> str:
    return "equal at this tolerance" if LABEL_LO <= x <= LABEL_HI else "differs from configured"


def strategy_class(v: tuple):
    _, api, side, fixed = v
    mod = legacy_strategy if api == "legacy" else message_strategy
    kind = "Fixed" if fixed else "Adaptive"
    name = f"DifferentialPrivacy{'ServerSide' if side == 'server' else 'ClientSide'}{kind}Clipping"
    return getattr(mod, name)


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


class CaptureFedAvg(message_strategy.FedAvg):
    """FedAvg for the Message API that records the config it is asked to send."""

    cfg = None

    def configure_train(self, server_round, arrays, config, grid):
        self.cfg = config
        return []


def inner_strategy(api: str, n: int):
    if api == "legacy":
        return legacy_strategy.FedAvg(
            inplace=True, fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=n, min_available_clients=n,
        )
    return CaptureFedAvg(fraction_train=1.0, fraction_evaluate=0.0)


class DrawLog:
    """Wraps np.random.normal for one call and forwards to the original, so the stream is unchanged."""

    def __enter__(self):
        self.orig = np.random.normal
        self.calls: list[tuple] = []
        np.random.normal = self
        return self

    def __call__(self, *a, **k):
        ret = self.orig(*a, **k)
        self.calls.append((a, k, ret))
        return ret

    def __exit__(self, *exc):
        np.random.normal = self.orig

    def count_draws(self) -> list[tuple[float, float, float]]:
        return [(float(a[0]), float(a[1]), float(ret)) for a, k, ret in self.calls if len(a) == 2 and not k]


class Runner:
    """Drive one strategy for single rounds with a chosen set of client replies."""

    def __init__(self, variant: tuple, z: float, n: int, c0: float, **kw) -> None:
        self.name, self.api, self.side, self.fixed = variant
        self.n = n
        cls = strategy_class(variant)
        inner = inner_strategy(self.api, n)
        if self.fixed:
            self.strategy = cls(inner, z, c0, n)
        else:
            self.strategy = cls(inner, z, n, initial_clipping_norm=c0, **kw)
        self.g = np.zeros(DIM, dtype=np.float32)
        self.sent_bits: list = []
        self.cfg_clip: float | None = None
        if self.api == "legacy":
            self.manager = SimpleClientManager()
            self.proxies = [StubProxy(str(i)) for i in range(n)]
            for p in self.proxies:
                self.manager.register(p)

    def set_clip(self, c: float) -> None:
        self.strategy.clipping_norm = c

    def run(self, norms, bits=None, mod: bool = False):
        self.sent_bits = []
        if self.api == "legacy":
            return self._run_legacy(norms, bits, mod)
        return self._run_message(norms, bits, mod)

    @staticmethod
    def _bit(bits, i: int) -> int:
        return int(bits[i]) if hasattr(bits, "__len__") else int(i < bits)

    # ------------------------------------------------------------ legacy
    def _legacy_through_mod(self, fit_ins, upd):
        msg = Message(
            compat.fitins_to_recorddict(fit_ins, True), dst_node_id=1, message_type="train"
        )

        def app(m, _ctxt):
            res = FitRes(OK, ndarrays_to_parameters([self.g + upd]), 1, {})
            return Message(compat.fitres_to_recorddict(res, True), reply_to=m)

        mod_fn = legacy_fixed_mod if self.fixed else legacy_adaptive_mod
        out = mod_fn(msg, CTX, app)
        res = compat.recorddict_to_fitres(out.content, True)
        if KEY_NORM_BIT in res.metrics:
            self.sent_bits.append(res.metrics[KEY_NORM_BIT])
        return res

    def _run_legacy(self, norms, bits, mod):
        s = self.strategy
        ins = s.configure_fit(1, ndarrays_to_parameters([self.g]), self.manager)
        if self.side == "client":
            self.cfg_clip = float(ins[0][1].config[KEY_CLIPPING_NORM])
        results = []
        for i, u in enumerate(norms):
            upd = (u * e1()).astype(np.float32)
            proxy, fit_ins = ins[i]
            if self.side == "server":
                res = FitRes(OK, ndarrays_to_parameters([self.g + upd]), 1, {})
            elif mod:
                res = self._legacy_through_mod(fit_ins, upd)
            else:
                metrics = {} if self.fixed else {KEY_NORM_BIT: self._bit(bits, i)}
                res = FitRes(OK, ndarrays_to_parameters([self.g + upd]), 1, metrics)
            results.append((proxy, res))
        params, _ = s.aggregate_fit(1, results, [])
        if params is None:
            return None
        return parameters_to_ndarrays(params)[0].astype(np.float64)

    # ----------------------------------------------------------- message
    def _message_config(self):
        inner = self.strategy.strategy
        inner.cfg = None
        self.strategy.configure_train(1, ArrayRecord([self.g]), ConfigRecord(), None)
        return inner.cfg

    def _run_message(self, norms, bits, mod):
        s = self.strategy
        arrays = ArrayRecord([self.g])
        cfg = None
        if self.side == "server":
            s.current_arrays = arrays
        else:
            cfg = self._message_config()
            self.cfg_clip = float(cfg[KEY_CLIPPING_NORM])
        replies = []
        mod_fn = message_fixed_mod if self.fixed else message_adaptive_mod
        for i, u in enumerate(norms):
            upd = (u * e1()).astype(np.float32)
            ask = Message(RecordDict(), dst_node_id=i + 1, message_type="train")
            if self.side == "client" and mod:
                ask = Message(
                    RecordDict({"arrays": arrays, "config": ConfigRecord(dict(cfg.items()))}),
                    dst_node_id=i + 1, message_type="train",
                )

                def app(m, _ctxt, upd=upd):
                    content = RecordDict({
                        "arrays": ArrayRecord([self.g + upd]),
                        "metrics": MetricRecord({"num-examples": 1}),
                    })
                    return Message(content, reply_to=m)

                rep = mod_fn(ask, CTX, app)
                for _, mrec in rep.content.metric_records.items():
                    if KEY_NORM_BIT in mrec:
                        self.sent_bits.append(mrec[KEY_NORM_BIT])
                replies.append(rep)
                continue
            metrics = {"num-examples": 1}
            if self.side == "client" and not self.fixed:
                metrics[KEY_NORM_BIT] = self._bit(bits, i)
            content = RecordDict({
                "arrays": ArrayRecord([self.g + upd]),
                "metrics": MetricRecord(metrics),
            })
            replies.append(Message(content, reply_to=ask))
        out, _ = s.aggregate_train(1, replies)
        if out is None:
            return None
        return out.to_numpy_ndarrays()[0].astype(np.float64)


def client_bits(norms, clip: float) -> list[int]:
    """The bit Flower's own function returns for each update at this clip."""
    out = []
    for u in norms:
        arr = [(u * e1()).astype(np.float32)]
        out.append(int(adaptive_clip_inputs_inplace(arr, clip)))
    return out


def row(name, variant, prediction, paper, measured, held, agrees, tol, **extra) -> dict:
    d = {
        "row": name,
        "variant": variant[0],
        "api": variant[1],
        "side": variant[2],
        "prediction": prediction,
        "paper_reading": paper,
        "measured": measured,
        "tolerance": tol,
        "prediction_held": bool(held),
        "agrees_with_paper": None if agrees is None else bool(agrees),
    }
    d.update(extra)
    return d


# ------------------------------------------------------------------ G0
G0_CLIPS = [0.3, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.6, 2.0]


def rows_g0(v) -> list[dict]:
    use_mod = v[2] == "client"
    seen: set[float] = set()
    types: set[str] = set()
    bit_ok = True
    loc_ok = True
    scale_ok = True
    detail = []
    for c in G0_CLIPS:
        r = Runner(v, 1.0, N, c, clipped_count_stddev=5.0)
        with DrawLog() as log:
            r.run(NORMS, bits=None, mod=use_mod)
        draws = log.count_draws()
        expected = sum(1 for u in NORMS if u > c)
        ok_loc = len(draws) == 1 and draws[0][0] == float(expected)
        ok_scale = len(draws) == 1 and draws[0][1] == float(r.strategy.clipped_count_stddev) == 5.0
        loc_ok &= ok_loc
        scale_ok &= ok_scale
        item = {"clip": c, "expected_clipped": expected,
                "count_location": draws[0][0] if draws else None,
                "count_scale": draws[0][1] if draws else None}
        if use_mod:
            vals = [float(b) for b in r.sent_bits]
            seen |= set(vals)
            types |= {type(b).__name__ for b in r.sent_bits}
            exp_bits = [1.0 if u > c else 0.0 for u in NORMS]
            bit_ok &= vals == exp_bits
            item["bits_equal_norm_above_clip"] = vals == exp_bits
        detail.append(item)
    out = []
    if use_mod:
        held = bit_ok and seen == {0.0, 1.0}
        out.append(row(
            "G0a", v, {"values": [0.0, 1.0], "bit_is_one_when_norm_exceeds_clip": True},
            {"values": [-0.5, 0.5], "note": "centred bit"},
            {"values_seen": sorted(seen), "python_types": sorted(types), "all_clips_match": bit_ok},
            held, False, "exact"))
    out.append(row(
        "G0b", v, "location of the count draw equals the number of norms above the clip, no shift",
        "sum of centred bits", {"clips": detail}, loc_ok, False, "exact"))
    out.append(row(
        "G0c", v, "scale of the count draw equals the stored sigma_b exactly",
        "sigma_b", {"all_equal": scale_ok, "sigma_b": 5.0}, scale_ok, True, "exact"))

    # G0d: all clients inside the clip, each moved outside once (clip 1.0)
    inside, outside, clip = 0.5, 1.5, 1.0
    deltas = []
    for i in range(N):
        norms = [inside] * N
        norms[i] = outside
        r = Runner(v, 1.0, N, clip, clipped_count_stddev=5.0)
        with DrawLog() as log:
            r.run(norms, mod=use_mod)
        deltas.append(log.count_draws()[0][0])
    # the location with every client inside is 0 by construction of the bits
    r = Runner(v, 1.0, N, clip, clipped_count_stddev=5.0)
    with DrawLog() as log:
        r.run([inside] * N, mod=use_mod)
    base = log.count_draws()[0][0]
    changes = [d - base for d in deltas]
    out.append(row(
        "G0d", v, {"change_when_one_client_moves_outside": 1.0, "cases": N},
        {"change": 1.0, "note": "range of the centred bit"},
        {"base_location": base, "changes_min": min(changes), "changes_max": max(changes)},
        base == 0.0 and all(c == 1.0 for c in changes), True, "exact"))

    # G0e: leave one reply out
    r = Runner(v, 1.0, N, clip, clipped_count_stddev=5.0)
    with DrawLog() as log:
        r.run(NORMS, mod=use_mod)
    full = log.count_draws()[0][0]
    diffs, exp = [], []
    for i in range(N):
        r = Runner(v, 1.0, N, clip, clipped_count_stddev=5.0)
        with DrawLog() as log:
            r.run(NORMS[:i] + NORMS[i + 1:], mod=use_mod)
        diffs.append(full - log.count_draws()[0][0])
        exp.append(1.0 if NORMS[i] > clip else 0.0)
    out.append(row(
        "G0e", v, {"change_equals_left_out_bit": True, "largest_magnitude": 1.0},
        {"magnitude": 0.5, "note": "centred bit"},
        {"changes_equal_bits": diffs == exp, "largest_magnitude": max(abs(d) for d in diffs),
         "ones": int(sum(1 for d in diffs if d == 1.0))},
        diffs == exp and max(abs(d) for d in diffs) == 1.0, False, "exact",
        sensitivity_add_or_remove=1.0, range_of_bit=1.0))
    return out


# ------------------------------------------------------------------ G1
G1_CLIPS = [0.2, 0.7, 1.0, 1.3, 2.0]


def rows_g1(v) -> list[dict]:
    use_mod = v[2] == "client"
    items, ok, cfg_ok, fixed_ok = [], True, True, True
    for c in G1_CLIPS:
        if v[3]:
            r = Runner(v, 0.0, N, c)
        else:
            r = Runner(v, 0.0, N, c, clip_norm_lr=0.2)
        before = r.strategy.clipping_norm
        out = r.run(NORMS, bits=client_bits(NORMS, c) if (v[2] == "client" and not use_mod) else None,
                    mod=use_mod)
        expected = float(np.mean([min(u, before) for u in NORMS]))
        got = float(out[0])
        rel = abs(got - expected) / expected
        this_ok = rel <= 1e-5
        ok &= this_ok
        if r.cfg_clip is not None:
            cfg_ok &= r.cfg_clip == before
        after = r.strategy.clipping_norm
        if v[3]:
            fixed_ok &= after == before
        items.append({"clip_before": before, "clip_after": after, "output_coordinate0": got,
                      "expected_mean_min_norm_clip": expected, "relative_difference": rel,
                      "config_clip": r.cfg_clip})
    pred = {"output_equals_mean_min_norm_clip_before_call": True, "config_clip_equals_clip_before": True}
    if v[3]:
        pred["clip_unchanged_by_the_call"] = True
    return [row("G1", v, pred, "clip of the round used for clipping", {"clips": items},
                ok and cfg_ok and fixed_ok, True, "1e-5 relative, exact for config and fixed clip",
                config_clip_checked=v[2] == "client")]


# ------------------------------------------------------------------ G2 to G4 (adaptive trajectories)
def run_trajectory(v, sb: float, lr: float, c0: float, seed: int, rounds: int) -> dict:
    r = Runner(v, Z, N, c0, clip_norm_lr=lr, clipped_count_stddev=sb)
    zd = float(r.strategy.noise_multiplier)
    sbs = float(r.strategy.clipped_count_stddev)
    np.random.seed(seed)
    rec = {"C": [c0], "loc": [], "scale": [], "ret": [], "ss": []}
    for _ in range(rounds):
        c = r.strategy.clipping_norm
        bits = client_bits(NORMS, c) if v[2] == "client" else None
        with DrawLog() as log:
            out = r.run(NORMS, bits=bits)
        draws = log.count_draws()
        if len(draws) != 1:
            raise RuntimeError(f"expected one count draw, saw {len(draws)}")
        cn = r.strategy.clipping_norm
        s = compute_stdv(zd, cn, N)
        noise = out[1:]
        rec["ss"].append(float(np.sum((noise / s) ** 2)))
        rec["C"].append(float(cn))
        rec["loc"].append(draws[0][0])
        rec["scale"].append(draws[0][1])
        rec["ret"].append(draws[0][2])
    rec["z_delta"] = zd
    rec["sigma_b"] = sbs
    return rec


def zflower(zeta, sb):
    return (zeta ** -2.0 + sb ** -2.0) ** -0.5


def zpaper(zeta, sb):
    return (zeta ** -2.0 + (2.0 * sb) ** -2.0) ** -0.5


def cell_seed(a: int, b: int, c: int, j: int) -> int:
    return 20261201 + 100000 * a + 10000 * b + 1000 * c + j


def analyse_cell(v, sb, lr, c0, trajs: list[dict], rounds: int) -> tuple[list[dict], dict, dict]:
    """Rows, summary and pooled noise sums for one cell."""
    zd, sbs = trajs[0]["z_delta"], trajs[0]["sigma_b"]
    C = np.array([t["C"] for t in trajs])  # (R, T+1)
    loc = np.array([t["loc"] for t in trajs])
    ret = np.array([t["ret"] for t in trajs])
    ss = np.array([t["ss"] for t in trajs])
    c_t, c_n = C[:, :-1], C[:, 1:]
    r_t = c_n / c_t
    ft = ret / N
    s_t = np.vectorize(lambda cn: compute_stdv(zd, cn, N))(c_n)
    zeta = N * s_t / c_t
    zf, zp = zflower(zeta, sbs), zpaper(zeta, sbs)
    z_eq = (np.mean(zf ** -2.0, axis=1)) ** -0.5
    zq = (np.mean(zpaper(zeta, sbs) ** -2.0, axis=1)) ** -0.5
    n_rounds = r_t.size
    cell = {"variant": v[0], "sigma_b": sbs, "lr": lr, "c0": c0, "z_delta": zd, "trajectories": len(trajs),
            "rounds_per_trajectory": rounds}
    rows: list[dict] = []
    ident_err = float(np.max(np.abs(r_t - np.exp(-lr * (ft - Q))) / r_t))
    zeta_attr_err = float(np.max(np.abs(zeta - zd * r_t) / zeta))
    rows.append(row(
        "G2a", v, {"r_equals_exp_minus_lr_times_f_minus_q": True, "zeta_equals_z_delta_times_r": True},
        {"rho_delta": 1.0},
        {"max_relative_error_r": ident_err, "max_relative_error_zeta": zeta_attr_err,
         "rho_delta_mean": float(r_t.mean()), "rho_delta_min": float(r_t.min()),
         "rho_delta_max": float(r_t.max())},
        ident_err <= 1e-12 and zeta_attr_err <= 1e-12, bool(np.max(np.abs(r_t - 1.0)) <= 1e-9),
        "1e-12 relative", sigma_b=sbs, lr=lr, c0=c0))
    lo = math.exp(-lr * (1 + 5 * sbs / N - Q))
    hi = math.exp(lr * (Q + 5 * sbs / N))
    in_band = bool(r_t.min() >= lo and r_t.max() <= hi)
    rows.append(row(
        "G2c", v, {"band": [lo, hi]}, {"rho_delta": 1.0},
        {"min": float(r_t.min()), "max": float(r_t.max())}, in_band,
        bool(np.max(np.abs(r_t - 1.0)) <= 1e-9), "inside band", sigma_b=sbs, lr=lr, c0=c0))
    pred_r = math.exp(-lr * (1 - Q) + lr ** 2 * (sbs / N) ** 2 / 2)
    pred_zeta = zd * pred_r
    pred_zf, pred_zp = zflower(pred_zeta, sbs), zpaper(pred_zeta, sbs)
    se = lambda x: float(np.std(x, ddof=1) / math.sqrt(x.size))  # noqa: E731
    if c0 == 0.1:
        all_clipped = bool(np.all(loc == N))
        m_r = float(r_t.mean())
        held = all_clipped and abs(m_r - pred_r) <= 4 * se(r_t) + 1e-12
        rows.append(row(
            "G2b", v, {"all_clipped": True, "rho_delta_mean": pred_r},
            {"rho_delta_mean": 1.0}, {"all_clipped": all_clipped, "rho_delta_mean": m_r,
                                      "standard_error": se(r_t)},
            held, abs(m_r - 1.0) <= 4 * se(r_t), "4 standard errors", sigma_b=sbs, lr=lr, c0=c0))
        m_zf, m_zp = float(zf.mean() / Z), float(zp.mean() / Z)
        tol_f = max(0.01 * pred_zf / Z, 4 * se(zf / Z))
        tol_p = max(0.01 * pred_zp / Z, 4 * se(zp / Z))
        held = abs(m_zf - pred_zf / Z) <= tol_f and abs(m_zp - pred_zp / Z) <= tol_p
        rows.append(row(
            "G3a", v, {"flower_ratio": pred_zf / Z, "paper_count_ratio": pred_zp / Z},
            {"ratio": 1.0},
            {"flower_ratio": m_zf, "paper_count_ratio": m_zp,
             "tolerance_flower": tol_f, "tolerance_paper": tol_p},
            held, abs(m_zp - 1.0) <= 0.01, "4 standard errors or 1 percent relative, larger",
            sigma_b=sbs, lr=lr, c0=c0))
    strict = bool(np.all(zf < zp) and np.all(zf < zeta))
    rows.append(row(
        "G3b", v, {"z_flower_below_z_paper_and_below_zeta_every_round": True}, "not specified",
        {"holds_every_round": strict, "rounds": int(n_rounds)}, strict, None, "strict",
        sigma_b=sbs, lr=lr, c0=c0))
    r1_paper = zpaper(zd, sbs) / Z
    r1_flower = zflower(zd, sbs) / Z
    pred_r1 = (1 + 0.75 * (Z / sbs) ** 2) ** -0.5
    held = close(r1_paper, 1.0) and close(r1_flower, pred_r1)
    rows.append(row(
        "G3c", v, {"paper_count_ratio_at_r1": 1.0, "flower_ratio_at_r1": pred_r1}, {"ratio": 1.0},
        {"paper_count_ratio_at_r1": r1_paper, "flower_ratio_at_r1": r1_flower},
        held, close(r1_paper, 1.0), "1e-9 relative", sigma_b=sbs, lr=lr, c0=c0))
    stats = lambda x: {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}  # noqa: E731
    cell.update({
        "rho_delta": stats(r_t),
        "rho_z": stats(zeta / Z),
        "z_flower_over_z": stats(zf / Z),
        "z_paper_count_over_z": stats(zp / Z),
        "z_eq_flower_over_z": {"mean": float(z_eq.mean() / Z), "min": float(z_eq.min() / Z)},
        "z_eq_paper_count_over_z": {"mean": float(zq.mean() / Z), "min": float(zq.min() / Z)},
        "fraction_clipped_noisy_mean": float(ft.mean()),
        "final_clip_over_start": {"min": float((C[:, -1] / c0).min()), "median": float(np.median(C[:, -1] / c0)),
                                  "max": float((C[:, -1] / c0).max())},
        "labels": {
            "rho_delta_mean": label(float(r_t.mean())),
            "z_flower_over_z_mean": label(float(zf.mean() / Z)),
            "z_paper_count_over_z_mean": label(float(zp.mean() / Z)),
        },
    })
    pooled = {"ss": float(ss.sum()), "values": int(ss.size * (DIM - 1)),
              "alt_sum_r_inv2": float((r_t ** -2.0).sum()), "alt_rounds": int(r_t.size)}
    return rows, cell, pooled


def pooled_rows(v, pooled_by: dict) -> list[dict]:
    rows = []
    for lr in LRS:
        entries = {}
        for key, cells in (("all_cells", pooled_by[lr]["all"]), ("start_0.1_cells", pooled_by[lr]["c01"])):
            vals = sum(c["values"] for c in cells)
            rms = math.sqrt(sum(c["ss"] for c in cells) / vals)
            alt = math.sqrt(sum(c["alt_sum_r_inv2"] for c in cells) / sum(c["alt_rounds"] for c in cells))
            entries[key] = {"rms": rms, "values": vals, "standard_error": 1 / math.sqrt(2 * vals),
                            "rms_if_noise_used_old_clip": alt}
        ok = (abs(entries["all_cells"]["rms"] - 1) <= 0.005
              and abs(entries["start_0.1_cells"]["rms"] - 1) <= 0.007)
        rows.append(row(
            "G4", v, {"rms_all_cells": 1.0, "tolerance_all": 0.005, "rms_start_0.1_cells": 1.0,
                      "tolerance_start_0.1": 0.007},
            {"noise_scaled_with": "clip of the round"}, entries, ok, False,
            "0.5 percent over all cells and 0.7 percent over the start 0.1 cells", lr=lr))
    return rows


# ------------------------------------------------------------------ G5 (fixed)
def run_fixed(v, z: float, clip: float, seed: int, rounds: int) -> dict:
    r = Runner(v, z, N, clip)
    np.random.seed(seed)
    ss, unchanged, counts, zeta_err = 0.0, True, 0, 0.0
    ss_rounds = []
    for _ in range(rounds):
        before = r.strategy.clipping_norm
        norms = NORMS if v[2] == "server" else [min(u, before) for u in NORMS]
        with DrawLog() as log:
            out = r.run(norms)
        after = r.strategy.clipping_norm
        unchanged &= after == before
        counts += len(log.count_draws())
        s = compute_stdv(r.strategy.noise_multiplier, after, N)
        zeta = N * s / before
        zeta_err = max(zeta_err, abs(zeta / z - 1.0))
        scaled = (out[1:] / (z * before / N)) ** 2
        ss_rounds.append(float(scaled.sum()))
    return {"ss": float(sum(ss_rounds)), "values": rounds * (DIM - 1), "clip_unchanged": unchanged,
            "count_draws": counts, "zeta_ratio_max_error": zeta_err, "ss_rounds": ss_rounds}


def rows_g5(v, rounds: int, reps: int) -> tuple[list[dict], dict]:
    rows, total_ss, total_n = [], 0.0, 0
    per_cell = {}
    for a, z in enumerate(FIXED_Z):
        for b, clip in enumerate(FIXED_CLIPS):
            res = [run_fixed(v, z, clip, 20261301 + 1000 * a + 100 * b + j, rounds) for j in range(reps)]
            ok = (all(x["clip_unchanged"] for x in res) and sum(x["count_draws"] for x in res) == 0
                  and max(x["zeta_ratio_max_error"] for x in res) <= 1e-12)
            total_ss += sum(x["ss"] for x in res)
            total_n += sum(x["values"] for x in res)
            per_cell[(z, clip)] = [x["ss_rounds"] for x in res]
            rows.append(row(
                "G5a", v, {"clip_unchanged": True, "count_draws": 0, "zeta_over_z": 1.0}, {"zeta_over_z": 1.0},
                {"clip_unchanged": all(x["clip_unchanged"] for x in res),
                 "count_draws": sum(x["count_draws"] for x in res),
                 "zeta_over_z_max_error": max(x["zeta_ratio_max_error"] for x in res),
                 "composed_over_z": 1.0, "label": label(1.0)},
                ok, True, "exact clip and draws, 1e-12 relative for the ratio", z=z, clip=clip))
    rms = math.sqrt(total_ss / total_n)
    rows.append(row(
        "G5b", v, {"pooled_rms": 1.0, "tolerance": 0.015}, {"pooled_rms": 1.0},
        {"pooled_rms": rms, "values": total_n, "standard_error": 1 / math.sqrt(2 * total_n)},
        abs(rms - 1.0) <= 0.015, abs(rms - 1.0) <= 0.015, "1.5 percent"))
    return rows, per_cell


# ------------------------------------------------------------------ G6
def rows_g6(v) -> list[dict]:
    inputs = [(z, k * z) for z in (0.5, 1.0, 2.0, 4.0) for k in (0.6, 2.5, 5.0, 10.0)]
    inputs += [(z, 5.0) for z in (0.5, 1.0, 2.0, 4.0, 8.0)]
    cases, ok = [], True
    for z, sb in inputs:
        cls = strategy_class(v)
        s = cls(inner_strategy(v[1], N), z, N, clipped_count_stddev=sb)
        zd, sbs = float(s.noise_multiplier), float(s.clipped_count_stddev)
        zp, zf = zpaper(zd, sbs) / z, zflower(zd, sbs) / z
        pred = (1 + 0.75 * (z / sbs) ** 2) ** -0.5
        this = close(zp, 1.0) and close(zf, pred) and zf < 1.0
        ok &= this
        cases.append({"z": z, "sigma_b": sbs, "z_delta": zd, "z_paper_over_z": zp, "z_flower_over_z": zf,
                      "predicted_flower": pred, "label_flower": label(zf)})
    return [row("G6", v, {"z_paper_over_z": 1.0, "z_flower_over_z": "(1 + 0.75 (z / sigma_b)^2)^(-1/2) and below 1"},
                {"z_over_z": 1.0}, {"cases": cases, "count": len(cases)}, ok, True, "1e-9 relative")]


# ------------------------------------------------------------------ main
def run_all(args) -> int:
    rounds = 10 if args.quick else 30
    reps = 2 if args.quick else 10
    f_rounds = 5 if args.quick else 10
    f_reps = 2 if args.quick else 4
    logging.getLogger("flwr").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    rows: list[dict] = []
    cells: list[dict] = []
    raw: dict = {}
    seq_by_cell: dict = {}
    for v in ADAPTIVE:
        rows += rows_g0(v)
        rows += rows_g1(v)
        rows += rows_g6(v)
        pooled_by = {lr: {"all": [], "c01": []} for lr in LRS}
        for a, sb in enumerate(SIGMAS):
            for b, lr in enumerate(LRS):
                for c, c0 in enumerate(STARTS):
                    trajs = [run_trajectory(v, sb, lr, c0, cell_seed(a, b, c, j), rounds) for j in range(reps)]
                    r_rows, cell, pooled = analyse_cell(v, sb, lr, c0, trajs, rounds)
                    rows += r_rows
                    cells.append(cell)
                    pooled_by[lr]["all"].append(pooled)
                    if c0 == 0.1:
                        pooled_by[lr]["c01"].append(pooled)
                    key = f"{sb}|{lr}|{c0}"
                    seq_by_cell.setdefault(key, {})[v[0]] = [t["C"] for t in trajs]
                    raw.setdefault(v[0], {})[key] = {
                        "C": [t["C"] for t in trajs], "count_location": [t["loc"] for t in trajs],
                        "noisy_count": [t["ret"] for t in trajs]}
        rows += pooled_rows(v, pooled_by)
        print(f"{v[0]} done {time.time() - t0:.0f}s", flush=True)

    # G7 across variants
    for key, per in seq_by_cell.items():
        ref = np.array(per["V1"])
        for name in ("V2", "V3", "V4"):
            cur = np.array(per[name])
            err = float(np.max(np.abs(cur - ref) / np.abs(ref)))
            sb, lr, c0 = key.split("|")
            rows.append(row("G7", next(x for x in ADAPTIVE if x[0] == name), {"agrees_with_V1_within": 1e-9},
                            "not specified", {"max_relative_difference_to_V1": err},
                            err <= 1e-9, None, "1e-9 relative", sigma_b=float(sb), lr=float(lr), c0=float(c0)))

    fixed_ss = {}
    for v in FIXED:
        rows += rows_g1(v)
        g5, per_cell = rows_g5(v, f_rounds, f_reps)
        rows += g5
        fixed_ss[v[0]] = per_cell
        print(f"{v[0]} done {time.time() - t0:.0f}s", flush=True)
    for key in fixed_ss["F1"]:
        ref = np.array(fixed_ss["F1"][key])
        for name in ("F2", "F3", "F4"):
            cur = np.array(fixed_ss[name][key])
            err = float(np.max(np.abs(cur - ref) / ref))
            rows.append(row("G7", next(x for x in FIXED if x[0] == name), {"agrees_with_F1_within": 1e-6},
                            "not specified", {"max_relative_difference_to_F1": err},
                            err <= 1e-6, None, "1e-6 relative", z=key[0], clip=key[1]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "feddisclosure.flower_adaptive_guarantee.v1",
        "quick": args.quick,
        "environment": {"flwr": flwr.__version__, "flwr_file": flwr.__file__, "numpy": np.__version__,
                        "python": platform.python_version()},
        "parameters": {"dim": DIM, "n": N, "z": Z, "q": Q, "sigma_b": SIGMAS, "lr": LRS, "start_clips": STARTS,
                       "trajectories": reps, "rounds": rounds, "fixed_z": FIXED_Z, "fixed_clips": FIXED_CLIPS,
                       "fixed_trajectories": f_reps, "fixed_rounds": f_rounds, "norms": "100 values 0.5 to 1.5",
                       "seed_rule": "20261201 + 100000 a + 10000 b + 1000 c + j",
                       "fixed_seed_rule": "20261301 + 1000 a + 100 b + j", "rel_tol": REL},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "cells": cells,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "flower_adaptive_guarantee.json").write_text(json.dumps(doc, indent=2) + "\n")
    with gzip.open(out / "trajectories.json.gz", "wt") as fh:
        json.dump({"flwr": flwr.__version__, "note": "per cell: clip sequences, count locations, noisy counts",
                   "data": raw}, fh)
    failed = [(r["row"], r["variant"], r.get("sigma_b"), r.get("lr"), r.get("c0")) for r in rows
              if not r["prediction_held"]]
    print(f"wrote {out / 'flower_adaptive_guarantee.json'} in {doc['runtime_seconds']}s, {len(rows)} rows")
    print("predictions not held:", failed or "none")
    return 0


def compare(args) -> int:
    docs = []
    for d in args.compare:
        with gzip.open(Path(d) / "trajectories.json.gz", "rt") as fh:
            docs.append(json.load(fh))
    a, b = docs
    worst, cells = 0.0, 0
    for var, per in a["data"].items():
        for key, rec in per.items():
            x, y = np.array(rec["C"]), np.array(b["data"][var][key]["C"])
            worst = max(worst, float(np.max(np.abs(x - y) / np.abs(x))))
            cells += 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"schema": "feddisclosure.flower_adaptive_guarantee.compare.v1", "versions": [d["flwr"] for d in docs],
           "cells_compared": cells, "max_relative_difference_of_clip_sequences": worst,
           "prediction_held": worst <= 1e-9, "tolerance": "1e-9 relative", "row": "G7",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "cross_version.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0 if res["prediction_held"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="small run for tests")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    args = ap.parse_args()
    return compare(args) if args.compare else run_all(args)


if __name__ == "__main__":
    sys.exit(main())
