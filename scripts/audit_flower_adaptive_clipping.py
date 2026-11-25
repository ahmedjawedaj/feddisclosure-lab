#!/usr/bin/env python3
"""Measurement of Flower's adaptive clipping against Andrew et al. 2021 (v0.16 T8b).

Runs the four public adaptive clipping strategy classes in process with a
FedAvg inner strategy and fake client replies. The rows, predictions and
tolerances are fixed in docs/preregistration/v0.16-adaptive-clipping.md,
which was committed before this script existed.

Usage: python scripts/audit_flower_adaptive_clipping.py --out DIR [--quick]
"""
from __future__ import annotations

import argparse
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
    )
except ImportError:
    from flwr.common.differential_privacy_constants import (
        KEY_CLIPPING_NORM,
        KEY_NORM_BIT,
    )
try:
    from flwr.common import recorddict_compat as compat
except ImportError:
    from flwr.compat.common import recorddict_compat as compat

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from flwr.client.mod import adaptiveclipping_mod as legacy_mod
from flwr.clientapp.mod import adaptiveclipping_mod as message_mod
from flwr.server import strategy as legacy_strategy
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.serverapp import strategy as message_strategy

DIM = 16
OK = Status(code=Code.OK, message="")
CTX = Context(run_id=0, node_id=0, node_config={}, state=RecordDict(), run_config={})
VARIANTS = [
    ("V1", "legacy", "server"),
    ("V2", "legacy", "client"),
    ("V3", "message", "server"),
    ("V4", "message", "client"),
]
REL = 1e-9


# ---------------------------------------------------------------- helpers
def close(a: float, b: float, rel: float = REL) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


def counted_fraction(c0: float, c1: float, lr: float = 0.2, q: float = 0.5) -> float:
    return q - math.log(c1 / c0) / lr


def geo(c: float, frac: float, lr: float = 0.2, q: float = 0.5) -> float:
    return c * math.exp(-lr * (frac - q))


def e1() -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def strategy_class(api: str, side: str):
    mod = legacy_strategy if api == "legacy" else message_strategy
    name = (
        "DifferentialPrivacyServerSideAdaptiveClipping"
        if side == "server"
        else "DifferentialPrivacyClientSideAdaptiveClipping"
    )
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


def construct(variant: tuple, z: float, n: int, **kw):
    """Build one strategy object with a FedAvg inner strategy."""
    _, api, side = variant
    cls = strategy_class(api, side)
    if api == "legacy":
        inner = legacy_strategy.FedAvg(
            inplace=True, fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=n, min_available_clients=n,
        )
    else:
        inner = CaptureFedAvg(fraction_train=1.0, fraction_evaluate=0.0)
    return cls(inner, z, n, **kw)


class Runner:
    """Drive one strategy for single rounds with a chosen set of client replies."""

    def __init__(self, variant: tuple, z: float, n: int, c0: float = 0.1, **kw) -> None:
        self.name, self.api, self.side = variant
        self.n = n
        self.strategy = construct(variant, z, n, initial_clipping_norm=c0, **kw)
        self.g = np.zeros(DIM, dtype=np.float32)
        if self.api == "legacy":
            self.manager = SimpleClientManager()
            self.proxies = [StubProxy(str(i)) for i in range(n)]
            for p in self.proxies:
                self.manager.register(p)

    def set_clip(self, c: float) -> None:
        self.strategy.clipping_norm = c

    def run(self, norms, bits: int | None = None, mod: bool = False):
        """One round. norms has one entry per reply. Client side needs bits or mod."""
        if self.api == "legacy":
            return self._run_legacy(norms, bits, mod)
        return self._run_message(norms, bits, mod)

    # ------------------------------------------------------------ legacy
    def _legacy_through_mod(self, fit_ins, upd):
        msg = Message(
            compat.fitins_to_recorddict(fit_ins, True), dst_node_id=1, message_type="train"
        )

        def app(m, _ctxt):
            res = FitRes(OK, ndarrays_to_parameters([self.g + upd]), 1, {})
            return Message(compat.fitres_to_recorddict(res, True), reply_to=m)

        out = legacy_mod(msg, CTX, app)
        return compat.recorddict_to_fitres(out.content, True)

    def _run_legacy(self, norms, bits, mod):
        s = self.strategy
        ins = s.configure_fit(1, ndarrays_to_parameters([self.g]), self.manager)
        results = []
        for i, u in enumerate(norms):
            upd = (u * e1()).astype(np.float32)
            proxy, fit_ins = ins[i]
            if self.side == "server":
                res = FitRes(OK, ndarrays_to_parameters([self.g + upd]), 1, {})
            elif mod:
                res = self._legacy_through_mod(fit_ins, upd)
            else:
                res = FitRes(
                    OK, ndarrays_to_parameters([self.g + upd]), 1,
                    {KEY_NORM_BIT: int(i < bits)},
                )
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
        if self.side == "server":
            s.current_arrays = arrays
        else:
            cfg = self._message_config()
        replies = []
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

                replies.append(message_mod(ask, CTX, app))
                continue
            metrics = {"num-examples": 1}
            if self.side == "client":
                metrics[KEY_NORM_BIT] = int(i < bits)
            content = RecordDict({
                "arrays": ArrayRecord([self.g + upd]),
                "metrics": MetricRecord(metrics),
            })
            replies.append(Message(content, reply_to=ask))
        out, _ = s.aggregate_train(1, replies)
        if out is None:
            return None
        return out.to_numpy_ndarrays()[0].astype(np.float64)


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


# ------------------------------------------------------------------ rows
def row_a0(v) -> list[dict]:
    cls = strategy_class(v[1], v[2])
    sig = inspect.signature(cls.__init__).parameters
    defaults = {
        k: sig[k].default
        for k in (
            "initial_clipping_norm", "target_clipped_quantile", "clip_norm_lr",
            "clipped_count_stddev",
        )
    }
    stored = construct(v, 1.0, 100).clipped_count_stddev
    pred = {
        "initial_clipping_norm": 0.1, "target_clipped_quantile": 0.5,
        "clip_norm_lr": 0.2, "clipped_count_stddev": None, "stored_stddev_m100": 5.0,
    }
    meas = dict(defaults, stored_stddev_m100=stored)
    paper = {
        "initial_clipping_norm": 0.1, "target_clipped_quantile": 0.5,
        "clip_norm_lr": 0.2, "stored_stddev_m100": 5.0,
    }
    agrees = all(meas[k] == paper[k] for k in paper)
    return [row("A0", v, pred, paper, meas, meas == pred, agrees, "exact")]


def row_a1(v) -> list[dict]:
    out = []

    def build(z, n, sb):
        s = construct(v, z, n, clipped_count_stddev=sb)
        return float(s.clipped_count_stddev), float(s.noise_multiplier)

    def zd(z, sb):
        return (z ** -2 - (2 * sb) ** -2) ** -0.5

    for name, sb, exp_sb, exp_m in (
        ("A1a", None, 5.0, 1.005037815259212),
        ("A1b", 10.0, 10.0, 1.0012523486435176),
    ):
        got = build(1.0, 100, sb)
        held = abs(got[0] - exp_sb) <= 1e-9 and abs(got[1] - exp_m) <= 1e-9
        paper_m = zd(1.0, exp_sb)
        out.append(row(
            name, v, {"stddev": exp_sb, "multiplier": exp_m}, {"multiplier": paper_m},
            {"stddev": got[0], "multiplier": got[1],
             "configured": {"z": 1.0, "m": 100, "clipped_count_stddev": sb}},
            held, abs(got[1] - paper_m) <= 1e-9, "1e-9 absolute",
        ))

    def raises(z, n, sb):
        try:
            construct(v, z, n, clipped_count_stddev=sb)
            return None
        except Exception as exc:  # recorded, not hidden
            return type(exc).__name__

    for name, z, n, sb in (
        ("A1c", 1.0, 100, 0.5), ("A1c2", 10.0, 100, None),
    ):
        got = raises(z, n, sb)
        out.append(row(
            name, v, "ValueError", "invalid input", {"exception": got,
            "configured": {"z": z, "m": n, "clipped_count_stddev": sb}},
            got == "ValueError", got is not None, "exception type",
        ))

    z, exp = 9.99, (9.99 ** -2 - 10.0 ** -2) ** -0.5
    try:
        got = build(z, 100, None)
        err = None
    except Exception as exc:
        got, err = (None, None), type(exc).__name__
    held = err is None and abs(got[1] - exp) <= 1e-9 * exp
    out.append(row(
        "A1c3", v, {"multiplier": exp}, {"multiplier": exp},
        {"multiplier": got[1], "exception": err,
         "configured": {"z": z, "m": 100, "clipped_count_stddev": None}},
        held, held, "1e-9 relative",
    ))

    grid = []
    for z in (0.1, 0.5, 1.0, 2.0, 5.0):
        for k in (0.6, 5.0):
            grid.append((z, z * k))
    grid += [(1.0, 0.51), (1.0, 100.0)]
    cases = []
    for z, sb in grid:
        got = build(z, 100, sb)
        cases.append({"z": z, "sigma_b": sb, "multiplier": got[1],
                      "at_least_z": got[1] >= z * (1 - 1e-12)})
    ok = all(c["at_least_z"] for c in cases)
    out.append(row("A1d", v, "stored multiplier at least z in every case",
                   "z_delta at least z", {"cases": cases, "count": len(cases)},
                   ok, ok, "inequality"))

    e = []
    s = construct(v, 0.0, 100)
    e.append({"input": "z 0", "stddev": s.clipped_count_stddev, "multiplier": s.noise_multiplier})
    s = construct(v, 0.0, 100, clipped_count_stddev=3.0)
    e.append({"input": "z 0, stddev 3", "stddev": s.clipped_count_stddev,
              "multiplier": s.noise_multiplier})
    e.append({"input": "z 1, stddev 0", "exception": raises(1.0, 100, 0.0)})
    ok = (
        e[0]["stddev"] == 0.0 and e[0]["multiplier"] == 0.0
        and e[1]["stddev"] == 3.0 and e[1]["multiplier"] == 0.0
        and e[2]["exception"] == "ValueError"
    )
    out.append(row("A1e", v, "(0.0, 0.0), (3.0, 0.0), ValueError", "not specified",
                   {"cases": e}, ok, None, "exact"))
    return out


def row_a2(v) -> list[dict]:
    cases = []
    for z, m, sb in ((1.0, 100, None), (1.0, 100, 10.0), (0.5, 10, None), (0.3, 200, 2.0)):
        s = construct(v, z, m, clipped_count_stddev=sb)
        zd, sbs = s.noise_multiplier, s.clipped_count_stddev
        back = (zd ** -2 + (2 * sbs) ** -2) ** -0.5
        cases.append({"z": z, "m": m, "sigma_b": sbs, "stored_multiplier": zd,
                      "recovered_z": back, "abs_error": abs(back - z)})
    ok = all(c["abs_error"] <= 1e-9 for c in cases)
    return [row("A2", v, "recovered z equals configured z", "recovered z equals z",
                {"cases": cases}, ok, ok, "1e-9 absolute")]


def mixed(n_in: int, n_out: int, inside: float, outside: float) -> list[float]:
    return [inside] * n_in + [outside] * n_out


def row_a3(v) -> list[dict]:
    c0 = 0.1
    scen = {0.0: (0.11051709180756478, 0.09048374180359596),
            0.5: (0.1, 0.1),
            1.0: (0.09048374180359596, 0.11051709180756478)}
    meas, ok_pred, ok_paper = [], True, True
    for u, (paper, pred) in scen.items():
        k = int(round(100 * u))
        r = Runner(v, 0.0, 100, c0)
        r.run(mixed(k, 100 - k, 0.5 * c0, 2 * c0), mod=True)
        c1 = r.strategy.clipping_norm
        meas.append({"u": u, "next_clip": c1, "paper": paper, "prediction": pred})
        ok_pred &= close(c1, pred)
        ok_paper &= close(c1, paper)
    return [row("A3", v, {str(u): p[1] for u, p in scen.items()},
                {str(u): p[0] for u, p in scen.items()}, {"scenarios": meas},
                ok_pred, ok_paper, "1e-9 relative")]


def row_a4(v) -> list[dict]:
    c0 = 0.1
    r = Runner(v, 0.0, 100, c0)
    r.run(mixed(70, 30, 0.5 * c0, 2 * c0), mod=True)
    c1 = r.strategy.clipping_norm
    f = counted_fraction(c0, c1)
    out = [row(
        "A4", v, {"fraction": 0.30, "next_clip": 0.10408107741923883},
        {"fraction": 0.70, "next_clip": 0.09607894391523232},
        {"fraction": f, "next_clip": c1, "inside_clip": 70, "clipped": 30},
        abs(f - 0.30) <= 1e-9, abs(f - 0.70) <= 1e-9, "1e-9 absolute",
    )]
    r = Runner(v, 0.0, 1, 1.0)
    r.run([1.0], mod=True)
    c1 = r.strategy.clipping_norm
    pred, paper = 1.1051709180756477, 0.9048374180359596
    out.append(row(
        "A4b", v, {"next_clip": pred, "clipped": False},
        {"next_clip": paper, "bit": 1}, {"next_clip": c1,
        "counted_fraction": counted_fraction(1.0, c1)},
        close(c1, pred), close(c1, paper), "1e-9 relative",
    ))
    return out


def row_a5(v, rounds: int) -> list[dict]:
    out = []
    c0 = 0.1
    norms = mixed(70, 30, 0.5 * c0, 2 * c0)
    for sb, sb_val in ((None, 5.0), (10.0, 10.0)):
        r = Runner(v, 1.0, 100, c0, clipped_count_stddev=sb)
        np.random.seed(20261125)
        fr = np.empty(rounds)
        for i in range(rounds):
            r.set_clip(c0)
            r.run(norms, bits=30)
            fr[i] = counted_fraction(c0, r.strategy.clipping_norm)
        sd, mean = float(fr.std(ddof=1)), float(fr.mean())
        exp_sd = sb_val / 100
        se = sd / math.sqrt(rounds)
        held = abs(sd / exp_sd - 1) <= 0.03 and abs(mean - 0.30) <= 4 * se
        out.append(row(
            "A5", v, {"std": exp_sd, "mean": 0.30}, {"std": exp_sd},
            {"std": sd, "mean": mean, "std_error_of_mean": se, "rounds": rounds,
             "ratio_to_prediction": sd / exp_sd, "clipped_count_stddev": sb_val},
            held, abs(sd / exp_sd - 1) <= 0.03, "std within 3 percent, mean within 4 SE",
        ))
    return out


def noise_vs_scale(r, quiet, norms, rounds, c_old, scale_fn, **run_kw):
    """Standardised model noise per round and the clip before and after."""
    base = quiet_output(quiet, c_old, norms, **run_kw)
    old, new = [], []
    z_over = []
    for _ in range(rounds):
        r.set_clip(c_old)
        out = r.run(norms, **run_kw)
        c_new = r.strategy.clipping_norm
        noise = out - base
        z_over.append(noise)
        new.append(c_new)
    return np.stack(z_over), np.array(new)


def quiet_output(quiet, c0, norms, **kw):
    quiet.set_clip(c0)
    return quiet.run(norms, **kw)


def row_a6(v, rounds: int) -> list[dict]:
    z, n, lr, c0 = 0.5, 10, 1.0, 1.0
    r = Runner(v, z, n, c0, clip_norm_lr=lr)
    quiet = Runner(v, 0.0, n, c0, clip_norm_lr=lr)
    zd = float(r.strategy.noise_multiplier)
    norms = [5.0] * n if v[2] == "server" else [1.0] * n
    kw = {} if v[2] == "server" else {"bits": n}
    np.random.seed(20261126)
    noise, c_new = noise_vs_scale(r, quiet, norms, rounds, c0, None, **kw)
    z_new = noise / (zd * c_new[:, None] / n)
    z_old = noise / (zd * c0 / n)
    sd_new = float(np.sqrt((z_new ** 2).mean()))
    sd_old = float(np.sqrt((z_old ** 2).mean()))
    held = abs(sd_new - 1) <= 0.02 and (sd_old < 0.9 or sd_old > 1.1)
    agrees = abs(sd_old - 1) <= 0.02
    return [row(
        "A6", v,
        {"std_by_old_clip": "about 0.61, outside 0.9 to 1.1", "std_by_new_clip": 1.0},
        {"std_by_old_clip": 1.0, "std_by_new_clip": "about 1.65"},
        {"std_by_old_clip": sd_old, "std_by_new_clip": sd_new, "rounds": rounds,
         "mean_new_over_old_clip": float(c_new.mean() / c0), "z_delta": zd,
         "noise_uses": "new clip" if abs(sd_new - 1) <= 0.02 else
         ("old clip" if abs(sd_old - 1) <= 0.02 else "neither")},
        held, agrees, "std_by_new_clip within 0.02 of 1, std_by_old_clip outside 0.9 to 1.1",
    )]


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[tuple[str, str]] = []

    def emit(self, record) -> None:
        self.records.append((record.levelname, record.getMessage()))


def row_a7(v, rounds: int) -> list[dict]:
    n, c0 = 10, 1.0
    if v[2] == "server":
        norms, kw = mixed(2, 3, 0.5, 2.0), {}
    else:
        norms, kw = [0.5] * 5, {"bits": 3}
    # fraction part
    r = Runner(v, 0.0, n, c0)
    r.run(norms, **kw)
    c1 = r.strategy.clipping_norm
    f = counted_fraction(c0, c1)
    pred_c, alt_c = 0.9801986733067553, 1.0408107741923882
    frac_ok = close(c1, pred_c)
    # noise part
    z = 0.5
    r = Runner(v, z, n, c0)
    quiet = Runner(v, 0.0, n, c0)
    zd = float(r.strategy.noise_multiplier)
    np.random.seed(20261127)
    noise, c_new = noise_vs_scale(r, quiet, norms, rounds, c0, None, **kw)
    sd_n = float(np.sqrt(((noise / (zd * c_new[:, None] / n)) ** 2).mean()))
    sd_len = float(np.sqrt(((noise / (zd * c_new[:, None] / 5)) ** 2).mean()))
    noise_ok = abs(sd_n - 1) <= 0.02
    # warning part
    handler = ListHandler()
    lg = logging.getLogger("flwr")
    old_level = lg.level
    lg.setLevel(logging.WARNING)
    lg.addHandler(handler)
    try:
        r.set_clip(c0)
        out = r.run(norms, **kw)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)
    warned = [m for lvl, m in handler.records if "sampled clients" in m]
    warn_ok = bool(warned) and out is not None
    return [row(
        "A7", v,
        {"counted_fraction": 0.6, "next_clip": pred_c, "std_by_num_sampled": 1.0,
         "warning_logged": True, "aggregates": True},
        "not specified, the paper fixes m",
        {"counted_fraction": f, "next_clip": c1, "next_clip_if_divided_by_n": alt_c,
         "std_by_num_sampled": sd_n, "std_by_reply_count": sd_len, "rounds": rounds,
         "warning_messages": warned, "aggregate_returned": out is not None},
        frac_ok and noise_ok and warn_ok, None,
        "next_clip 1e-9 relative, std within 0.02, warning present",
    )]


def row_a8(v) -> list[dict]:
    cls = strategy_class(v[1], v[2])
    params = list(inspect.signature(cls.__init__).parameters)
    seedlike = [p for p in params if any(t in p.lower() for t in ("seed", "rng", "generator", "random"))]
    # The pre-registration named n = 10 with z = 1, which Flower rejects at
    # construction (z equals 2 times the default sigma_b). n = 20 is used. See
    # the deviation log.
    n, c0 = 20, 1.0
    norms = [5.0] * n if v[2] == "server" else [1.0] * n
    kw = {} if v[2] == "server" else {"bits": 8}

    def once(seed):
        r = Runner(v, 1.0, n, c0)
        np.random.seed(seed)
        state0 = np.random.get_state()[1].copy()
        out = r.run(norms, **kw)
        state1 = np.random.get_state()[1]
        return r.strategy.clipping_norm, out, bool((state0 != state1).any())

    a, b, c = once(123), once(123), once(124)
    same = a[0] == b[0] and bool(np.array_equal(a[1], b[1]))
    diff = a[0] != c[0] and not np.array_equal(a[1], c[1])
    held = not seedlike and same and diff and a[2]
    return [row(
        "A8", v,
        {"seed_parameter": False, "same_seed_identical": True, "different_seed_differs": True,
         "global_state_advances": True},
        "not specified",
        {"constructor_parameters": params, "seed_like_parameters": seedlike,
         "same_seed_identical": same, "different_seed_differs": diff,
         "global_state_advances": a[2], "clip_seed123": a[0], "clip_seed123_again": b[0],
         "clip_seed124": c[0]},
        held, None, "exact equality, then inequality",
    )]


def row_a10(v) -> list[dict]:
    c0, rounds = 0.1, 23
    r = Runner(v, 0.0, 100, c0)
    factors = []
    for _ in range(rounds):
        before = r.strategy.clipping_norm
        r.run([1.0] * 100, mod=True)
        factors.append(r.strategy.clipping_norm / before)
    c = r.strategy.clipping_norm
    pred, paper = 0.010025884372280375, 0.9974182454814718
    in_band = all(0.9048 <= f <= 1.1052 for f in factors)
    return [row(
        "A10", v, {"clip_after_23": pred, "per_round_factor_band": [0.9048, 1.1052]},
        {"clip_after_23": paper}, {"clip_after_23": c, "min_factor": min(factors),
        "max_factor": max(factors)},
        close(c, pred) and in_band, close(c, paper), "1e-9 relative",
    )]


def row_a11(v) -> list[dict]:
    norms = list(np.linspace(0.5, 1.5, 100))
    out = []
    for c0, pred in ((0.1, 2.061153622438558e-10), (10.0, 4851651954.097902)):
        r = Runner(v, 0.0, 100, c0)
        for _ in range(200):
            r.run(norms, mod=True)
        c = r.strategy.clipping_norm
        out.append(row(
            "A11", v, {"clip_after_200": pred}, {"clip_after_200": "within 10 percent of 1.0"},
            {"clip_after_200": c, "start": c0}, close(c, pred), abs(c - 1.0) <= 0.1,
            "1e-9 relative",
        ))
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="small run for tests")
    args = ap.parse_args()
    r5 = 500 if args.quick else 10_000
    r67 = 1_000 if args.quick else 20_000

    logging.getLogger("flwr").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    rows: list[dict] = []
    for v in VARIANTS:
        for fn in (row_a0, row_a1, row_a2, row_a3, row_a4):
            rows.extend(fn(v))
        rows.extend(row_a5(v, r5))
        rows.extend(row_a6(v, r67))
        rows.extend(row_a7(v, r67))
        rows.extend(row_a8(v))
        rows.extend(row_a10(v))
        rows.extend(row_a11(v))
        print(f"{v[0]} done {time.time() - t0:.0f}s", flush=True)

    for v in VARIANTS:
        if v[2] != "client":
            continue
        mine = [r for r in rows if r["variant"] == v[0]]
        rows.append({
            "row": "A9", "variant": v[0], "api": v[1], "side": v[2],
            "note": "client-side summary, one entry per row of this variant",
            "prediction_held_by_row": {
                f"{r['row']}:{i}": r["prediction_held"] for i, r in enumerate(mine)
            },
            "prediction_held": all(r["prediction_held"] for r in mine),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "feddisclosure.flower_adaptive_clipping.v1",
        "quick": args.quick,
        "environment": {
            "flwr": flwr.__version__,
            "flwr_file": flwr.__file__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "parameters": {"dim": DIM, "a5_rounds": r5, "a6_a7_rounds": r67, "rel_tol": REL},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "flower_adaptive_clipping.json").write_text(json.dumps(doc, indent=2) + "\n")
    failed = [(r["row"], r["variant"]) for r in rows if not r["prediction_held"]]
    print(f"wrote {out / 'flower_adaptive_clipping.json'} in {doc['runtime_seconds']}s")
    print("predictions not held:", failed or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
