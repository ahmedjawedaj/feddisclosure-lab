"""Protected vs unprotected update-inversion comparison (see feddisclosure.attacks.view_study).

Seeds. The saved v0.14.2 results used the per-client seed `seed * 10 + client`. That scheme
collides from 11 clients upward, so new runs should use `feddisclosure.evaluation.intervals.client_seed`.
`--legacy-seeds` (default on) keeps the old scheme so that the saved v0.14.2 results reproduce
exactly. Pass `--no-legacy-seeds` for new runs. That mode refuses to write into the saved v0.14.2
directory and defaults to its own output directory.

Intervals. Rates are reported with a pooled Wilson interval over all victim images. Continuous
metrics keep the t-interval over seeds. See `scripts/recompute_view_intervals.py` for the full
statistics on the saved v0.14.2 runs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from feddisclosure.attacks.view_seed import client_step_seed, select_view_records
from feddisclosure.attacks.view_study import (attack, chance_floor, client_step, flatten, label_floor, score_victim,
                                               secagg_aggregate, unflatten)
from feddisclosure.evaluation.intervals import wilson_interval
from feddisclosure.training.datasets import _source
from feddisclosure.training.task import DigitMLP

ROOT = Path(__file__).resolve().parents[1]
LR, B, N = 0.15, 8, 4
V0142_DIR = ROOT / "reports/results/privacy/v014/view_comparison"
NEW_SEEDS_DIR = ROOT / "reports/results/privacy/v0143/view_comparison_client_seed"


def single_step_epsilon(sigma: float, delta: float = 1e-5) -> float:
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant(); acc.history = [(sigma, 1.0, 1)]
    return float(acc.get_epsilon(delta))


def one_seed(seed: int, sigmas, max_weights, iterations: int, legacy_seeds: bool = True) -> dict:
    torch.set_num_threads(1)
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    sel = select_view_records(len(y), N, B, seed)
    rng, idx, pool_idx = sel.rng, sel.idx, sel.pool_idx
    cx, cy = x[idx].reshape(N, B, 64), y[idx].reshape(N, B)          # client 0 is the victim
    torch.manual_seed(seed); init = {k: v.clone() for k, v in DigitMLP(64, 10).state_dict().items()}
    sizes = [B] * N
    victim_x = cx[0]; all_x = cx.reshape(-1, 64); all_y = cy.reshape(-1)
    out = {"seed": seed, "floor": chance_floor(victim_x, x[pool_idx], rng), "views": {}}
    lf = label_floor(victim_x, cy[0], x[pool_idx], y[pool_idx], rng=np.random.default_rng(5000 + seed))
    out["label_floor"] = {k: lf[k] for k in ("class_average_mse", "class_average_rate_mse_below_0_05", "matched_mse",
                                             "matched_rate_mse_below_0_05", "identification_rate")}

    def delta_of(state): return {k: init[k] - state[k] for k in init}

    def run(name, observer, delta, labels, truth_x, victim_rows):
        t = time.perf_counter()
        rec = attack(init, delta, labels, observer, lr=LR, iterations=iterations, seed=seed + 1)
        out["views"][name] = {**score_victim(truth_x, rec, victim_rows), "seconds": round(time.perf_counter() - t, 1)}

    def build(sigma):
        return [client_step(init, cx[i], cy[i], lr=LR, sigma=sigma, seed=client_step_seed(seed, i, legacy=legacy_seeds)) for i in range(N)]

    def aggregate(states, mw):
        vecs = [flatten(s) for s in states]
        return unflatten(secagg_aggregate(vecs, sizes, max_weight=mw, rng_seed=seed), init)

    def collude(states, mw):
        agg = flatten(aggregate(states, mw)); total = sum(sizes)
        others = sum(flatten(states[j]) * sizes[j] / total for j in range(1, N))
        return unflatten((agg - others) * total / sizes[0], init)

    plain = build(None)
    run("plaintext_individual", "server_individual_model_delta", delta_of(plain[0]), cy[0], victim_x, B)
    for mw in max_weights:
        run(f"secagg_aggregate_maxw{mw:g}", "server_aggregate_model_delta_joint_oracle",
            delta_of(aggregate(plain, mw)), all_y, all_x, B)
        run(f"secagg_collusion_n-1_maxw{mw:g}", "server_collusion_isolated_delta",
            delta_of(collude(plain, mw)), cy[0], victim_x, B)
    mw = max_weights[-1]
    for s in sigmas:
        dp = build(s)
        run(f"dp{s:g}_individual", "server_individual_model_delta", delta_of(dp[0]), cy[0], victim_x, B)
        run(f"dp{s:g}_secagg_aggregate_maxw{mw:g}", "server_aggregate_model_delta_joint_oracle",
            delta_of(aggregate(dp, mw)), all_y, all_x, B)
        run(f"dp{s:g}_secagg_collusion_maxw{mw:g}", "server_collusion_isolated_delta",
            delta_of(collude(dp, mw)), cy[0], victim_x, B)
    return out


def _wilson_pooled(rates, records: int = B):
    """[mean rate, low, high] with a Wilson interval pooled over every victim image of every seed."""
    r = np.asarray(rates, float)
    k, n = int(round(r.sum() * records)), int(len(r) * records)
    lo, hi = wilson_interval(k, n)
    return [k / n, lo, hi]


def _ci(v):
    from scipy import stats
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.2, 0.5, 1.0])
    p.add_argument("--max-weights", nargs="+", type=float, default=[1000.0, 16.0])
    p.add_argument("--legacy-seeds", action=argparse.BooleanOptionalAction, default=True,
                   help="per-client seed seed*10+client (reproduces saved v0.14.2). --no-legacy-seeds uses intervals.client_seed")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--overwrite-summary", action="store_true", help="allow replacing the saved v0.14.2 summary.json")
    a = p.parse_args()
    if a.output is None:
        a.output = V0142_DIR if a.legacy_seeds else NEW_SEEDS_DIR
    if not a.legacy_seeds and a.output.resolve() == V0142_DIR.resolve():
        p.error("--no-legacy-seeds must not write into the saved v0.14.2 directory")
    if a.output.resolve() == V0142_DIR.resolve() and (a.output / "summary.json").is_file() and not a.overwrite_summary:
        p.error("saved v0.14.2 summary.json exists, pass --overwrite-summary to replace it (the saved evidence is kept by default)")
    a.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for s in range(a.seeds):
        f = a.output / f"seed{s}.json"
        if f.is_file():
            runs.append(json.loads(f.read_text())); continue
        r = one_seed(s, a.sigmas, a.max_weights, a.iterations, a.legacy_seeds)
        f.write_text(json.dumps(r, indent=1)); runs.append(r); print("seed", s, "done", flush=True)
    names = list(runs[0]["views"])
    plain = np.array([r["views"]["plaintext_individual"]["victim_mse"] for r in runs])
    summary = {"schema_version": 1, "clients": N, "records_per_client": B, "seeds": a.seeds,
               "lr": LR, "attack_iterations": a.iterations, "labels_known_to_attacker": True,
               "legacy_seeds": a.legacy_seeds,
               "dp": {"mechanism": "Opacus DP-SGD single step, per-sample clip 1.0",
                      "single_step_epsilon_delta_1e-5_no_subsampling": {str(s): single_step_epsilon(s) for s in a.sigmas}},
               "secagg": "upstream Flower SecAgg+ quantiser arithmetic, clipping_range 8, quantization_range 2**22",
               "floor": {k: _ci([r["floor"][k] for r in runs]) for k in runs[0]["floor"]}, "views": {}}
    for n in names:
        mse = np.array([r["views"][n]["victim_mse"] for r in runs])
        rate = np.array([r["views"][n]["victim_rate_mse_below_0_05"] for r in runs])
        summary["views"][n] = {"victim_mse": _ci(mse), "rate_mse_below_0_05": _wilson_pooled(rate),
                               "paired_mse_minus_plaintext": _ci(mse - plain)}
    summary["limitations"] = [
        "Observation-model study with real quantiser arithmetic, not an attack on captured Docker traffic",
        "One local SGD step on 8 distinct records per client, known labels, known lr and initial model",
        "DP noise is Opacus single-step DP-SGD, the attacker matches naively without modelling clipping or noise",
        "A failed reconstruction is evidence about this attack only, not a privacy guarantee",
        "Rate intervals are pooled Wilson intervals over all victim images. MSE intervals are t-intervals over seeds"]
    if "label_floor" in runs[0]:
        summary["label_floor"] = {k: float(np.mean([r["label_floor"][k] for r in runs])) for k in runs[0]["label_floor"]}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for n, v in summary["views"].items():
        print("%-40s mse %.4f  rate %.2f" % (n, v["victim_mse"][0], v["rate_mse_below_0_05"][0]))
    print("floor", summary["floor"])


if __name__ == "__main__":
    main()
