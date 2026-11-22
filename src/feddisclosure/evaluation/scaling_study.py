"""Aggregate and collusion views at 4 and 8 clients (pre-registered rows S1 and S2).

The protocol is the v0.14.2 view study (`feddisclosure.attacks.view_study`) with the number of
clients as a parameter. Every client holds 8 distinct public digits and takes one full-batch SGD
step from a shared initial model. Client 0 is the victim. The attacker knows the labels and the
learning rate. Three views are scored against the victim records:

  plaintext_individual     the victim update itself (upper bound for any protected view)
  secagg_aggregate         only the SecAgg+ aggregate, produced with the upstream quantiser
  secagg_collusion_n-1     the aggregate minus the updates of the n-1 colluding clients

The label-only floor rebuilds every victim record as the class-average image of the public pool.
"Recognisable" means per-image MSE below 0.05.

The victim records are the first 8 entries of the seeded permutation whatever the number of
clients, so a seed has the same victim images at 4 and 8 clients. Only the other clients and the
size of the public pool change.

Rates use Wilson intervals pooled over all victim images. Images of one seed share an initial model
and a data draw, so the seed-level percentile bootstrap is reported next to it.
"""
from __future__ import annotations

import numpy as np
import torch

from feddisclosure.attacks.view_seed import client_step_seed, select_view_records
from feddisclosure.attacks.view_study import (attack, class_average_images, client_step, flatten, secagg_aggregate,
                                               unflatten)
from feddisclosure.evaluation.gradient import score_reconstruction
from feddisclosure.evaluation.intervals import wilson_interval
from feddisclosure.training.task import DigitMLP

LR = 0.15
RECORDS_PER_CLIENT = 8
RECOGNISABLE = 0.05
DEFAULT_MAX_WEIGHT = 1000.0     # upstream SecAggPlusWorkflow default, the primary setting
BOOT_RESAMPLES = 4000
BOOT_SEED = 9100                # a tuning-range seed, used only for the bootstrap resampling


def victim_per_image_mse(originals: np.ndarray, recovered: np.ndarray, victim_count: int) -> np.ndarray:
    """Per-image MSE of the victim rows. Optimal matching is evaluator privilege, as in `score_victim`."""
    if recovered.shape[0] == originals.shape[0]:
        return np.asarray(score_reconstruction(originals, recovered)["per_image_mse"])[:victim_count]
    return np.asarray(score_reconstruction(originals[:victim_count], recovered)["per_image_mse"])


def collusion_view(aggregate: np.ndarray, plain_states: list[np.ndarray], sizes: list[int]) -> np.ndarray:
    """What the server and clients 1..n-1 can compute for client 0.

    The colluders know their own plain updates, so they remove their weighted share from the
    aggregate and rescale by the total weight over the victim weight.
    """
    if len(plain_states) != len(sizes) or len(sizes) < 2:
        raise ValueError("need one state per client and at least two clients")
    total = float(sum(sizes))
    others = sum(plain_states[j] * sizes[j] / total for j in range(1, len(sizes)))
    return (aggregate - others) * total / sizes[0]


def run_seed(x: np.ndarray, y: np.ndarray, seed: int, n_clients: int, *, iterations: int = 300,
             max_weights=(DEFAULT_MAX_WEIGHT,), legacy_seeds: bool = False,
             records_per_client: int = RECORDS_PER_CLIENT) -> dict:
    """One evaluation seed at one client count. Returns per-image victim MSE for every view."""
    torch.set_num_threads(1)
    b, n = records_per_client, n_clients
    sel = select_view_records(len(y), n, b, seed)
    idx, pool_idx = sel.idx, sel.pool_idx
    cx, cy = x[idx].reshape(n, b, -1), y[idx].reshape(n, b)
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in DigitMLP(cx.shape[2], 10).state_dict().items()}
    sizes = [b] * n
    victim_x, all_x, all_y = cx[0], cx.reshape(-1, cx.shape[2]), cy.reshape(-1)
    pool_x, pool_y = x[pool_idx], y[pool_idx]
    floor_recon = class_average_images(pool_x, pool_y)[cy[0].astype(int)]
    floor_mse = ((victim_x - floor_recon) ** 2).mean(axis=1)
    mean_mse = ((victim_x - pool_x.mean(axis=0)) ** 2).mean(axis=1)
    out = {"seed": seed, "clients": n, "records_per_client": b, "legacy_seeds": legacy_seeds,
           "public_pool_size": int(len(pool_idx)),
           "label_floor_per_image_mse": floor_mse.tolist(), "mean_image_per_image_mse": mean_mse.tolist(),
           "views": {}}

    def delta_of(state):
        return {k: init[k] - state[k] for k in init}

    def view(name, observer, delta, labels, truth_x):
        rec = attack(init, delta, labels, observer, lr=LR, iterations=iterations, seed=seed + 1)
        out["views"][name] = victim_per_image_mse(truth_x, rec, b).tolist()

    states = [client_step(init, cx[i], cy[i], lr=LR, sigma=None,
                          seed=client_step_seed(seed, i, legacy=legacy_seeds)) for i in range(n)]
    vecs = [flatten(s) for s in states]
    view("plaintext_individual", "server_individual_model_delta", delta_of(states[0]), cy[0], victim_x)
    for mw in max_weights:
        agg = secagg_aggregate(vecs, sizes, max_weight=mw, rng_seed=seed)
        agg_state = unflatten(agg, init)        # float32 parameters, as the server holds them
        view(f"secagg_aggregate_maxw{mw:g}", "server_aggregate_model_delta_joint_oracle",
             delta_of(agg_state), all_y, all_x)
        view(f"secagg_collusion_n-1_maxw{mw:g}", "server_collusion_isolated_delta",
             delta_of(unflatten(collusion_view(flatten(agg_state), vecs, sizes), init)), cy[0], victim_x)
    return out


# ---- statistics ---------------------------------------------------------------------------------

def bootstrap_ci(values, resamples: int = BOOT_RESAMPLES, seed: int = BOOT_SEED, confidence: float = 0.95):
    """Percentile bootstrap of the mean over seeds. Returns [mean, low, high]."""
    v = np.asarray(values, float)
    if v.ndim != 1 or len(v) < 2 or resamples < 2000:
        raise ValueError("need a 1-D sample of at least 2 values and at least 2000 resamples")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    a = (1 - confidence) / 2
    return [float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def pooled_rate(per_seed_mse: list[list[float]]) -> dict:
    """Recognisable rate over every victim image of every seed, with a Wilson interval and a
    seed-level bootstrap interval."""
    arr = np.asarray(per_seed_mse, float)
    flags = arr < RECOGNISABLE
    k, total = int(flags.sum()), int(flags.size)
    lo, hi = wilson_interval(k, total)
    return {"successes": k, "images": total, "rate": k / total, "wilson": [lo, hi],
            "seed_bootstrap": bootstrap_ci(flags.mean(axis=1)), "mean_mse": float(arr.mean()),
            "mse_seed_bootstrap": bootstrap_ci(arr.mean(axis=1))}


def summarise(runs: list[dict]) -> dict:
    """One client count. `runs` are `run_seed` outputs for the same n_clients."""
    ns = {r["clients"] for r in runs}
    if len(ns) != 1:
        raise ValueError("summarise expects runs from one client count")
    out = {"clients": ns.pop(), "seeds": len(runs),
           "label_floor": pooled_rate([r["label_floor_per_image_mse"] for r in runs]),
           "mean_image_floor": pooled_rate([r["mean_image_per_image_mse"] for r in runs]),
           "views": {name: pooled_rate([r["views"][name] for r in runs]) for name in runs[0]["views"]}}
    floor_seed = np.array([np.mean(np.asarray(r["label_floor_per_image_mse"]) < RECOGNISABLE) for r in runs])
    for name in out["views"]:
        rate_seed = np.array([np.mean(np.asarray(r["views"][name]) < RECOGNISABLE) for r in runs])
        out["views"][name]["rate_minus_floor"] = out["views"][name]["rate"] - out["label_floor"]["rate"]
        out["views"][name]["paired_rate_minus_floor_bootstrap"] = bootstrap_ci(rate_seed - floor_seed)
    return out


def check_s1(summary8: dict, view: str) -> dict:
    """S1: the aggregate view at 8 clients is at most floor plus 5 points (point estimates)."""
    v, floor = summary8["views"][view], summary8["label_floor"]
    gap = v["rate"] - floor["rate"]
    return {"id": "S1", "prediction": "At 8 clients the aggregate view stays at the floor",
            "pass_rule": "Recognisable rate at most floor plus 5 points",
            "observed": {"view": view, "aggregate_rate": v["rate"], "aggregate_wilson": v["wilson"],
                         "floor_rate": floor["rate"], "floor_wilson": floor["wilson"], "gap": gap,
                         "aggregate_mean_mse": v["mean_mse"], "floor_mean_mse": floor["mean_mse"],
                         "gap_upper_limit_is_conservative_check":
                             v["wilson"][1] - floor["rate"] <= 0.05},
            "pass": bool(gap <= 0.05 + 1e-9)}   # tolerance only absorbs float rounding at the boundary


def check_s2(summaries: dict[int, dict], view: str) -> dict:
    """S2: n-1 collusion recognisable rate is at least 85% at every listed client count."""
    rows = {str(n): {"rate": s["views"][view]["rate"], "wilson": s["views"][view]["wilson"],
                     "wilson_lower_at_least_0_85": s["views"][view]["wilson"][0] >= 0.85}
            for n, s in sorted(summaries.items())}
    return {"id": "S2", "prediction": "n-1 collusion stays at plaintext level at 4 and 8 clients",
            "pass_rule": "Recognisable rate at least 85%", "observed": {"view": view, "by_clients": rows},
            "pass": bool(all(r["rate"] >= 0.85 - 1e-9 for r in rows.values()))}
