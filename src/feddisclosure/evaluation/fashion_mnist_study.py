"""Ordering study on a second dataset: Fashion-MNIST with a small CNN (pre-registered row X1).

Same protocol as the digits view study (`feddisclosure.evaluation.scaling_study`): 4 clients with 8
distinct records each, one full-batch SGD step from a shared initial model, the attacker knows the
labels, the learning rate and the initial model, client 0 is the victim. Views:

  plaintext_individual     the victim update itself
  secagg_collusion_n-1     the SecAgg+ aggregate minus the other clients' updates
  secagg_aggregate         only the SecAgg+ aggregate (upstream quantiser arithmetic), joint oracle labels
  label floor              every victim record rebuilt as the class-average image of the public pool

Recognisable means per-image MSE below 0.05 on pixels scaled to [0, 1]. The attack is the repository's
gradient matching, generalised from the 8x8 MLP to the CNN. It receives no ground truth.
Attack settings (iterations, step size, total-variation weight) are chosen on tuning seeds 9000 and up.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from feddisclosure.attacks.view_seed import select_view_records
from feddisclosure.attacks.view_study import class_average_images, flatten, secagg_aggregate, unflatten
from feddisclosure.evaluation.scaling_study import RECOGNISABLE, collusion_view

SIDE = 28
LR = 0.15
RECORDS_PER_CLIENT = 8
N_CLIENTS = 4
DEFAULT_MAX_WEIGHT = 1000.0
# Fixed before any evaluation run, see docs/v0.15-fashion-mnist.md.
FLOOR_PLUS_PLAINTEXT = 0.10
FLOOR_PLUS_COLLUSION = 0.10
FLOOR_PLUS_AGGREGATE = 0.05


class SmallCNN(nn.Module):
    """Two 3x3 convolutions with max pooling and one linear layer, about 9,000 parameters."""

    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(16 * 7 * 7, classes)

    def forward(self, x):
        x = x.reshape(-1, 1, SIDE, SIDE)
        return self.classifier(self.features(x).flatten(1))


def client_step(init: dict, x: np.ndarray, y: np.ndarray, *, lr: float = LR) -> dict:
    """One plain full-batch SGD step. Returns the new state dict."""
    model = SmallCNN()
    model.load_state_dict(init)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt.zero_grad()
    nn.functional.cross_entropy(model(torch.from_numpy(x).float()), torch.from_numpy(y).long()).backward()
    opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def invert(init: dict, observed_delta: dict, labels: np.ndarray, *, lr: float = LR, iterations: int = 300,
           attack_lr: float = 0.1, tv_weight: float = 0.0005, seed: int = 0) -> np.ndarray:
    """Gradient matching. Inputs are only what the observer holds: the initial model, the observed model
    delta, the labels and the learning rate. Exact for one SGD step."""
    torch.set_num_threads(1)
    model = SmallCNN()
    model.load_state_dict(init)
    model.eval()
    names = list(dict(model.named_parameters()))
    targets = [(observed_delta[k] / lr).detach() for k in names]
    y = torch.as_tensor(labels, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    dummy = (torch.rand((len(y), SIDE * SIDE), generator=gen) * 0.6 + 0.2).requires_grad_(True)
    opt = torch.optim.Adam([dummy], lr=attack_lr)
    for _ in range(iterations):
        opt.zero_grad()
        sim = torch.autograd.grad(nn.functional.cross_entropy(model(dummy), y), tuple(model.parameters()),
                                  create_graph=True)
        match = sum((a - b).square().sum() for a, b in zip(sim, targets))
        tiles = dummy.reshape(-1, SIDE, SIDE)
        tv = (tiles[:, 1:, :] - tiles[:, :-1, :]).abs().mean() + (tiles[:, :, 1:] - tiles[:, :, :-1]).abs().mean()
        (match + tv_weight * tv).backward()
        opt.step()
        with torch.no_grad():
            dummy.clamp_(0, 1)
    return dummy.detach().numpy()


def victim_per_image_mse(originals: np.ndarray, recovered: np.ndarray, victim_count: int) -> np.ndarray:
    """Per-image MSE of the victim rows under optimal matching (evaluator privilege, as in the digits study)."""
    if recovered.shape[0] != originals.shape[0]:
        originals = originals[:victim_count]
    d = ((originals[:, None, :] - recovered[None, :, :]) ** 2).mean(axis=2)
    rows, cols = linear_sum_assignment(d)
    return d[rows, cols][:victim_count]


PRIMARY_MAX_WEIGHT = DEFAULT_MAX_WEIGHT


def view_names(max_weights=(PRIMARY_MAX_WEIGHT,)) -> list[str]:
    return ["plaintext_individual"] + [f"{kind}_maxw{mw:g}" for mw in max_weights
                                       for kind in ("secagg_collusion", "secagg_aggregate")]


def run_seed(x: np.ndarray, y: np.ndarray, seed: int, *, settings: dict, n_clients: int = N_CLIENTS,
             max_weights=(PRIMARY_MAX_WEIGHT,), records_per_client: int = RECORDS_PER_CLIENT,
             only: tuple[str, ...] | None = None) -> dict:
    """One seed. `settings` maps 'individual', 'collusion' and 'aggregate' to invert() keyword arguments.
    `only` restricts the views (used by tuning)."""
    torch.set_num_threads(1)
    b, n = records_per_client, n_clients
    sel = select_view_records(len(y), n, b, seed)
    cx, cy = x[sel.idx].reshape(n, b, -1), y[sel.idx].reshape(n, b)
    pool_x, pool_y = x[sel.pool_idx], y[sel.pool_idx]
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in SmallCNN().state_dict().items()}
    victim_x, all_x, all_y = cx[0], cx.reshape(-1, cx.shape[2]), cy.reshape(-1)
    floor_mse = ((victim_x - class_average_images(pool_x, pool_y)[cy[0].astype(int)]) ** 2).mean(axis=1)
    mean_mse = ((victim_x - pool_x.mean(axis=0)) ** 2).mean(axis=1)
    out = {"seed": seed, "clients": n, "records_per_client": b, "public_pool_size": int(len(sel.pool_idx)),
           "label_floor_per_image_mse": floor_mse.tolist(), "mean_image_per_image_mse": mean_mse.tolist(),
           "settings": settings, "views": {}}
    states = [client_step(init, cx[i], cy[i]) for i in range(n)]
    vecs, sizes = [flatten(s) for s in states], [b] * n

    def delta_of(state):
        return {k: init[k] - state[k] for k in init}

    def want(name):
        return only is None or name in only

    if want("plaintext_individual"):
        rec = invert(init, delta_of(states[0]), cy[0], seed=seed + 1, **settings["individual"])
        out["views"]["plaintext_individual"] = victim_per_image_mse(victim_x, rec, b).tolist()
    for mw in max_weights:
        col_name, agg_name = f"secagg_collusion_maxw{mw:g}", f"secagg_aggregate_maxw{mw:g}"
        if not (want(col_name) or want(agg_name)):
            continue
        agg_state = unflatten(secagg_aggregate(vecs, sizes, max_weight=mw, rng_seed=seed), init)
        if want(col_name):
            col = unflatten(collusion_view(flatten(agg_state), vecs, sizes), init)
            rec = invert(init, delta_of(col), cy[0], seed=seed + 1, **settings["collusion"])
            out["views"][col_name] = victim_per_image_mse(victim_x, rec, b).tolist()
        if want(agg_name):
            rec = invert(init, delta_of(agg_state), all_y, seed=seed + 1, **settings["aggregate"])
            out["views"][agg_name] = victim_per_image_mse(all_x, rec, b).tolist()
    return out


def check_x1(summary: dict, max_weight: float = PRIMARY_MAX_WEIGHT) -> dict:
    """X1 on point estimates, like S1 and S2, for one SecAgg+ max weight. Every part must hold."""
    floor = summary["label_floor"]["rate"]
    v = summary["views"]
    col, agg = f"secagg_collusion_maxw{max_weight:g}", f"secagg_aggregate_maxw{max_weight:g}"
    parts = {
        "plaintext_at_least_floor_plus_10": v["plaintext_individual"]["rate"] >= floor + FLOOR_PLUS_PLAINTEXT - 1e-9,
        "collusion_at_least_floor_plus_10": v[col]["rate"] >= floor + FLOOR_PLUS_COLLUSION - 1e-9,
        "aggregate_within_floor_plus_5": v[agg]["rate"] <= floor + FLOOR_PLUS_AGGREGATE + 1e-9}
    return {"id": "X1", "prediction": "The second dataset (Fashion-MNIST, small CNN) shows the same ordering",
            "pass_rule": "Plaintext and n-1 collusion at least floor plus 10 points, aggregate-only within floor plus 5 points",
            "observed": {"floor_rate": floor, "floor_wilson": summary["label_floor"]["wilson"],
                         "max_weight": max_weight,
                         **{k: {"rate": v[k]["rate"], "wilson": v[k]["wilson"], "rate_minus_floor": v[k]["rate"] - floor}
                            for k in ("plaintext_individual", col, agg)}},
            "parts": {k: bool(p) for k, p in parts.items()}, "pass": bool(all(parts.values()))}


# ---- tuning (tuning seeds only) -----------------------------------------------------------------

def tune_one(args) -> dict:
    """Mean victim MSE of one attack setting on tuning seeds. `view` is 'individual', 'collusion' or
    'aggregate' (SecAgg+ max weight 1000). Only seeds of 9000 and above are accepted, so tuning cannot
    touch the evaluation seeds."""
    from feddisclosure.evaluation.intervals import TUNING_SEED_BASE
    x, y, view, setting, seeds = args
    if any(s < TUNING_SEED_BASE for s in seeds):
        raise ValueError("tuning must use tuning seeds (9000 and up)")
    name = {"individual": "plaintext_individual", "collusion": f"secagg_collusion_maxw{PRIMARY_MAX_WEIGHT:g}",
            "aggregate": f"secagg_aggregate_maxw{PRIMARY_MAX_WEIGHT:g}"}[view]
    mses = []
    for s in seeds:
        r = run_seed(x, y, s, settings={"individual": setting, "collusion": setting, "aggregate": setting}, only=(name,))
        mses += r["views"][name]
    mses = np.asarray(mses)
    return {"view": view, "setting": setting, "seeds": list(seeds), "mean_mse": float(mses.mean()),
            "rate": float((mses < RECOGNISABLE).mean())}


def best_setting(rows: list[dict]) -> dict:
    """Lowest mean MSE wins. Ties go to the earlier row."""
    return min(rows, key=lambda r: r["mean_mse"])
