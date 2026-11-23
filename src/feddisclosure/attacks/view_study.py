"""Protected versus unprotected observation views for update inversion.

What a server (or an attacker next to it) can see differs by protection:
  plaintext        the victim's individual update
  dp_individual    the victim's update after Opacus DP-SGD clipping and noise
  secagg           only the aggregate of all clients, produced with the UPSTREAM Flower SecAgg+
                   quantiser (stochastic rounding, clipping range, weight ratio, dequantisation)
  secagg_collude   the server plus n-1 colluding clients subtract their own updates from the
                   aggregate and so obtain the victim's update up to quantisation error
  dp_secagg        aggregate of DP-noised updates
  dp_secagg_collude  collusion on DP-noised updates (recovers the noised victim update)
The attack is the repository's DLG-style gradient matching with known labels (a favourable
privilege for the attacker). All views use the same data, the same initial model and one local
SGD step, so differences come from the protection and not from the setup.
This is an observation-model study with real quantisation arithmetic. It is not an attack run on
captured Docker network traffic.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feddisclosure.attacks.gradient_inversion import reconstruct
from feddisclosure.evaluation.gradient import score_reconstruction
from feddisclosure.training.task import DigitMLP

CLIP_RANGE = 8.0          # upstream SecAggPlusWorkflow default clipping_range
QUANT_RANGE = 2 ** 22     # upstream default quantization_range


def flatten(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().astype(np.float64).ravel() for v in state.values()])


def unflatten(vec: np.ndarray, like: dict) -> dict:
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = torch.tensor(vec[i:i + n].reshape(tuple(v.shape)), dtype=v.dtype)
        i += n
    return out


def client_step(init: dict, x: np.ndarray, y: np.ndarray, *, lr: float, sigma: float | None, clip: float = 1.0,
                seed: int = 0) -> dict:
    """One local full-batch step. sigma=None is plain SGD. Otherwise genuine Opacus DP-SGD
    (per-sample clipping to `clip` and Gaussian noise sigma*clip) on the same batch."""
    torch.manual_seed(seed)
    model = DigitMLP(64, 10)
    model.load_state_dict(init)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    xs, ys = torch.from_numpy(x).float(), torch.from_numpy(y).long()
    if sigma is None:
        opt.zero_grad()
        nn.functional.cross_entropy(model(xs), ys).backward()
        opt.step()
    else:
        from opacus import PrivacyEngine
        dl = DataLoader(TensorDataset(xs, ys), batch_size=len(ys), shuffle=False)
        pm, po, pdl = PrivacyEngine(secure_mode=False).make_private(
            module=model, optimizer=opt, data_loader=dl, noise_multiplier=sigma, max_grad_norm=clip,
            poisson_sampling=False)
        for bx, by in pdl:
            po.zero_grad()
            nn.functional.cross_entropy(pm(bx), by).backward()
            po.step()
        model = pm._module
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def secagg_aggregate(vectors: list[np.ndarray], sizes: list[int], *, max_weight: float = 1000.0,
                     rng_seed: int = 0) -> np.ndarray:
    """Weighted mean as the server recovers it from upstream SecAgg+ (masks cancel, so only the
    quantisation arithmetic of secaggplus_mod and the unmask stage remains)."""
    from flwr.common.secure_aggregation.quantization import dequantize, quantize
    np.random.seed(rng_seed)    # seeds the stochastic rounding
    total_q, summed = 0, None
    for v, n in zip(vectors, sizes):
        q_ratio = round(n / max_weight * QUANT_RANGE)
        dq_ratio = q_ratio / QUANT_RANGE
        q = quantize([v * dq_ratio], CLIP_RANGE, QUANT_RANGE)[0].astype(np.int64)
        summed = q if summed is None else summed + q
        total_q += q_ratio
    agg = dequantize([summed], CLIP_RANGE, QUANT_RANGE)[0]
    agg = (agg - (len(vectors) - 1) * CLIP_RANGE) * (QUANT_RANGE / total_q)
    return agg


def attack(init: dict, observed_delta: dict, labels: np.ndarray, observer: str, *, lr: float,
           iterations: int, seed: int) -> np.ndarray:
    obs = {"schema_version": 1, "observer": observer, "batch_size": int(len(labels)), "features": 64,
           "classes": 10, "model_state": init, "model_delta": observed_delta, "learning_rate": lr,
           "local_steps": 1, "known_labels": labels.tolist()}
    rec, _ = reconstruct(obs, iterations=iterations, seed=seed)
    return rec


def score_victim(originals: np.ndarray, recovered: np.ndarray, victim_count: int) -> dict:
    """Optimal matching of ALL original records to the recovered set, then report the victim rows.
    Evaluator privilege only. The first `victim_count` originals are the victim's."""
    if recovered.shape[0] == originals.shape[0]:
        full = score_reconstruction(originals, recovered)
        mse = np.array(full["per_image_mse"])[:victim_count]
    else:                                            # individual view: only the victim's batch
        full = score_reconstruction(originals[:victim_count], recovered)
        mse = np.array(full["per_image_mse"])
    return {"victim_mse": float(mse.mean()), "victim_rate_mse_below_0_05": float((mse < 0.05).mean())}


def chance_floor(originals_victim: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> dict:
    """What a reconstruction with no information achieves: the mean public digit, and random public digits."""
    mean_img = np.repeat(pool.mean(axis=0, keepdims=True), len(originals_victim), axis=0)
    rand = pool[rng.choice(len(pool), len(originals_victim), replace=False)]
    return {"mean_image_mse": score_reconstruction(originals_victim, mean_img)["mean_mse"],
            "random_digits_mse": score_reconstruction(originals_victim, rand)["mean_mse"]}


def class_average_images(pool_x: np.ndarray, pool_y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Mean public image of every class. A class absent from the pool is an error, not a silent zero."""
    out = np.empty((n_classes, pool_x.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        members = pool_x[pool_y == c]
        if len(members) == 0:
            raise ValueError(f"public pool has no record of class {c}")
        out[c] = members.mean(axis=0)
    return out


def identification_rate(originals_victim: np.ndarray, reconstructions: np.ndarray, victim_labels: np.ndarray,
                        pool_x: np.ndarray, pool_y: np.ndarray, *, u: int = 10, repeats: int = 200,
                        rng: np.random.Generator | None = None) -> float:
    """How often the true record is the closest one to its reconstruction among u same-label candidates.

    For every victim record the candidate set is the true record plus u - 1 public records of the
    same label drawn without replacement. The candidate nearest to the reconstruction in squared
    error is the guess. A tie shares the credit equally. Averaged over victim records and `repeats`
    random candidate sets. A reconstruction that carries no record-specific information scores
    about 1 / u, because the true record is then exchangeable with the candidates. Caveat: when the
    reconstruction is built from the same pool the candidates come from (as the class average is in
    `label_floor`), every candidate contributed to it, which tilts the result slightly against the
    true record. With about 170 same-label pool records the tilt is small.

    Not comparable to the MSE based numbers. MSE measures how close a reconstruction is to the
    record. This measures how well the reconstruction separates the record from look-alikes of the
    same class, so it answers a different question and the two must not be put in one ranking.
    """
    if u < 2:
        raise ValueError("u must be at least 2")
    rng = rng if rng is not None else np.random.default_rng(0)
    scores = []
    for xi, ri, label in zip(originals_victim, reconstructions, victim_labels):
        same = pool_x[pool_y == label]
        if len(same) < u - 1:
            raise ValueError("public pool has too few same-label records for the requested u")
        d_true = float(((xi - ri) ** 2).mean())
        for _ in range(repeats):
            cand = same[rng.choice(len(same), u - 1, replace=False)]
            d_other = ((cand - ri) ** 2).mean(axis=1)
            wins, ties = float((d_other > d_true).all()), int((d_other == d_true).sum())
            scores.append(wins if ties == 0 else float((d_other >= d_true).all()) / (ties + 1))
    return float(np.mean(scores))


def label_floor(originals_victim: np.ndarray, victim_labels: np.ndarray, pool_x: np.ndarray,
                pool_y: np.ndarray, *, u: int = 10, repeats: int = 200,
                rng: np.random.Generator | None = None) -> dict:
    """Baseline for an attacker who knows the victim's labels: rebuild every victim record as the
    class-average image of the public pool. The pool must not contain the victim records.

    Returns the mean per-image MSE and the share of records below 0.05 (the same definition of
    recognisable as `score_victim`), both without matching because the label already says which
    class image belongs to which record, and the same two numbers under optimal matching, which is
    how the attack views are scored. Also returns `identification_rate`, which is NOT comparable
    to MSE (see `identification_rate`).
    """
    cm = class_average_images(pool_x, pool_y)
    recon = cm[np.asarray(victim_labels, dtype=int)]
    per_image = ((originals_victim - recon) ** 2).mean(axis=1)
    matched = score_reconstruction(originals_victim, recon)
    return {"class_average_mse": float(per_image.mean()),
            "class_average_rate_mse_below_0_05": float((per_image < 0.05).mean()),
            "per_image_mse": per_image.tolist(),
            "matched_mse": float(matched["mean_mse"]),
            "matched_rate_mse_below_0_05": float(matched["reconstruction_rate_mse_below_0_05"]),
            "identification_rate": identification_rate(originals_victim, recon, victim_labels, pool_x, pool_y,
                                                       u=u, repeats=repeats, rng=rng),
            "identification_u": int(u),
            "identification_chance": 1.0 / u}
