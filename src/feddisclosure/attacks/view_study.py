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
    np.random.seed(rng_seed)    # upstream stochastic rounding uses numpy's global RNG
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
