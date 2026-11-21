"""Shadow-model likelihood-ratio membership inference (LiRA-style) against federated models.

Threat model (strong, stated): the attacker knows the training pipeline, hyperparameters,
the privacy mechanism and its epsilon, can train its own shadow federations on data drawn
from the same pool, and sees the FINAL GLOBAL MODEL's output on candidate records.
It does not see client updates. This is an empirical audit, not a privacy guarantee:
a low attack success does not prove privacy and the epsilon (if any) is reported separately.
"""
from __future__ import annotations
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local


def logit_confidence(model: nn.Module, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """phi = log(p_y / (1 - p_y)) computed stably from logits (LiRA's scaled confidence)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float()).double()
    true = logits.gather(1, torch.from_numpy(y).long().view(-1, 1)).squeeze(1)
    rest = logits.clone()
    rest.scatter_(1, torch.from_numpy(y).long().view(-1, 1), -float("inf"))
    return (true - torch.logsumexp(rest, dim=1)).numpy()


def train_federation(site_arrays, *, features, classes, mode, rounds, local_epochs, batch_size, lr,
                     seed, target_epsilon=None, delta=1e-5, max_grad_norm=1.0):
    """Train one federation on in-memory per-site arrays. Returns (model, achieved_eps list)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DigitMLP(features, classes)
    private = []
    if mode == "opacus_dp":
        from opacus import PrivacyEngine
        from opacus.accountants.utils import get_noise_multiplier
        for (x, y) in site_arrays:
            n = len(y)
            if not 0 < delta < 1 / n:
                raise ValueError("delta must be smaller than 1/n for every client")
            steps = rounds * local_epochs * math.ceil(n / batch_size)
            sigma = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
                                         sample_rate=batch_size / n, steps=steps, accountant="rdp")
            local = DigitMLP(features, classes)
            opt = torch.optim.SGD(local.parameters(), lr=lr)
            dl = DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
                            batch_size=batch_size, shuffle=True)
            engine = PrivacyEngine(accountant="rdp", secure_mode=False)
            pm, po, pdl = engine.make_private(module=local, optimizer=opt, data_loader=dl,
                                              noise_multiplier=sigma, max_grad_norm=max_grad_norm,
                                              poisson_sampling=True)
            private.append((engine, pm, po, pdl))
    for _ in range(rounds):
        states, sizes = [], []
        for c, (x, y) in enumerate(site_arrays):
            if mode == "nonprivate":
                local = copy.deepcopy(model)
                train_local(local, x, y, epochs=local_epochs, batch_size=batch_size, lr=lr, seed=seed + c)
                states.append({k: v.detach().clone() for k, v in local.state_dict().items()})
            else:
                _, pm, po, pdl = private[c]
                pm._module.load_state_dict(model.state_dict())
                pm.train()
                for _ in range(local_epochs):
                    for xb, yb in pdl:
                        po.zero_grad()
                        nn.functional.cross_entropy(pm(xb), yb).backward()
                        po.step()
                states.append({k: v.detach().clone() for k, v in pm._module.state_dict().items()})
            sizes.append(len(y))
        model.load_state_dict(fedavg_weighted(states, sizes))
    eps = [float(e.accountant.get_epsilon(delta)) for e, *_ in private]
    if eps and any(not math.isfinite(e) or e > target_epsilon + 1e-3 for e in eps):
        raise RuntimeError("Native accountant exceeds target epsilon")
    return model, eps


def lira_scores(target_phi: np.ndarray, shadow_phi_in: np.ndarray, shadow_phi_out: np.ndarray,
                min_shadows: int = 4) -> np.ndarray:
    """Per-example online LiRA log likelihood ratio.

    shadow_phi_in / out are lists-of-arrays aligned to examples: shape (n_examples,) object of
    1-D arrays. Examples with too few IN or OUT shadows fall back to the pooled variance.
    """
    n = len(target_phi)
    all_in = np.concatenate([a for a in shadow_phi_in if len(a)]) if n else np.array([])
    all_out = np.concatenate([a for a in shadow_phi_out if len(a)]) if n else np.array([])
    fallback_in = max(float(np.std(all_in)), 1e-3) if len(all_in) > 1 else 1.0
    fallback_out = max(float(np.std(all_out)), 1e-3) if len(all_out) > 1 else 1.0
    scores = np.zeros(n)
    for i in range(n):
        a, b = shadow_phi_in[i], shadow_phi_out[i]
        if len(a) < min_shadows or len(b) < min_shadows:
            scores[i] = 0.0
            continue
        mu_in, mu_out = float(a.mean()), float(b.mean())
        sd_in = max(float(a.std(ddof=1)), 0.25 * fallback_in)
        sd_out = max(float(b.std(ddof=1)), 0.25 * fallback_out)
        t = target_phi[i]
        scores[i] = (-0.5 * ((t - mu_in) / sd_in) ** 2 - math.log(sd_in)
                     + 0.5 * ((t - mu_out) / sd_out) ** 2 + math.log(sd_out))
    return scores


def roc_summary(scores: np.ndarray, membership: np.ndarray, fprs=(0.01, 0.05, 0.1)) -> dict:
    """AUC and true-positive rate at fixed false-positive rates (member = positive)."""
    member, non = scores[membership == 1], scores[membership == 0]
    if len(member) == 0 or len(non) == 0:
        raise ValueError("Need both members and non-members")
    order = np.concatenate([member, non]).argsort().argsort().astype(np.float64) + 1
    uniq, inv, cnt = np.unique(np.concatenate([member, non]), return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=order) / cnt)[inv]
    n1, n0 = len(member), len(non)
    auc = float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    out = {"auc": auc}
    for f in fprs:
        threshold = np.quantile(non, 1 - f)
        out[f"tpr_at_fpr_{f}"] = float((member > threshold).mean())
    return out


def lira_scores_global_var(target_phi: np.ndarray, shadow_phi_in, shadow_phi_out, min_shadows: int = 4) -> np.ndarray:
    """LiRA with one pooled IN variance and one pooled OUT variance (more stable with few shadows).

    Per-example means are kept. Variances are the pooled variance of residuals around those means.
    """
    n = len(target_phi)
    res_in, res_out = [], []
    for i in range(n):
        a, b = shadow_phi_in[i], shadow_phi_out[i]
        if len(a) >= min_shadows and len(b) >= min_shadows:
            res_in.append(a - a.mean())
            res_out.append(b - b.mean())
    if not res_in:
        return np.zeros(n)
    sd_in = max(float(np.concatenate(res_in).std(ddof=1)), 1e-6)
    sd_out = max(float(np.concatenate(res_out).std(ddof=1)), 1e-6)
    scores = np.zeros(n)
    for i in range(n):
        a, b = shadow_phi_in[i], shadow_phi_out[i]
        if len(a) < min_shadows or len(b) < min_shadows:
            continue
        t = target_phi[i]
        scores[i] = (-0.5 * ((t - a.mean()) / sd_in) ** 2 - math.log(sd_in)
                     + 0.5 * ((t - b.mean()) / sd_out) ** 2 + math.log(sd_out))
    return scores
