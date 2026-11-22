"""Work package WP11: do the repository's own attacks fall short of established external ones?

Two checks from docs/preregistration/v0.15.md.

B1  inversion: the repo gradient matching attack against the `breaching` library on the same
    update (4 clients x 8 Digits records, one local SGD step, lr 0.15, plaintext individual update,
    labels known). Pass rule: repo mean MSE at most 1.5 times the external baseline mean MSE.
B2  membership: the repo LiRA against a reference LiRA on the same shadow models. Pass rule: repo
    AUC at least reference AUC minus 0.03.

This module holds the pure parts (record selection, scoring, verdict logic, bootstrap, an
independent LiRA reimplementation) and thin lazy wrappers around the external packages. The external
packages are imported only inside the wrapper functions, so the module imports in the main
environment even when `breaching` or `mialibrary` are not installed. No external code is copied here.
The thresholds are constants of the pre-registration and are never changed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from feddisclosure.attacks.view_seed import select_view_records
from feddisclosure.training.lira_study import lira_scores, roc_summary

B1_MAX_RATIO = 1.5
B2_AUC_MARGIN = 0.03
RECOGNISABLE_MSE = 0.05
N_CLIENTS, RECORDS_PER_CLIENT, LOCAL_LR = 4, 8, 0.15
PARAM_NAMES = ("network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias")


# ----------------------------------------------------------------------------------------------
# Matched setup
# ----------------------------------------------------------------------------------------------
def matched_records(x: np.ndarray, y: np.ndarray, seed: int, *, n_clients: int = N_CLIENTS,
                    records: int = RECORDS_PER_CLIENT) -> dict:
    """The victim records and public pool of the repo view study for this seed.

    Uses `select_view_records` itself, so the data selection is identical by construction.
    Client 0 is the victim."""
    sel = select_view_records(len(y), n_clients, records, seed)
    cx, cy = x[sel.idx].reshape(n_clients, records, -1), y[sel.idx].reshape(n_clients, records)
    return {"victim_x": cx[0], "victim_y": cy[0], "client_x": cx, "client_y": cy,
            "pool_x": x[sel.pool_idx], "pool_y": y[sel.pool_idx], "idx": sel.idx}


def victim_gradient(init: dict, delta: dict, *, lr: float = LOCAL_LR) -> list:
    """Gradient the attacker derives from a one-step plaintext update: (init - final) / lr.

    Exactly what the repo attack divides by (model_delta / (lr * steps)), returned in parameter
    order so a library that wants a gradient list can take it."""
    return [delta[k] / lr for k in PARAM_NAMES]


# ----------------------------------------------------------------------------------------------
# Scoring and verdicts
# ----------------------------------------------------------------------------------------------
def mse_summary(per_image_mse: Sequence[float]) -> dict:
    m = np.asarray(per_image_mse, dtype=float)
    return {"mse": float(m.mean()), "recognisable": float((m < RECOGNISABLE_MSE).mean())}


def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int = 10000, seed: int = 0,
                      confidence: float = 0.95) -> list[float]:
    """Percentile bootstrap interval of the mean. Returns [mean, low, high]."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        raise ValueError("need at least two values for a bootstrap interval")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - confidence) / 2
    return [float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def b1_verdict(repo_mse: Sequence[float], external_mse: Sequence[float], *,
               max_ratio: float = B1_MAX_RATIO) -> dict:
    """B1. PASS when mean(repo MSE) <= max_ratio * mean(external MSE). Lower MSE means a stronger attack."""
    r, e = np.asarray(repo_mse, float), np.asarray(external_mse, float)
    if len(r) != len(e) or len(r) == 0:
        raise ValueError("repo and external MSE must be paired and non-empty")
    if not (np.isfinite(r).all() and np.isfinite(e).all()):
        raise ValueError("nonfinite MSE")
    ratio = float(r.mean() / e.mean()) if e.mean() > 0 else float("inf")
    return {"repo_mean_mse": float(r.mean()), "external_mean_mse": float(e.mean()), "ratio": ratio,
            "threshold_max_ratio": max_ratio, "verdict": "PASS" if r.mean() <= max_ratio * e.mean() else "FAIL"}


def b2_verdict(repo_auc: float, reference_auc: float, *, margin: float = B2_AUC_MARGIN) -> dict:
    """B2. PASS when repo AUC >= reference AUC - margin."""
    if not (math.isfinite(repo_auc) and math.isfinite(reference_auc)):
        raise ValueError("nonfinite AUC")
    return {"repo_auc": float(repo_auc), "reference_auc": float(reference_auc),
            "difference_repo_minus_reference": float(repo_auc - reference_auc),
            "threshold_min_difference": -margin,
            "verdict": "PASS" if repo_auc >= reference_auc - margin else "FAIL"}


# ----------------------------------------------------------------------------------------------
# Membership: independent LiRA reimplementation and AUC bootstrap
# ----------------------------------------------------------------------------------------------
def independent_lira_scores(target_phi: np.ndarray, shadow_phi: np.ndarray, shadow_member: np.ndarray, *,
                            fix_variance: bool = False, use_median: bool = False,
                            eps: float = 1e-30) -> np.ndarray:
    """Online LiRA (Carlini et al. 2022) written from the paper, as a cross-check.

    target_phi (n,) logit-scaled confidences of the target model. shadow_phi and shadow_member
    (m, n). Per example the IN and OUT shadow values are fitted with a Gaussian (mean or median for
    the centre, per-example or one global standard deviation) and the score is
    log N(target | in) - log N(target | out). Higher means more likely a member. No fallback is used,
    so every example needs at least two IN and two OUT shadows when the variance is per example."""
    phi, mem = np.asarray(shadow_phi, float), np.asarray(shadow_member, bool)
    if phi.shape != mem.shape or phi.ndim != 2 or phi.shape[1] != len(target_phi):
        raise ValueError("shadow arrays must be (m, n) and match the target length")
    sin, sout = np.where(mem, phi, np.nan), np.where(~mem, phi, np.nan)
    if (mem.sum(axis=0) < 2).any() or ((~mem).sum(axis=0) < 2).any():
        raise ValueError("every example needs at least two IN and two OUT shadow models")
    centre = np.nanmedian if use_median else np.nanmean
    mu_in, mu_out = centre(sin, axis=0), centre(sout, axis=0)
    if fix_variance:
        sd_in, sd_out = np.nanstd(sin - mu_in), np.nanstd(sout - mu_out)
    else:
        sd_in, sd_out = np.nanstd(sin, axis=0), np.nanstd(sout, axis=0)
    t = np.asarray(target_phi, float)

    def logpdf(v, mu, sd):
        sd = sd + eps
        return -0.5 * ((v - mu) / sd) ** 2 - np.log(sd) - 0.5 * math.log(2 * math.pi)

    return logpdf(t, mu_in, sd_in) - logpdf(t, mu_out, sd_out)


def auc(scores: np.ndarray, member: np.ndarray) -> float:
    """Rank AUC, member is the positive class. Same function the repo LiRA study uses."""
    return float(roc_summary(np.asarray(scores, float), np.asarray(member).astype(int))["auc"])


def repo_lira_scores(target_phi: np.ndarray, shadow_phi: np.ndarray, shadow_member: np.ndarray) -> np.ndarray:
    """The repo's own LiRA scoring on arrays, in the way scripts/run_lira_study.py attack() calls it."""
    n = shadow_phi.shape[1]
    ins = [shadow_phi[shadow_member[:, i], i] for i in range(n)]
    outs = [shadow_phi[~shadow_member[:, i], i] for i in range(n)]
    return lira_scores(target_phi, ins, outs)


def bootstrap_auc(score_fn_outputs: Sequence[np.ndarray], members: Sequence[np.ndarray], *,
                  n_boot: int = 2000, seed: int = 0, confidence: float = 0.95,
                  paired_with: Sequence[np.ndarray] | None = None) -> dict:
    """Two level percentile bootstrap of the mean per-target AUC.

    Each replicate resamples the targets with replacement and, inside every resampled target, the
    examples with replacement (members and non-members separately, so both classes stay present).
    With `paired_with` (scores of a second attack on the same targets and the same example
    resamples) the interval is for the difference first minus second.
    Returns {"mean", "low", "high"} for the AUC, or for the difference when paired."""
    k = len(score_fn_outputs)
    if k < 2 or len(members) != k:
        raise ValueError("need at least two targets and one membership vector per target")
    rng = np.random.default_rng(seed)
    a = (1 - confidence) / 2
    point = [auc(s, m) for s, m in zip(score_fn_outputs, members)]
    if paired_with is not None:
        point = [p - auc(s2, m) for p, s2, m in zip(point, paired_with, members)]
    reps = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for t in rng.integers(0, k, size=k):
            m = np.asarray(members[t]).astype(bool)
            pos, neg = np.flatnonzero(m), np.flatnonzero(~m)
            ex = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            v = auc(score_fn_outputs[t][ex], m[ex])
            if paired_with is not None:
                v -= auc(paired_with[t][ex], m[ex])
            vals.append(v)
        reps[b] = np.mean(vals)
    return {"mean": float(np.mean(point)), "low": float(np.quantile(reps, a)),
            "high": float(np.quantile(reps, 1 - a))}


# ----------------------------------------------------------------------------------------------
# External tools (lazy imports)
# ----------------------------------------------------------------------------------------------
def external_versions() -> dict:
    """Installed versions and declared licences of the external tools, read from package metadata."""
    from importlib import metadata
    out = {}
    for name in ("breaching", "mialibrary", "torch", "torchvision", "numpy", "scipy", "jax", "hydra-core", "omegaconf"):
        try:
            md = metadata.metadata(name)
            lic = (md.get("License-Expression") or md.get("License") or "").strip()
            out[name] = {"version": md["Version"],
                         "license": (lic.splitlines()[0][:80] if lic else None)}
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def breaching_reconstruct(init: dict, gradients: list, labels: np.ndarray, *, attack: str = "invertinggradients",
                          overrides: Sequence[str] = (), seed: int = 0) -> np.ndarray:
    """Run a `breaching` optimisation attack on one known gradient and return (n, 64) reconstructions.

    The library classes are called directly (`breaching.attacks.prepare_attack`). Adaptations that
    the 64 feature MLP needs, all documented in docs/v0.15-external-baselines.md:
      * the 8x8 digit is presented to the library as a 1x8x8 image and the model is wrapped so that it
        flattens it, which leaves the weights and the gradient unchanged,
      * the library total variation regulariser is built for three channels, so its fixed difference
        kernel is cut to one channel (same kernel, same scale),
      * data normalisation is the identity (mean 0, std 1) so the box constraint is the pixel range [0, 1].
    Labels are passed in, so the label recovery step of the library is not used (known labels, as in
    the repo attack). The default attack configuration is the library's own `attack` yaml, changed only
    by `overrides` (hydra syntax such as optim.max_iterations=4000)."""
    import breaching
    import torch
    from breaching.attacks import prepare_attack
    from breaching.attacks.auxiliaries.regularizers import TotalVariation
    from omegaconf import OmegaConf
    from torch import nn

    from feddisclosure.training.task import DigitMLP

    class Flatten1Channel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x.flatten(1))

    torch.set_num_threads(1)
    inner = DigitMLP(64, 10)
    inner.load_state_dict(init)
    model = Flatten1Channel(inner)
    n = int(len(labels))
    cfg = breaching.get_attack_config(attack, overrides=list(overrides))
    meta = OmegaConf.create({"shape": [1, 8, 8], "modality": "vision", "mean": [0.0], "std": [1.0]})
    payload = [{"metadata": meta, "parameters": [p.detach().clone() for p in model.parameters()], "buffers": None}]
    shared = [{"gradients": [g.detach().clone() for g in gradients], "buffers": None,
               "metadata": {"num_data_points": n, "labels": torch.as_tensor(np.asarray(labels), dtype=torch.long),
                            "local_hyperparams": None}}]
    attacker = prepare_attack(model, nn.CrossEntropyLoss(), cfg)
    for reg in attacker.regularizers:
        if isinstance(reg, TotalVariation):
            reg.groups = 1
            reg.weight = reg.weight[:2].contiguous()
    torch.manual_seed(seed)
    rec, _ = attacker.reconstruct(payload, shared, server_secrets={})
    out = rec["data"].detach().cpu().numpy().reshape(n, 64).astype(np.float64)
    if not np.isfinite(out).all():
        raise RuntimeError("breaching returned a nonfinite reconstruction")
    return out


def mialibrary_scores(target_phi: np.ndarray, shadow_phi: np.ndarray, shadow_member: np.ndarray, *,
                      fix_variance: bool, use_median: bool) -> np.ndarray:
    """LiRA scores from the `mialibrary` package (simple_lira, numpy path) for one target model.

    Only the supplied shadow models are used (the helper that removes a target from a pool of models is
    not used), so the shadow set is exactly the one the repo LiRA uses."""
    from mialibrary.lira.simple_lira import run_simple_lira
    phi = np.asarray(shadow_phi, float)[:, :, None]
    mem = np.asarray(shadow_member, bool)[:, :, None]
    stat_in, stat_out = np.where(mem, phi, np.nan), np.where(~mem, phi, np.nan)
    return np.asarray(run_simple_lira(np.asarray(target_phi, float)[:, None], stat_in, stat_out,
                                      fix_variance=fix_variance, use_median=use_median))


def load_saved_lira(path: Path, mode: str = "nonprivate", epsilon: float = -1.0) -> dict:
    """Saved shadow and target models of scripts/run_lira_study.py (phi and member per model)."""
    with np.load(path, allow_pickle=False) as d:
        sel = (d["mode"] == mode) & np.isclose(d["target_epsilon"], epsilon)
        order = np.argsort(d["model_id"][sel])
        return {"phi": d["phi"][sel][order].astype(np.float64), "member": d["member"][sel][order].astype(bool),
                "model_id": d["model_id"][sel][order]}


def split_models(models: dict, shadows: int) -> tuple[dict, dict]:
    """First `shadows` models are shadows and the rest are targets, as in the repo study."""
    sh = {k: v[:shadows] for k, v in models.items()}
    tg = {k: v[shadows:] for k, v in models.items()}
    return sh, tg


def evaluate_lira(models: dict, shadows: int, score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
                  ) -> tuple[list[np.ndarray], list[float]]:
    """Scores and AUC for every target model with a given scoring function."""
    sh, tg = split_models(models, shadows)
    scores = [score_fn(p, sh["phi"], sh["member"]) for p in tg["phi"]]
    return scores, [auc(s, m) for s, m in zip(scores, tg["member"])]
