"""Privileged experiment setup: simulate exposing exact per-batch gradients.

This module must NOT be imported by an attack.  Creates separate attacker and
privileged evaluator artifacts from approved public digits.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP


def create_gradient_observation(site_file: str | Path, directory: str | Path, *,
                                batch_size: int = 1, seed: int = 231, known_labels: bool = False,
                                gradient_clip: float | None = None, gradient_noise_std: float = 0) -> dict:
    if batch_size not in (1, 8, 32) or (batch_size != 1 and not known_labels):
        raise ValueError("Batch > 1 needs an explicit known-label attacker assumption")
    if gradient_clip is not None and gradient_clip <= 0 or gradient_noise_std < 0:
        raise ValueError("Invalid clipping/noise parameters")
    x, y = load_site_partition(site_file)
    if len(y) < batch_size or x.shape[1] != 64:
        raise ValueError("Expected at least batch_size 8x8 digit samples")
    torch.set_num_threads(1)
    generator = np.random.default_rng(seed)
    picks = generator.choice(len(y), size=batch_size, replace=False)
    batch_x, batch_y = torch.from_numpy(x[picks]).float(), torch.from_numpy(y[picks]).long()
    torch.manual_seed(seed)
    model = DigitMLP(features=64, classes=int(max(10, int(y.max()) + 1)))
    logits = model(batch_x)
    loss = nn.functional.cross_entropy(logits, batch_y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    obs_grads = {k: v.detach().clone() for (k, _), v in zip(model.named_parameters(), grads)}
    original_norm = torch.sqrt(sum(g.square().sum() for g in obs_grads.values()))
    if gradient_clip is not None:
        factor = min(1.0, float(gradient_clip / original_norm))
        obs_grads = {k: v * factor for k, v in obs_grads.items()}
    if gradient_noise_std:
        noise_generator = torch.Generator().manual_seed(seed + 9007)
        obs_grads = {k: v + gradient_noise_std * torch.randn(v.shape, generator=noise_generator)
                     for k, v in obs_grads.items()}
    out = Path(directory)
    attacker = out / "attacker"
    evaluator = out / "evaluator_only"
    attacker.mkdir(parents=True, exist_ok=True)
    evaluator.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "observer": "server_individual_exact_gradient", "batch_size": batch_size,
               "features": 64, "classes": model.network[-1].out_features,
               "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
               "gradients": obs_grads}
    if known_labels:
        payload["known_labels"] = batch_y.tolist()
    torch.save(payload, attacker / "observation.pt")
    np.savez_compressed(evaluator / "original.npz", x=batch_x.numpy(), y=batch_y.numpy(), source_idx=picks)
    report = {"observer": payload["observer"], "batch_size": batch_size, "known_labels": known_labels,
              "gradient_clip": gradient_clip, "gradient_noise_std": gradient_noise_std,
              "gradient_l2_before_defense": float(original_norm),
              "setting": "controlled simulated full gradient; not actual Flower coordinator traffic",
              "privacy_guarantee": "none; clipping and arbitrary noise do not constitute formal DP",
              "attacker_receives": list(payload)}
    (out / "observation_contract.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
