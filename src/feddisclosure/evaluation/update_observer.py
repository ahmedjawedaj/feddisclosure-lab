"""Privileged setup for individual SGD client-update observations on public digits.

A simulated unprotected coordinator receives initial weights and a local model
update, never the pixels. Evaluator-only original images and label ground truth
are written outside the attacker directory. For batches >1, labels are an
explicitly optional *attacker privilege*, not inferred by this module.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP


def create_update_observation(site_file: str | Path, directory: str | Path, *,
                              local_steps: int = 1, seed: int = 231, learning_rate: float = 0.15,
                              batch_size: int = 1, disclose_labels: bool = False) -> dict:
    if not 1 <= local_steps <= 100 or not math.isfinite(learning_rate) or learning_rate <= 0 or batch_size not in (1, 8, 32):
        raise ValueError("Invalid local SGD update config")
    x, y = load_site_partition(site_file)
    if x.ndim != 2 or x.shape[1] != 64 or len(y) < batch_size:
        raise ValueError("Only approved public 8x8 digits with enough records supported")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y), size=batch_size, replace=False)
    images = torch.tensor(x[indices], dtype=torch.float32)
    labels = torch.tensor(y[indices], dtype=torch.long)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DigitMLP(64, max(10, int(y.max()) + 1))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    names = list(dict(model.named_parameters()))
    first_grads = torch.autograd.grad(nn.functional.cross_entropy(model(images), labels), tuple(model.parameters()))
    exact = {k: t.detach().clone() for k, t in zip(names, first_grads)}
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # This is local full-batch SGD on B DIFFERENT private records over K steps.
    # For K>1 it is an idealized fixed local dataset, not multiple changing minibatches.
    for _ in range(local_steps):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
    model_delta = {k: initial[k] - model.state_dict()[k].detach() for k in names}
    gradient_proxy = {k: model_delta[k] / (learning_rate * local_steps) for k in names}
    relative_error = float(torch.sqrt(sum((gradient_proxy[k] - exact[k]).square().sum() for k in names)) /
                           torch.sqrt(sum(g.square().sum() for g in exact.values())).clamp(min=1e-12))
    root = Path(directory)
    (root / "attacker").mkdir(parents=True, exist_ok=True)
    (root / "evaluator_only").mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "observer": "server_individual_model_delta",
               "batch_size": batch_size, "features": 64, "classes": model.network[-1].out_features,
               "model_state": initial, "model_delta": model_delta,
               "learning_rate": learning_rate, "local_steps": local_steps}
    if disclose_labels:
        payload["known_labels"] = labels.tolist()
    torch.save(payload, root / "attacker" / "observation.pt")
    np.savez_compressed(root / "evaluator_only" / "original.npz", x=images.numpy(), y=labels.numpy(),
                        source_idx=indices)
    report = {"observer": payload["observer"], "local_steps": local_steps, "batch_size": batch_size,
              "learning_rate_known": True, "distinct_training_records": batch_size,
              "fixed_local_batch_repeated": local_steps > 1,
              "labels_disclosed_to_attacker": disclose_labels,
              "gradient_proxy_relative_l2_error_vs_exact_initial_gradient": relative_error,
              "attacker_receives": list(payload),
              "limitations": ["Individual client model update without secure aggregation or DP",
                              "One fixed local batch of distinct public images reused across steps",
                              "Server knows learning rate, local steps, initial weights and architecture",
                              "Batch reconstruction with labels requires the privileged known-label scenario",
                              "The model delta proxy differs from the initial gradient after multiple steps"]}
    (root / "observation_contract.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
