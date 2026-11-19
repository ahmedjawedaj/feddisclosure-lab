"""Privileged experiment setup for an unprotected client update across changing SGD minibatches.

An attacker may see one full client model delta and, in explicitly privileged
cases, the labels of all contributing examples. Original pixels, example IDs,
SGD minibatch membership and evaluator-only ground truth are not released.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP


def create_minibatch_update_observation(
    site_file: str | Path,
    directory: str | Path,
    *,
    batch_size: int = 8,
    local_steps: int = 4,
    seed: int = 241,
    schedule_seed: int = 0,
    learning_rate: float = 0.15,
    disclose_labels: bool = True,
) -> dict:
    """Run K local SGD steps using K disjoint batches, exactly once per record.

    The implied gradient average is over K*B distinct training examples.
    It is *not* an exact first-step gradient after the first local update.
    """
    if (batch_size not in (1, 8, 16, 32) or not 1 <= local_steps <= 16
            or batch_size * local_steps > 64
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not isinstance(seed, int) or not isinstance(schedule_seed, int)):
        raise ValueError("Invalid changing-minibatch SGD configuration")
    x, y = load_site_partition(site_file)
    count = batch_size * local_steps
    if x.ndim != 2 or x.shape[1] != 64 or len(y) < count or len(y) != len(x):
        raise ValueError("Require sufficient approved public 8x8 digit examples")
    indices = np.random.default_rng(seed).choice(len(y), size=count, replace=False)
    images = torch.as_tensor(x[indices], dtype=torch.float32).clone()
    labels = torch.as_tensor(y[indices], dtype=torch.long).clone()
    order = np.random.default_rng(schedule_seed).permutation(count)

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DigitMLP(64, max(10, int(y.max()) + 1))
    model.train()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    named_parameters = list(model.named_parameters())
    full_initial_grads = torch.autograd.grad(
        nn.functional.cross_entropy(model(images), labels), tuple(model.parameters())
    )
    full_initial = {k: grad.detach().clone() for (k, _), grad in zip(named_parameters, full_initial_grads)}
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for step in range(local_steps):
        selected = order[step * batch_size:(step + 1) * batch_size]
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(images[selected]), labels[selected]).backward()
        optimizer.step()
    delta = {name: initial[name] - model.state_dict()[name].detach() for name, _ in named_parameters}
    proxy = {name: value / (learning_rate * local_steps) for name, value in delta.items()}
    numerator = torch.sqrt(sum((proxy[k] - full_initial[k]).square().sum() for k in proxy))
    denominator = torch.sqrt(sum(g.square().sum() for g in full_initial.values())).clamp(min=1e-12)
    proxy_error = float(numerator / denominator)

    directory = Path(directory)
    attack_dir, eval_dir = directory / "attacker", directory / "evaluator_only"
    attack_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    observation = {
        "schema_version": 1,
        "observer": "server_individual_model_delta",
        "batch_size": count,  # total distinct target images, NOT SGD minibatch size
        "features": 64,
        "classes": model.network[-1].out_features,
        "model_state": initial,
        "model_delta": delta,
        "learning_rate": learning_rate,
        "local_steps": local_steps,
    }
    if disclose_labels:
        observation["known_labels"] = labels.tolist()
    torch.save(observation, attack_dir / "observation.pt")
    np.savez_compressed(eval_dir / "original.npz", x=images.numpy(), y=labels.numpy(), source_idx=indices)
    report = {
        "observer": "server_individual_model_delta",
        "dataset": "approved_public_8x8_digits",
        "local_steps": local_steps,
        "sgd_minibatch_size": batch_size,
        "distinct_training_records": count,
        "sampling": "disjoint_minibatches_without_replacement",
        "each_record_used_exactly_once": True,
        "learning_rate": learning_rate,
        "data_seed": seed,
        "schedule_seed": schedule_seed,
        "labels_disclosed_to_attacker": disclose_labels,
        "minibatch_schedule_disclosed_to_attacker": False,
        "gradient_proxy_relative_l2_error_vs_exact_initial_full_dataset_gradient": proxy_error,
        "attacker_receives": list(observation),
        "observation_sha256": hashlib.sha256((attack_dir / "observation.pt").read_bytes()).hexdigest(),
        "limitations": [
            "One unprotected client update from disjoint SGD minibatches, not Flower or secure aggregation",
            "Attacker receives the label vector for every contributing image if disclose_labels=True",
            "No optimizer-state/momentum/weight-decay effects; SGD learning rate is known",
            "Gradient proxy is delta/(learning_rate*steps) and differs from a true single-batch gradient",
            "Distinct target records equal minibatch_size*steps; comparisons to repeated batches are not compute-matched",
            "Code-level, not filesystem/OS-level attacker-evaluator isolation",
        ],
    }
    (directory / "observation_contract.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
