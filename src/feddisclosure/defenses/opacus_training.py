"""Native Opacus DP-SGD proof-of-execution for ONE public-data silo.

A single local training job only. No claim about FedAvg, federated composition,
secure aggregation, or formal guarantees before a native Opacus accountant run.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feddisclosure.training.task import DigitMLP, evaluate_local


class OpacusUnavailable(RuntimeError):
    pass


def validate_dp_config(*, records: int, epochs: int, batch_size: int,
                       learning_rate: float, target_epsilon: float, target_delta: float,
                       max_grad_norm: float) -> None:
    if not isinstance(records, int) or records < 40:
        raise ValueError("DP test requires at least 40 records")
    if not isinstance(epochs, int) or not 1 <= epochs <= 50:
        raise ValueError("epochs must be in [1, 50]")
    if not isinstance(batch_size, int) or not 1 <= batch_size < records:
        raise ValueError("batch_size must be smaller than records")
    for name, val in (("learning_rate", learning_rate), ("target_epsilon", target_epsilon),
                      ("max_grad_norm", max_grad_norm)):
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(target_delta) or not (0 < target_delta < 1 / records):
        raise ValueError("target_delta must be positive and smaller than 1/records")


def run_local_dp_sgd(x: np.ndarray, y: np.ndarray, *, seed: int = 241,
                     epochs: int = 2, batch_size: int = 32,
                     learning_rate: float = .15, target_epsilon: float = 5.0,
                     target_delta: float = 1e-5, max_grad_norm: float = 1.0,
                     initial_state: dict | None = None):
    """Returns trained state and native-accountant-only metadata.

    Poisson sampling is enabled. No custom noise, no manual epsilon estimate.
    The model is used for research only (secure_mode=False).
    """
    n = len(y)
    validate_dp_config(records=n, epochs=epochs, batch_size=batch_size,
                       learning_rate=learning_rate, target_epsilon=target_epsilon,
                       target_delta=target_delta, max_grad_norm=max_grad_norm)
    if x.shape != (n, 64) or y.shape != (n,) or not np.isfinite(x).all() or (y < 0).any() or (y > 9).any():
        raise ValueError("Expected finite public 8x8 digit inputs and labels 0..9")
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise OpacusUnavailable("Native Opacus is required; refusing a simulated DP-SGD result") from exc
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DigitMLP()
    if initial_state is not None:
        model.load_state_dict(initial_state)
    initial_hash = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    ds = TensorDataset(torch.tensor(x.copy(), dtype=torch.float32), torch.tensor(y.copy(), dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    private_model, private_opt, private_dl = engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=dl,
        target_epsilon=target_epsilon, target_delta=target_delta, epochs=epochs,
        max_grad_norm=max_grad_norm, poisson_sampling=True,
    )
    total_steps, loss_sum = 0, 0.0
    private_model.train()
    for _ in range(epochs):
        for xb, yb in private_dl:
            private_opt.zero_grad()
            loss = nn.functional.cross_entropy(private_model(xb), yb)
            loss.backward()
            private_opt.step()
            loss_sum += float(loss.detach())
            total_steps += 1
    if not total_steps:
        raise RuntimeError("DP-SGD completed no optimization steps")
    epsilon = float(engine.get_epsilon(delta=target_delta))
    if not math.isfinite(epsilon) or epsilon > target_epsilon + 1e-5:
        raise RuntimeError("Native Opacus accountant exceeds the configured epsilon cap")
    noise_multiplier = float(private_opt.noise_multiplier)
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise RuntimeError("Invalid native noise multiplier")
    state = {k: v.detach().clone().cpu() for k, v in private_model._module.state_dict().items()}
    return state, {
        "mechanism": "native_opacus_poisson_dp_sgd", "unit_of_privacy": "one training record",
        "adjacency": "add_remove_one_record", "accountant": "Opacus RDP",
        "epsilon_accounted": epsilon, "delta": target_delta,
        "target_epsilon": target_epsilon, "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm, "poisson_sampling": True,
        "sample_rate": batch_size / n, "records": n, "epochs": epochs,
        "steps": total_steps, "avg_training_loss": loss_sum / total_steps,
        "learning_rate": learning_rate, "seed": seed,
        "initial_weights_sha256": initial_hash,
        "limitations": [
            "ONE local training invocation only; repeated FL rounds require cumulative accounting",
            "Model selection, preprocessing, privacy of validation data and public model initialization are separate assumptions",
            "Opacus secure_mode=False is a reproducible research setting, NOT cryptographically secure randomness",
            "Poisson minibatch sampling changes realized optimizer steps and batch sizes versus a standard SGD control",
            "This component alone is not a federated DP guarantee nor secure aggregation",
        ],
    }
