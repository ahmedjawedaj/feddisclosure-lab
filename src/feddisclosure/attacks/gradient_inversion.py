"""Exact-gradient reconstruction in an explicitly controlled public-data sandbox.

The attack imports neither datasets nor evaluator code.  Its only inputs are an
initial public model state and gradients released by a *simulated* coordinator
observer, plus labels when the threat model explicitly permits them.  This is
not an attack on secure aggregation or a practical multi-step FedAvg update.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from feddisclosure.training.task import DigitMLP


def infer_single_label(grads: dict[str, torch.Tensor]) -> int:
    """iDLG-style label inference from the final layer bias gradient, batch=1.

    Applicable to a softmax cross-entropy classifier with a single example;
    do not apply this shortcut to general multi-example observations.
    """
    final_bias = grads["network.2.bias"].detach()
    if final_bias.ndim != 1 or len(final_bias) < 2:
        raise ValueError("Expected a classifier final-layer bias gradient")
    return int(torch.argmin(final_bias))


def reconstruct(observation: dict, *, iterations: int = 350, lr: float = 0.12,
                seed: int = 17, tv_weight: float = 0.0005) -> tuple[np.ndarray, dict]:
    """Optimize dummy input images to match only permitted gradient observations."""
    if not (1 <= iterations <= 3000) or lr <= 0 or tv_weight < 0:
        raise ValueError("Invalid reconstruction hyperparameters")
    observer = observation.get("observer")
    if observation.get("schema_version") != 1 or observer not in (
        "server_individual_exact_gradient", "server_individual_model_delta",
        "server_aggregate_model_delta_target_probe", "server_collusion_isolated_delta",
        "server_aggregate_model_delta_joint_oracle"
    ):
        raise ValueError("Only explicitly allowlisted gradient/update observer contracts are supported")
    torch.set_num_threads(1)
    features, classes = int(observation["features"]), int(observation["classes"])
    batch_size = int(observation["batch_size"])
    if batch_size not in (1, 8, 16, 24, 32, 64) or features != 64:
        raise ValueError("Supported controlled experiments use 1, 8, 16, 24, 32, or 64 public 8x8 digits")
    model = DigitMLP(features, classes)
    model.load_state_dict(observation["model_state"])
    model.eval()
    if observer in ("server_individual_model_delta",
                    "server_aggregate_model_delta_target_probe", "server_collusion_isolated_delta",
        "server_aggregate_model_delta_joint_oracle"):
        steps = int(observation["local_steps"])
        rate = float(observation["learning_rate"])
        if steps < 1 or rate <= 0:
            raise ValueError("Untrusted model-delta optimizer configuration")
        # The attacker *computes* an effective average gradient from the allowed
        # initial parameters, final-model delta, step count and SGD rate. This
        # is exact for one SGD step, approximate for multiple steps.
        gradients = {k: delta / (rate * steps)
                     for k, delta in observation["model_delta"].items()}
    else:
        gradients = observation["gradients"]
    names = list(dict(model.named_parameters()))
    if list(gradients) != names:
        raise ValueError("Observation gradient structure does not match the model")
    if any(gradients[k].shape != p.shape or not torch.isfinite(gradients[k]).all()
           for k, p in model.named_parameters()):
        raise ValueError("Invalid gradient shape or value")
    if "known_labels" in observation:
        labels = torch.as_tensor(observation["known_labels"], dtype=torch.long)
    elif batch_size == 1:
        labels = torch.tensor([infer_single_label(gradients)])
    else:
        raise ValueError("Multi-example reconstruction requires explicitly observed labels")
    if labels.shape != (batch_size,) or (labels < 0).any() or (labels >= classes).any():
        raise ValueError("Observed labels do not match the declared batch")
    generator = torch.Generator().manual_seed(seed)
    dummy = (torch.rand((batch_size, features), generator=generator) * 0.6 + 0.2).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy], lr=lr)
    targets = [gradients[name].detach() for name in names]
    history = []
    start = time.perf_counter()
    for step in range(1, iterations + 1):
        optimizer.zero_grad()
        score = nn.functional.cross_entropy(model(dummy), labels)
        simulated = torch.autograd.grad(score, tuple(model.parameters()), create_graph=True)
        match = sum((a - b).square().sum() for a, b in zip(simulated, targets))
        tiles = dummy.reshape(-1, 8, 8)
        tv = (tiles[:, 1:, :] - tiles[:, :-1, :]).abs().mean() + (tiles[:, :, 1:] - tiles[:, :, :-1]).abs().mean()
        objective = match + tv_weight * tv
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            dummy.clamp_(0, 1)
        if step == 1 or step % 25 == 0 or step == iterations:
            history.append({"iteration": step, "gradient_match_sum_squared": float(match.detach()),
                            "objective": float(objective.detach())})
    recovery = dummy.detach().cpu().numpy()
    return recovery, {"algorithm": "DLG_Adam_gradient_matching_or_model_delta_proxy",
                      "observer": observation["observer"],
                      "local_steps": int(observation.get("local_steps", 1)),
                      "gradient_target": "model_delta_divided_by_lr_times_steps" if observer != "server_individual_exact_gradient" else "exact_released_gradient",
                      "batch_size": batch_size, "labels_available": "known_labels" in observation,
                      "recovered_labels": labels.tolist(), "iterations": iterations,
                      "lr": lr, "tv_weight": tv_weight, "seed": seed,
                      "duration_seconds": round(time.perf_counter() - start, 4),
                      "gradient_matching_history": history,
                      "limitations": ["An exact gradient or model-delta proxy; aggregate-only is NOT individual-update reconstruction",
                                      "Requires full per-example or per-batch exact gradients and known architecture/weights",
                                      "Batch>1 runs require true labels as an explicitly declared extra attacker privilege",
                                      "A low gradient residual does not prove recognizable image reconstruction",
                                      "Model-delta proxy is not a true initial gradient when local_steps > 1"]}


def run_attack(observation_path: str | Path, output_dir: str | Path, **kwargs) -> dict:
    # weights_only excludes arbitrary pickle object execution, even for local benchmarks.
    obs = torch.load(observation_path, map_location="cpu", weights_only=True)
    if any(key in obs for key in ("x", "images", "original_images", "source_idx", "membership", "ground_truth")):
        raise ValueError("Ground-truth data must not appear in attacker observations")
    recovered, report = reconstruct(obs, **kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "recovered.npz", x=recovered)
    (out / "attack_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
