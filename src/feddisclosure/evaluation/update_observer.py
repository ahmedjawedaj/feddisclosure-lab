"""Privileged public-data setup for actual LOCAL SGD model-delta observations.

Emits the initial weights and local final-minus-initial parameter DELTA, which
is the information a server can obtain from an unprotected individual client
update. Local steps can be >1, so the averaged delta is NOT an exact initial
model gradient except in the single-step case.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP


def create_update_observation(site_file: str | Path, directory: str | Path, *,
                              local_steps: int = 1, seed: int = 231, learning_rate: float = 0.15) -> dict:
    if not 1 <= local_steps <= 100 or learning_rate <= 0:
        raise ValueError("Invalid local SGD update config")
    x, y = load_site_partition(site_file)
    if x.shape[1] != 64 or len(y) < 1:
        raise ValueError("Only approved public 8x8 digits supported")
    rng = np.random.default_rng(seed)
    index = int(rng.choice(len(y), size=1, replace=False)[0])
    image = torch.tensor(x[[index]], dtype=torch.float32)
    label = torch.tensor(y[[index]], dtype=torch.long)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DigitMLP(64, max(10, int(y.max()) + 1))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # This one-record repeated SGD regime is a controlled limitation, not a
    # representative many-record, many-minibatch Flower local training loop.
    first_grads = torch.autograd.grad(nn.functional.cross_entropy(model(image), label), tuple(model.parameters()))
    names = list(dict(model.named_parameters()))
    exact = {k: t.detach().clone() for k, t in zip(names, first_grads)}
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(local_steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(image), label)
        loss.backward()
        optimizer.step()
    model_delta = {k: initial[k] - model.state_dict()[k].detach() for k in names}
    # An observer knowing local step count and learning rate may compute this
    # proxy. It need not equal the gradient at the INITIAL weights for steps>1.
    gradient_proxy = {k: model_delta[k] / (learning_rate * local_steps) for k in names}
    relative_error = float(torch.sqrt(sum((gradient_proxy[k] - exact[k]).square().sum() for k in names)) /
                           torch.sqrt(sum(g.square().sum() for g in exact.values())).clamp(min=1e-12))
    root = Path(directory)
    (root / "attacker").mkdir(parents=True, exist_ok=True)
    (root / "evaluator_only").mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "observer": "server_individual_model_delta",
               "batch_size": 1, "features": 64, "classes": model.network[-1].out_features,
               "model_state": initial, "model_delta": model_delta,
               "learning_rate": learning_rate, "local_steps": local_steps}
    torch.save(payload, root / "attacker" / "observation.pt")
    np.savez_compressed(root / "evaluator_only" / "original.npz", x=image.numpy(), y=label.numpy(), source_idx=[index])
    report = {"observer": payload["observer"], "local_steps": local_steps, "batch_size": 1,
              "learning_rate_known": True, "one_record_repeated_sgd": True,
              "gradient_proxy_relative_l2_error_vs_exact_initial_gradient": relative_error,
              "attacker_receives": list(payload),
              "limitations": ["Unprotected individual client model update, not an aggregate secure-aggregation output",
                              "One public image repeatedly used in local SGD; not general multi-example FL",
                              "The averaged model delta is a proxy rather than an exact gradient for >1 steps"]}
    (root / "observation_contract.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
