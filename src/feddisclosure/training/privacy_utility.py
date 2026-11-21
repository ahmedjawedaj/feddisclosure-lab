"""Matched non-private vs Opacus DP-FedAvg experiments on prepared public partitions.

Privacy accounting (honest scope):
- Mechanism: record-level (add/remove-one) DP-SGD with Poisson sampling, per-sample clipping
  and Gaussian noise, from native Opacus. Each client keeps ONE persistent PrivacyEngine for
  the whole federation, so its RDP accountant composes over every local step of every round
  (adaptive composition). The reported epsilon is per client, for that client's own records,
  against an adversary that sees all of that client's released model updates.
- Not covered: client-level DP, secure aggregation, the evaluator/test data, hyperparameter
  search, and Opacus `secure_mode=False` (research RNG, not cryptographically secure).
- Empirical membership-inference AUC is a measurement, never a privacy guarantee.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import (DigitMLP, evaluate_local, fedavg_weighted, model_bytes,
                                         train_local)


def _load_sites(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    sites = []
    for p in manifest["client_partitions"]:
        x, y = load_site_partition(data_dir / p["file"])
        xv, yv = load_site_partition(data_dir / p["val_file"])
        sites.append((x, y, xv, yv))
    with np.load(data_dir / manifest["evaluator_test"]["file"]) as t:
        test = (t["x"].astype(np.float32), t["y"].astype(np.int64))
    return manifest, sites, test


def _auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Mann-Whitney AUC: P(member score > nonmember score), ties count 1/2."""
    scores = np.concatenate([member_scores, nonmember_scores])
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    # average ranks for ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    n1, n0 = len(member_scores), len(nonmember_scores)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def membership_auc(model: DigitMLP, members: tuple, nonmembers: tuple, cap: int = 3000, seed: int = 0) -> float:
    """Loss-threshold membership inference (Yeom et al.) on the final global model."""
    rng = np.random.default_rng(seed)

    def losses(x, y):
        idx = rng.permutation(len(y))[:cap]
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(x[idx]).float())
            return -nn.functional.cross_entropy(out, torch.from_numpy(y[idx]).long(), reduction="none").numpy()
    return _auc(losses(*members), losses(*nonmembers))


def run_federation(data_dir: str | Path, *, mode: str, rounds: int, local_epochs: int, batch_size: int,
                   lr: float, seed: int, target_epsilon: float | None = None, delta: float = 1e-5,
                   max_grad_norm: float = 1.0) -> dict:
    if mode not in ("nonprivate", "opacus_dp"):
        raise ValueError("mode must be nonprivate or opacus_dp")
    data_dir = Path(data_dir)
    manifest, sites, (tx, ty) = _load_sites(data_dir)
    features, classes = manifest["num_features"], manifest["num_classes"]
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    global_model = DigitMLP(features, classes)
    start = time.perf_counter()
    engines, private = [], []
    sigmas, epsilons = [], []
    if mode == "opacus_dp":
        from opacus import PrivacyEngine
        from opacus.accountants.utils import get_noise_multiplier
        for (x, y, _, _) in sites:
            n = len(y)
            if not 0 < delta < 1 / n:
                raise ValueError("delta must be smaller than 1/n for every client")
            steps = rounds * local_epochs * math.ceil(n / batch_size)
            sigma = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
                                         sample_rate=batch_size / n, steps=steps, accountant="rdp")
            model = DigitMLP(features, classes)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            dl = DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
                            batch_size=batch_size, shuffle=True)
            engine = PrivacyEngine(accountant="rdp", secure_mode=False)
            pm, po, pdl = engine.make_private(module=model, optimizer=opt, data_loader=dl,
                                              noise_multiplier=sigma, max_grad_norm=max_grad_norm,
                                              poisson_sampling=True)
            engines.append(engine); private.append((pm, po, pdl)); sigmas.append(sigma)
    history = []
    for r in range(1, rounds + 1):
        states, sizes = [], []
        for c, (x, y, _, _) in enumerate(sites):
            if mode == "nonprivate":
                local = copy.deepcopy(global_model)
                train_local(local, x, y, epochs=local_epochs, batch_size=batch_size, lr=lr, seed=seed + c)
                states.append({k: v.detach().clone() for k, v in local.state_dict().items()})
            else:
                pm, po, pdl = private[c]
                pm._module.load_state_dict(global_model.state_dict())
                pm.train()
                for _ in range(local_epochs):
                    for xb, yb in pdl:
                        po.zero_grad()
                        nn.functional.cross_entropy(pm(xb), yb).backward()
                        po.step()
                states.append({k: v.detach().clone() for k, v in pm._module.state_dict().items()})
            sizes.append(len(y))
        global_model.load_state_dict(fedavg_weighted(states, sizes))
        test = evaluate_local(global_model, tx, ty)
        per_client = [evaluate_local(global_model, xv, yv)["accuracy"] for (_, _, xv, yv) in sites]
        history.append({"round": r, "test_accuracy": test["accuracy"], "test_loss": test["loss"],
                        "client_val_accuracy": per_client, "worst_client_val_accuracy": min(per_client)})
    if mode == "opacus_dp":
        epsilons = [float(e.accountant.get_epsilon(delta)) for e in engines]
        if any(not math.isfinite(e) or e > target_epsilon + 1e-3 for e in epsilons):
            raise RuntimeError("Native accountant exceeds target epsilon")
    members = (np.concatenate([s[0] for s in sites]), np.concatenate([s[1] for s in sites]))
    mia = membership_auc(global_model, members, (tx, ty), seed=seed)
    return {"mode": mode, "seed": seed, "rounds": rounds, "local_epochs": local_epochs,
            "batch_size": batch_size, "learning_rate": lr, "partition": manifest["partition"],
            "dataset": manifest["source"], "client_train_sizes": [len(s[1]) for s in sites],
            "target_epsilon": target_epsilon, "delta": delta if mode == "opacus_dp" else None,
            "max_grad_norm": max_grad_norm if mode == "opacus_dp" else None,
            "noise_multipliers": sigmas, "achieved_epsilon_per_client": epsilons,
            "final_test_accuracy": history[-1]["test_accuracy"], "final_test_loss": history[-1]["test_loss"],
            "final_worst_client_val_accuracy": history[-1]["worst_client_val_accuracy"],
            "membership_inference_auc": mia,
            "bytes_communicated": 2 * rounds * len(sites) * model_bytes(global_model.state_dict()),
            "wall_seconds": time.perf_counter() - start, "history": history}
