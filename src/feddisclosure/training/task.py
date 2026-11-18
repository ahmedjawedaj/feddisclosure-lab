"""Shared PyTorch model and local training math, with no federation access."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DigitMLP(nn.Module):
    def __init__(self, features: int = 64, classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, 32), nn.ReLU(), nn.Linear(32, classes))

    def forward(self, x):
        return self.network(x)


def loader(x: np.ndarray, y: np.ndarray, *, batch_size: int = 32, seed: int = 0, shuffle: bool = False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
                      batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_local(model: DigitMLP, x: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int,
                lr: float, seed: int) -> dict:
    torch.set_num_threads(1)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    total_loss, batches = 0.0, 0
    for epoch in range(epochs):
        for features, labels in loader(x, y, batch_size=batch_size, shuffle=True, seed=seed + epoch):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
    return {"train_loss": total_loss / max(1, batches), "num-examples": len(y)}


def evaluate_local(model: DigitMLP, x: np.ndarray, y: np.ndarray, batch_size: int = 64) -> dict:
    model.eval()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for features, labels in loader(x, y, batch_size=batch_size):
            output = model(features)
            total_loss += float(nn.functional.cross_entropy(output, labels, reduction="sum"))
            correct += int((output.argmax(dim=1) == labels).sum())
    return {"accuracy": correct / len(y), "loss": total_loss / len(y), "num-examples": len(y)}


def model_bytes(state: dict) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())


def fedavg_weighted(states: list[dict], sizes: list[int]) -> OrderedDict:
    """Transparent reference implementation for verifying Flower FedAvg's weighted mean."""
    if not states or len(states) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError("Nonempty updates and positive matching sample counts required")
    if any(list(s) != list(states[0]) for s in states):
        raise ValueError("Incompatible model keys")
    total = sum(sizes)
    result = OrderedDict()
    for name in states[0]:
        tensors = [s[name].detach().cpu() for s in states]
        if any(t.shape != tensors[0].shape or t.dtype != tensors[0].dtype for t in tensors):
            raise ValueError("Incompatible tensor shapes or dtypes")
        result[name] = sum((t.to(torch.float64) * (n / total) for t, n in zip(tensors, sizes))).to(tensors[0].dtype)
    return result
