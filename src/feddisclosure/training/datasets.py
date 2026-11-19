"""Explicit, immutable public digit partitions for cross-silo experiments.

Preparation is a privileged setup step; federated client code only reads its own file.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(name: str, root: Path):
    if name == "sklearn_digits":
        from sklearn.datasets import load_digits
        x, y = load_digits(return_X_y=True)
        return (np.asarray(x, dtype=np.float32) / 16.0,
                np.asarray(y, dtype=np.int64), "sklearn.load_digits, 8x8 public digits")
    if name == "mnist":
        from torchvision.datasets import MNIST
        from torchvision.transforms.functional import pil_to_tensor
        dataset = MNIST(root=str(root / "source_cache"), train=True, download=True)
        images = dataset.data.numpy().reshape(len(dataset), 784).astype(np.float32) / 255.0
        return images, dataset.targets.numpy().astype(np.int64), "torchvision MNIST train split"
    raise ValueError(f"Unknown dataset: {name}")


def _dirichlet_groups(train_idx, labels, clients, rng, alpha: float = 0.5):
    """Class-conditioned site allocation with minimum train/validation support.

    This is intentionally a data-preparation/evaluator helper, not a client API.
    It returns index partitions; site label statistics remain evaluator-only.
    """
    for _ in range(100):
        groups = [[] for _ in range(clients)]
        for label in np.unique(labels[train_idx]):
            indices = rng.permutation(train_idx[labels[train_idx] == label])
            proportions = rng.dirichlet([alpha] * clients)
            pieces = np.split(indices, (np.cumsum(proportions)[:-1] * len(indices)).astype(int))
            for i, piece in enumerate(pieces):
                groups[i].extend(piece.tolist())
        if all(len(g) >= 20 for g in groups):
            return [rng.permutation(np.asarray(g, dtype=np.int64)) for g in groups]
    raise RuntimeError("Dirichlet split failed to give each institution enough examples")


def prepare_partitions(output_dir: str | Path, *, dataset: str = "sklearn_digits", clients: int = 3,
                       partition: str = "iid", seed: int = 20261118, limit: int | None = None) -> dict:
    """Split once, write one NPZ per site and a separate evaluator-only test set.

    Existing output refuses overwrite to prevent silent changes to an experiment.
    This setup function may see the full public dataset. It must never run in clients.
    """
    if clients < 1 or partition not in ("iid", "label_skew", "dirichlet_0.5"):
        raise ValueError("clients >= 1 and partition in {iid,label_skew,dirichlet_0.5} required")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing partition dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    x, y, provenance = _source(dataset, output)
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(y))
    if limit is not None:
        if limit < clients * 12 or limit > len(index):
            raise ValueError("limit must hold >= 12 samples/client and not exceed source size")
        index = index[:limit]
    # Disjoint, deterministic train/test split; the test set is not copied into site folders.
    test_count = max(clients, int(0.2 * len(index)))
    test_idx, train_idx = index[:test_count], index[test_count:]
    if partition == "label_skew":
        # Class-sorted chunks produce severe label skew; client/sample sizes stay similar.
        train_idx = train_idx[np.argsort(y[train_idx], kind="stable")]
    else:
        train_idx = rng.permutation(train_idx)
    groups = _dirichlet_groups(train_idx, y, clients, rng) if partition == "dirichlet_0.5" else np.array_split(train_idx, clients)
    if any(len(g) == 0 for g in groups):
        raise ValueError("Empty client partition")
    manifest = {"schema_version": 2, "source": dataset, "provenance": provenance,
                "seed": seed, "partition": partition, "clients": clients, "source_limit": limit,
                "num_features": int(x.shape[1]), "num_classes": int(y.max() + 1),
                "dirichlet_alpha": 0.5 if partition == "dirichlet_0.5" else None,
                "client_partitions": [], "evaluator_test": {},
                "raw_data_policy": "only client-owned NPZ files in training; evaluator test is separate"}
    evaluator_histograms = []
    for i, group in enumerate(groups):
        split = max(1, len(group) // 5)
        train_group, val_group = group[split:], group[:split]
        path = output / f"institution-{i:02d}" / "train.npz"
        val_path = output / f"institution-{i:02d}" / "val.npz"
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, x=x[train_group], y=y[train_group], source_idx=train_group)
        np.savez_compressed(val_path, x=x[val_group], y=y[val_group], source_idx=val_group)
        manifest["client_partitions"].append({"client_id": i, "file": str(path.relative_to(output)),
                                               "val_file": str(val_path.relative_to(output)),
                                               "samples": len(train_group), "val_samples": len(val_group),
                                               "sha256": _hash(path), "val_sha256": _hash(val_path)})
        evaluator_histograms.append({"client_id": i, "label_histogram":
                                     np.bincount(y[train_group], minlength=manifest["num_classes"]).tolist()})
    test_path = output / "evaluator_only" / "test.npz"
    test_path.parent.mkdir(parents=True)
    np.savez_compressed(test_path, x=x[test_idx], y=y[test_idx], source_idx=test_idx)
    manifest["evaluator_test"] = {"file": str(test_path.relative_to(output)), "samples": len(test_idx),
                                  "sha256": _hash(test_path)}
    # Ground-truth label distributions are not available to the coordinator.
    # They are used only to characterize a public-data partition in evaluation.
    (output / "evaluator_only" / "distribution_stats.json").write_text(
        json.dumps({"client_label_histograms": evaluator_histograms}, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_site_partition(path: str | Path):
    """Return *only* the site training dataset, never an evaluator truth set."""
    with np.load(path, allow_pickle=False) as arrays:
        return arrays["x"].copy(), arrays["y"].copy()
