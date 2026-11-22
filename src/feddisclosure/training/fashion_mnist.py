"""Fashion-MNIST (Zalando Research) as an optional second public dataset, 28x28 greyscale, 10 classes.

Follows the MNIST conventions of `feddisclosure.training.datasets`: only the 60,000-image train split
is used, the SHA-256 of every file is pinned in this module, a copy that does not match is refused,
and the raw files live under `data/source/fashion_mnist/`, which Git ignores. Nothing is redistributed.

Licence: MIT, Copyright (c) 2017 Zalando SE. See docs/data-provenance.md.

`synthetic_standin` is a deterministic fake with the same shapes. It exists ONLY so that unit tests can
run without the download. No result may be computed from it, and `load_fashion_mnist` never returns it.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import urllib.request

import numpy as np

BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
}
# Published by the dataset authors in the repository README (zalandoresearch/fashion-mnist).
PUBLISHED_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
}
LOCAL_DIR = Path(__file__).resolve().parents[3] / "data" / "source" / "fashion_mnist"
PROVENANCE = "Fashion-MNIST train split (60,000), SHA-256-verified local copy, MIT licence, Zalando Research"
CLASS_NAMES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
SIDE = 28


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_verified(directory: Path) -> bool:
    return all((directory / n).is_file() and _sha256(directory / n) == h for n, h in SHA256.items())


def fetch_fashion_mnist(directory: Path = LOCAL_DIR, *, base_url: str = BASE_URL, timeout: float = 120.0) -> Path:
    """Download the two train files and keep them only if the SHA-256 matches the pinned value."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        target = directory / name
        if target.is_file() and _sha256(target) == expected:
            continue
        with urllib.request.urlopen(base_url + name, timeout=timeout) as response:   # noqa: S310 (fixed http URL)
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"{name}: downloaded file does not match the pinned SHA-256, not saved")
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
    (directory / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in SHA256.items()))
    return directory


def parse_idx(images_gz: bytes, labels_gz: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode the two gzip IDX files into float32 images (n x 784, in [0, 1]) and int64 labels."""
    raw, lab = gzip.decompress(images_gz), gzip.decompress(labels_gz)
    if int.from_bytes(raw[0:4], "big") != 2051 or int.from_bytes(lab[0:4], "big") != 2049:
        raise ValueError("Bad IDX magic number")
    n, rows, cols = (int.from_bytes(raw[i:i + 4], "big") for i in (4, 8, 12))
    if (rows, cols) != (SIDE, SIDE) or int.from_bytes(lab[4:8], "big") != n or len(raw) != 16 + n * rows * cols:
        raise ValueError("IDX header does not describe 28x28 images with matching labels")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows * cols)
    labels = np.frombuffer(lab, dtype=np.uint8, offset=8).astype(np.int64)
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError("Labels outside 0 to 9")
    return images.astype(np.float32) / 255.0, labels


def load_fashion_mnist(directory: Path = LOCAL_DIR, *, download: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
    """Verified Fashion-MNIST train split. Raises if the files are missing or wrong, never substitutes data."""
    if not files_verified(directory):
        if not download:
            raise FileNotFoundError(f"Fashion-MNIST files with the pinned SHA-256 are not in {directory}. "
                                    "Run with download enabled on a machine with network access")
        fetch_fashion_mnist(directory)
    x, y = parse_idx((directory / "train-images-idx3-ubyte.gz").read_bytes(),
                     (directory / "train-labels-idx1-ubyte.gz").read_bytes())
    return x, y, PROVENANCE


def synthetic_standin(per_class: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    """Deterministic 28x28 images in 10 classes, built from a fixed template per class plus noise.
    FOR UNIT TESTS ONLY. It carries no information about real clothing images."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, SIDE)
    templates = []
    for c in range(10):
        gx, gy = np.meshgrid(grid, grid)
        phase = 2 * np.pi * (c + 1) / 11
        templates.append(0.5 + 0.35 * np.sin(6 * gx * (1 + c % 3) + phase) * np.cos(5 * gy * (1 + c % 4) - phase))
    x = np.concatenate([np.clip(t + rng.normal(0, 0.08, size=(per_class, SIDE, SIDE)), 0, 1) for t in templates])
    y = np.repeat(np.arange(10), per_class).astype(np.int64)
    order = rng.permutation(len(y))
    return x.reshape(len(y), SIDE * SIDE)[order].astype(np.float32), y[order], "SYNTHETIC STAND-IN, unit tests only"
