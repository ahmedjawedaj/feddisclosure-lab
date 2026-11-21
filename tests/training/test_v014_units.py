"""Unit tests for v0.14 helpers. They check logic, not privacy guarantees."""
import gzip
import hashlib

import numpy as np
import pytest

from feddisclosure.training import datasets
from feddisclosure.training.privacy_utility import _auc


def test_auc_perfect_chance_and_ties():
    assert _auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert _auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert _auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_secagg_bound_scales_with_max_weight_and_total_samples():
    from scripts.verify_secagg_native import per_round_bound, QUANT_STEP
    assert per_round_bound(3, 1000, 1000) == pytest.approx(3 * QUANT_STEP)
    assert per_round_bound(3, 1000, 200) == pytest.approx(per_round_bound(3, 1000, 1000) / 5)
    assert per_round_bound(6, 1000, 1000) == pytest.approx(2 * per_round_bound(3, 1000, 1000))


def test_flower_literal_grammar(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from scripts.run_flower_docker import _flower_literal
    assert _flower_literal(True) == "true"
    assert _flower_literal(False) == "false"
    assert _flower_literal(3) == "3"
    assert _flower_literal(0.1) == "0.1"
    assert _flower_literal("/artifacts") == '"/artifacts"'


def _write_idx(directory, images, labels):
    n = len(labels)
    with gzip.open(directory / "train-images-idx3-ubyte.gz", "wb") as f:
        f.write((2051).to_bytes(4, "big") + n.to_bytes(4, "big") + (2).to_bytes(4, "big") * 2 + images.tobytes())
    with gzip.open(directory / "train-labels-idx1-ubyte.gz", "wb") as f:
        f.write((2049).to_bytes(4, "big") + n.to_bytes(4, "big") + labels.tobytes())


def test_mnist_loader_accepts_only_pinned_checksums(tmp_path, monkeypatch):
    images = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
    labels = np.array([1, 2, 3], dtype=np.uint8)
    _write_idx(tmp_path, images, labels)
    # Unpinned files must be refused.
    assert datasets._load_mnist_local(tmp_path) is None
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.gz")}
    monkeypatch.setattr(datasets, "MNIST_SHA256", pins)
    x, y = datasets._load_mnist_local(tmp_path)
    assert x.shape == (3, 4) and x.max() <= 1.0 and y.tolist() == [1, 2, 3]
    # Tampering after pinning is refused again.
    (tmp_path / "train-labels-idx1-ubyte.gz").write_bytes(b"tampered")
    assert datasets._load_mnist_local(tmp_path) is None
