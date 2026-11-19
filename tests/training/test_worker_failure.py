"""Tampered private partition must fail without producing a model update."""
import hashlib
from pathlib import Path
import subprocess
import sys
import torch
import pytest
pytest.importorskip("sklearn")
from feddisclosure.training.datasets import prepare_partitions
from feddisclosure.training.task import DigitMLP


def test_worker_rejects_modified_partition_before_release(tmp_path):
    manifest = prepare_partitions(tmp_path / "data", limit=150)
    site = manifest["client_partitions"][0]
    train = tmp_path / "data" / site["file"]
    val = tmp_path / "data" / site["val_file"]
    state = tmp_path / "state.pt"
    torch.save(DigitMLP().state_dict(), state)
    with train.open("ab") as file:
        file.write(b"UNTRUSTED_TAMPER")
    checkpoint = tmp_path / "out.pt"
    metrics = tmp_path / "metrics.json"
    run = subprocess.run([sys.executable, "-m", "feddisclosure.training.worker",
                          "--site-file", str(train), "--validation-file", str(val),
                          "--train-sha256", site["sha256"], "--validation-sha256", site["val_sha256"],
                          "--weights-in", str(state), "--weights-out", str(checkpoint),
                          "--metrics-out", str(metrics), "--features", "64", "--classes", "10",
                          "--epochs", "1", "--batch-size", "32", "--lr", "0.15", "--seed", "15"],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode != 0
    assert "dataset mismatch" in run.stderr
    assert not checkpoint.exists() and not metrics.exists()
