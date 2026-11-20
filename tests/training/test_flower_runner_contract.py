"""Tests runner failure recording, NOT native Flower functionality."""
from pathlib import Path
import json
import subprocess
import sys
import textwrap

import pytest
from scripts import run_flower_experiment as runner


def make_root(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text(json.dumps({
        "seed": 100, "partition": "iid", "clients": 3,
        "source_limit": 200, "source": "sklearn_digits", "num_features": 64,
        "num_classes": 10}))
    (tmp_path / "configs/cfg.yaml").write_text(textwrap.dedent("""
        seed: 100
        dataset: sklearn_digits
        partition: iid
        clients: 3
        source_limit: 200
        rounds: 1
        local_epochs: 1
        batch_size: 16
        learning_rate: 0.1
    """))
    return tmp_path


def test_runner_logs_timeout_as_failure(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "find_spec", lambda name: object())
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1200, output=b"partial stdout", stderr=b"timeout stderr")
    monkeypatch.setattr(runner.subprocess, "run", timeout)
    monkeypatch.setattr(sys, "argv", ["run", "--config", "configs/cfg.yaml",
                                   "--data-dir", "data", "--output-dir", "result"])
    with pytest.raises(SystemExit, match="not a verified run"):
        runner.main()
    record = json.loads((root / "result/flower_run.json").read_text())
    assert record["timed_out"] and record["exit_code"] == 124
    assert (root / "result/flower_stdout.log").read_text() == "partial stdout"


def test_runner_refuses_stale_checkpoints(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    (root / "result").mkdir()
    (root / "result/final_model.pt").write_bytes(b"stale")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "find_spec", lambda name: object())
    monkeypatch.setattr(sys, "argv", ["run", "--config", "configs/cfg.yaml",
                                   "--data-dir", "data", "--output-dir", "result"])
    with pytest.raises(SystemExit, match="Refusing to reuse"):
        runner.main()
