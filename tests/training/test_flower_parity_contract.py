"""Contract tests only. These fixtures are NOT native Flower runs."""
import hashlib
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from scripts.verify_flower_parity import certify


def test_parity_checker_accepts_matching_fixtures_and_rejects_tampering(tmp_path):
    flower = tmp_path / "flower"
    flower.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version":2}')
    config = {"seed": 1, "dataset": "sklearn_digits", "partition": "iid", "clients": 3,
              "rounds": 1, "local_epochs": 1, "batch_size": 16, "learning_rate": 0.15,
              "source_limit": 600, "algorithm": "fedavg", "proximal_mu": 0.0}
    torch.save({"layer": torch.tensor([1., 2.])}, flower / "final_model.pt")
    checkpoint = tmp_path / "reference.pt"
    torch.save({"layer": torch.tensor([1., 2.])}, checkpoint)
    reference_path = tmp_path / "ref.json"
    ref = {"engine": "torch_independent_process_reference_NOT_Flower", "config": config,
           "rounds": [{}], "final_checkpoint": {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
           "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    reference_path.write_text(json.dumps(ref))
    record = {"engine": "Flower ServerApp/ClientApp", "exit_code": 0, "config": config,
              "final_model_exists": True, "summary_exists": True}
    run_path = flower / "flower_run.json"
    run_path.write_text(json.dumps(record))
    (flower / "flower_summary.json").write_text(json.dumps({"engine": "Flower ServerApp/ClientApp",
                                                        "clients_requested": 3, "rounds": 1}))
    args = (flower, reference_path, checkpoint, manifest)
    assert certify(*args)["verified_numerical_parity"]
    torch.save({"layer": torch.tensor([1., 2.01])}, flower / "final_model.pt")
    report = certify(*args)
    assert not report["verified_numerical_parity"]
    assert not report["layers"]["layer"]["within_tolerance"]
    record["exit_code"] = 1
    run_path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="did not succeed"):
        certify(*args)
    record["exit_code"] = 0
    record["config"] = {**config, "seed": 2}
    run_path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="Not matched training"):
        certify(*args)


def test_parity_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        certify(Path("unused"), Path("unused"), Path("unused"), Path("unused"), atol=float("nan"))
