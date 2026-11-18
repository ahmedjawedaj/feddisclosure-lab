import os
from pathlib import Path
import pytest
pytest.importorskip("torch")
pytest.importorskip("sklearn")
from feddisclosure.training.offline_runner import run_reference


@pytest.mark.slow
@pytest.mark.parametrize("partition", ["iid", "label_skew"])
def test_independent_worker_training(tmp_path, partition):
    config = Path(__file__).resolve().parents[2] / f"configs/training/fedavg_digits_{'iid' if partition == 'iid' else 'non_iid'}.yaml"
    data_dir = tmp_path / "data"
    result = run_reference(config, data_dir, tmp_path / "result.json")
    assert result["engine"].endswith("NOT_Flower")
    assert len(result["rounds"]) == 2
    assert len(result["rounds"][-1]["client_metrics"]) == 3
    assert all(c["worker_pid"] != os.getpid() for r in result["rounds"] for c in r["client_metrics"])
    assert len({c["worker_pid"] for c in result["rounds"][0]["client_metrics"]}) == 3
    assert all(0 <= r["public_test"]["accuracy"] <= 1 for r in result["rounds"])
    assert result["rounds"][-1]["communication_model_bytes"] > 0
    assert 0 <= result["final_global_worst_client_validation_accuracy"] <= 1
    assert len(result["final_global_per_client_validation"]) == 3
    assert "raw" not in result["actual_observations"]
