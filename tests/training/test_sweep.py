import pytest
from scripts.run_training_sweep import summary


def test_paired_sweep_summary():
    entries = [{"seed": 3, "partition": "iid", "clients": 3, "public_test_accuracy": 0.9, "worst_client_accuracy": 0.6},
               {"seed": 3, "partition": "label_skew", "clients": 3, "public_test_accuracy": 0.5, "worst_client_accuracy": 0.2}]
    result = summary(entries)
    assert result["paired_differences"][0]["iid_minus_skew_accuracy"] == pytest.approx(0.4)
    assert result["conditions"]["iid"]["public_accuracy_std"] is None
