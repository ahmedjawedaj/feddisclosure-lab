import pytest
from scripts.run_balanced_heterogeneity import aggregate, summary_for_run


def test_aggregate_requires_complete_matched_seeds():
    rows = [{"seed": 1, "partition": "iid", "metrics": {"test_accuracy": .4,
             "worst_client_accuracy": .2, "mean_client_label_tv": .1}},
            {"seed": 1, "partition": "dirichlet_balanced_0.5", "metrics": {"test_accuracy": .6,
             "worst_client_accuracy": .4, "mean_client_label_tv": .3}},
            {"seed": 2, "partition": "iid", "metrics": {"test_accuracy": .5,
             "worst_client_accuracy": .3, "mean_client_label_tv": .1}}]
    s = aggregate(rows, [1, 2], ["iid", "dirichlet_balanced_0.5"])
    assert s["completed_seeds"] == [1]
    assert s["by_partition"]["iid"]["n"] == 1
    assert s["paired_minus_iid_accuracy_pp"]["dirichlet_balanced_0.5"]["mean"] == pytest.approx(20)


def test_summary_rejects_unmatched_sizes(tmp_path):
    with pytest.raises(ValueError, match="sizes differ"):
        summary_for_run({}, {"client_partitions": [
            {"samples": 10, "val_samples": 2}, {"samples": 9, "val_samples": 2}]}, tmp_path)
