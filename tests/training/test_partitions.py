import json
import numpy as np
import pytest
pytest.importorskip("sklearn")
from feddisclosure.training.datasets import prepare_partitions, load_site_partition


def test_partitions_are_disjoint_reproducible_and_private(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    a = prepare_partitions(one, clients=3, seed=8, limit=300)
    b = prepare_partitions(two, clients=3, seed=8, limit=300)
    assert [x["sha256"] for x in a["client_partitions"]] == [x["sha256"] for x in b["client_partitions"]]
    groups = []
    for part in a["client_partitions"]:
        x, y = load_site_partition(one / part["file"])
        assert len(x) == part["samples"]
        assert x.shape[1] == 64
        with np.load(one / part["file"]) as z:
            groups.extend(z["source_idx"].tolist())
        with np.load(one / part["val_file"]) as z:
            groups.extend(z["source_idx"].tolist())
    with np.load(one / a["evaluator_test"]["file"]) as z:
        groups.extend(z["source_idx"].tolist())
    assert len(groups) == len(set(groups)) == 300
    assert not list(one.glob("*.npz"))  # raw records are site-specific or evaluator-only
    with pytest.raises(FileExistsError):
        prepare_partitions(one, clients=3, seed=8, limit=300)


def test_non_iid_partition_more_skewed_than_iid(tmp_path):
    a = prepare_partitions(tmp_path / "iid", clients=3, seed=4, limit=600, partition="iid")
    b = prepare_partitions(tmp_path / "skew", clients=3, seed=4, limit=600, partition="label_skew")
    def avg_classes(m):
        return sum(sum(v > 0 for v in p["label_histogram"]) for p in m["client_partitions"]) / 3
    assert avg_classes(b) < avg_classes(a)
