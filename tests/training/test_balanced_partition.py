"""Balanced Dirichlet: no held-out leakage, equal site budgets, deterministic."""
import json
import numpy as np
import pytest
from feddisclosure.training.datasets import prepare_partitions


@pytest.mark.parametrize("part", ["iid", "label_skew", "dirichlet_balanced_0.1", "dirichlet_balanced_0.5", "dirichlet_balanced_10"])
def test_balanced_disjoint_and_equal_budgets(tmp_path, part):
    root = tmp_path / part
    m = prepare_partitions(root, clients=3, partition=part, seed=241, limit=900)
    sizes = [s["samples"] for s in m["client_partitions"]]
    val_sizes = [s["val_samples"] for s in m["client_partitions"]]
    assert max(sizes) - min(sizes) <= 1
    assert len(set(val_sizes)) == 1
    all_idx = set()
    for site in m["client_partitions"]:
        for name in (site["file"], site["val_file"]):
            with np.load(root / name, allow_pickle=False) as d:
                ids = d["source_idx"].tolist()
            assert not (all_idx & set(ids))
            all_idx.update(ids)
    with np.load(root / m["evaluator_test"]["file"], allow_pickle=False) as d:
        assert not (all_idx & set(d["source_idx"].tolist()))
    assert len(all_idx) == 720
    assert "label_histogram" not in (root / "manifest.json").read_text()
    assert "label_histogram" in (root / "evaluator_only/distribution_stats.json").read_text()


def test_matched_test_and_repeatable_balanced(tmp_path):
    samples = []
    for part in ("label_skew", "dirichlet_balanced_0.5", "dirichlet_balanced_10", "dirichlet_balanced_0.5"):
        directory = tmp_path / f"run-{len(samples)}"
        m = prepare_partitions(directory, clients=3, partition=part, seed=241, limit=900)
        with np.load(directory / "evaluator_only/test.npz", allow_pickle=False) as d:
            test = d["source_idx"].copy()
        with np.load(directory / m["client_partitions"][0]["file"], allow_pickle=False) as d:
            train = d["source_idx"].copy()
        samples.append((test, train))
    assert all(np.array_equal(samples[0][0], x[0]) for x in samples)
    assert np.array_equal(samples[1][1], samples[3][1])


def test_balanced_allocation_rejects_bad_alpha():
    from feddisclosure.training.datasets import _balanced_dirichlet_groups
    with pytest.raises(ValueError):
        _balanced_dirichlet_groups(np.arange(20), np.zeros(20, dtype=int), 3, np.random.default_rng(0), -1)
