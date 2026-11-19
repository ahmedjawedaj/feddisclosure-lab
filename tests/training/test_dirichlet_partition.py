import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from feddisclosure.training.datasets import prepare_partitions


def test_moderate_noniid_partitions_are_disjoint_and_truth_stays_evaluator_only(tmp_path):
    manifest = prepare_partitions(tmp_path / "sample", dataset="sklearn_digits", clients=3,
                                  partition="dirichlet_0.5", seed=241, limit=900)
    assert manifest["schema_version"] == 2 and manifest["dirichlet_alpha"] == 0.5
    original = set()
    for site in manifest["client_partitions"]:
        for name in (site["file"], site["val_file"]):
            with np.load(tmp_path / "sample" / name, allow_pickle=False) as f:
                indices = f["source_idx"].tolist()
            assert not original.intersection(indices)
            original.update(indices)
        assert "label_histogram" not in site
    with np.load(tmp_path / "sample/evaluator_only/test.npz", allow_pickle=False) as f:
        assert not original.intersection(f["source_idx"].tolist())
    assert len(original) == 720
    assert (tmp_path / "sample/evaluator_only/distribution_stats.json").is_file()
    assert "label_histogram" not in (tmp_path / "sample/manifest.json").read_text()


def test_moderate_partition_repeatability(tmp_path):
    data = []
    for d in ("a", "b"):
        m = prepare_partitions(tmp_path / d, clients=3, partition="dirichlet_0.5", seed=9, limit=900)
        indices = []
        for site in m["client_partitions"]:
            with np.load(tmp_path / d / site["file"], allow_pickle=False) as f:
                indices.append(f["source_idx"].tolist())
        data.append(indices)
    assert data[0] == data[1]
