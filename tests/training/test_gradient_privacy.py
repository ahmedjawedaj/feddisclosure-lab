import pytest
pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("skimage")
import numpy as np
import torch
from feddisclosure.attacks.gradient_inversion import infer_single_label, reconstruct, run_attack
from feddisclosure.evaluation.gradient import score_reconstruction
from feddisclosure.evaluation.gradient_observer import create_gradient_observation
from feddisclosure.training.datasets import prepare_partitions


def test_gradient_observer_excludes_ground_truth(tmp_path):
    prepare_partitions(tmp_path / "digits", seed=51, limit=150)
    path = tmp_path / "digits" / "institution-00" / "train.npz"
    contract = create_gradient_observation(path, tmp_path / "case", seed=12)
    observed = torch.load(tmp_path / "case" / "attacker" / "observation.pt", weights_only=True)
    assert set(observed) == {"schema_version", "observer", "batch_size", "features", "classes", "model_state", "gradients"}
    assert "original" not in str(contract["attacker_receives"])
    assert not any("original" in str(k) for k in observed)
    true_label = int(np.load(tmp_path / "case" / "evaluator_only" / "original.npz")["y"][0])
    assert infer_single_label(observed["gradients"]) == true_label


def test_multi_example_requires_known_labels(tmp_path):
    prepare_partitions(tmp_path / "digits", limit=150)
    path = tmp_path / "digits" / "institution-00" / "train.npz"
    with pytest.raises(ValueError, match="known-label"):
        create_gradient_observation(path, tmp_path / "case", batch_size=8)


def test_attacker_rejects_ground_truth_in_observation(tmp_path):
    observation = tmp_path / "observation.pt"
    torch.save({"x": torch.zeros(1, 64)}, observation)
    with pytest.raises(ValueError, match="Ground-truth"):
        run_attack(observation, tmp_path / "out", iterations=1)


def test_evaluator_optimal_matching_and_metrics():
    original = np.zeros((2, 64), dtype=np.float32)
    original[0, :12] = 1
    original[1, 12:30] = 1
    score = score_reconstruction(original, original[::-1].copy())
    assert score["mean_mse"] == 0
    assert score["mean_psnr_db"] >= 119
    assert score["reconstruction_rate_mse_below_0_05"] == 1


def test_controlled_inversion_reduces_gradient_residual(tmp_path):
    prepare_partitions(tmp_path / "digits", limit=150)
    path = tmp_path / "digits" / "institution-00" / "train.npz"
    create_gradient_observation(path, tmp_path / "case", seed=31)
    observed = torch.load(tmp_path / "case" / "attacker" / "observation.pt", weights_only=True)
    image, result = reconstruct(observed, iterations=60, seed=42)
    assert image.shape == (1, 64)
    assert 0 <= image.min() <= image.max() <= 1
    assert result["gradient_matching_history"][-1]["gradient_match_sum_squared"] < result["gradient_matching_history"][0]["gradient_match_sum_squared"]
