import pytest
pytest.importorskip("torch")
pytest.importorskip("sklearn")
import numpy as np
import torch
from feddisclosure.evaluation.update_observer import create_update_observation
from feddisclosure.attacks.gradient_inversion import reconstruct
from feddisclosure.training.datasets import prepare_partitions


def test_single_step_update_is_gradient_proxy_and_no_truth_exposed(tmp_path):
    prepare_partitions(tmp_path / "data", seed=32, limit=150)
    site = tmp_path / "data" / "institution-00" / "train.npz"
    contract = create_update_observation(site, tmp_path / "one", local_steps=1, seed=24)
    assert contract["gradient_proxy_relative_l2_error_vs_exact_initial_gradient"] < 1e-5
    obs = torch.load(tmp_path / "one" / "attacker" / "observation.pt", weights_only=True)
    assert "gradients" not in obs and "model_delta" in obs
    assert "original" not in str(list(obs))
    x, metrics = reconstruct(obs, iterations=40, seed=13)
    assert x.shape == (1, 64)
    assert metrics["gradient_target"].startswith("model_delta")
    assert metrics["gradient_matching_history"][-1]["gradient_match_sum_squared"] < metrics["gradient_matching_history"][0]["gradient_match_sum_squared"]


def test_multiple_steps_is_not_exact_initial_gradient(tmp_path):
    prepare_partitions(tmp_path / "data", seed=32, limit=150)
    site = tmp_path / "data" / "institution-00" / "train.npz"
    contract = create_update_observation(site, tmp_path / "four", local_steps=4, seed=24)
    assert contract["gradient_proxy_relative_l2_error_vs_exact_initial_gradient"] > 1e-5
