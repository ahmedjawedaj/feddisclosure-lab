import pytest
pytest.importorskip("torch")
pytest.importorskip("sklearn")
import numpy as np
import torch
from feddisclosure.attacks.membership import confidence_scores, attack_outputs
from feddisclosure.evaluation.membership import score_membership
from feddisclosure.evaluation.model_output_observer import create_output_observations
from feddisclosure.training.datasets import prepare_partitions
from feddisclosure.training.task import DigitMLP


def test_membership_scores_and_operating_points():
    truth = np.array([1, 1, 0, 0])
    pred = np.array([0.8, 0.9, 0.1, 0.2])
    score = score_membership(pred, truth)
    assert score["roc_auc"] == 1.0
    assert score["tpr_at_fpr"]["0.01"] == 1.0
    assert score["attack_advantage"] == 1.0
    assert score_membership(-pred, truth)["roc_auc"] == 0.0


def test_attacker_refuses_leaked_membership_truth(tmp_path):
    observation = tmp_path / "observations.npz"
    np.savez_compressed(observation, probabilities=np.array([[0.5, 0.5]]), membership=[1])
    with pytest.raises(ValueError, match="probabilities only"):
        attack_outputs(observation, tmp_path / "out")


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        confidence_scores(np.array([[0.2, 0.1]]))


def test_observer_creates_disjoint_attack_and_truth_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    manifest = prepare_partitions(data_dir, seed=51, limit=300)
    torch.manual_seed(7)
    path = tmp_path / "checkpoint.pt"
    torch.save(DigitMLP().state_dict(), path)
    contract = create_output_observations(path, [data_dir / p["file"] for p in manifest["client_partitions"]],
                                          data_dir / manifest["evaluator_test"]["file"], tmp_path / "experiment", per_class=2)
    assert contract["member_count"] == contract["nonmember_count"]
    with np.load(tmp_path / "experiment" / "attacker" / "model_predictions.npz") as observed:
        assert observed.files == ["probabilities"]
        assert observed["probabilities"].shape == (contract["query_count"], 10)
    with np.load(tmp_path / "experiment" / "evaluator_only" / "membership_labels.npz") as truth:
        assert sorted(set(truth["membership"])) == [0, 1]
