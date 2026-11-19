import pytest
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from feddisclosure.evaluation.minibatch_observer import create_minibatch_update_observation
from feddisclosure.attacks.gradient_inversion import reconstruct


def make_digits(path):
    rng = np.random.default_rng(100)
    x = rng.random((100, 64), dtype=np.float32)
    y = rng.integers(0, 10, size=100)
    np.savez_compressed(path, x=x, y=y)


def test_changing_minibatches_are_disjoint_and_do_not_leak_truth(tmp_path):
    source = tmp_path / "site.npz"
    make_digits(source)
    out = tmp_path / "experiment"
    contract = create_minibatch_update_observation(source, out, batch_size=8, local_steps=4,
                                                   seed=7, schedule_seed=1, disclose_labels=True)
    attack = torch.load(out / "attacker" / "observation.pt", map_location="cpu", weights_only=True)
    original = np.load(out / "evaluator_only" / "original.npz", allow_pickle=False)
    assert len(set(original["source_idx"].tolist())) == 32
    assert contract["distinct_training_records"] == 32
    assert contract["each_record_used_exactly_once"] is True
    assert attack["batch_size"] == 32
    assert attack["known_labels"] == original["y"].tolist()
    assert "schedule" not in attack and "source_idx" not in attack and "x" not in attack
    assert not any(isinstance(v, np.ndarray) for v in attack.values())
    assert all(torch.isfinite(v).all() for v in attack["model_delta"].values())
    assert contract["gradient_proxy_relative_l2_error_vs_exact_initial_full_dataset_gradient"] > 0
    assert (out / "attacker" / "original.npz").exists() is False
    assert (out / "evaluator_only" / "observation.pt").exists() is False


def test_same_targets_different_minibatch_order(tmp_path):
    source = tmp_path / "site.npz"
    make_digits(source)
    reports = []
    deltas = []
    for order in (0, 1):
        folder = tmp_path / str(order)
        reports.append(create_minibatch_update_observation(source, folder, batch_size=8,
                       local_steps=4, seed=12, schedule_seed=order))
        obs = torch.load(folder / "attacker" / "observation.pt", weights_only=True)
        deltas.append(obs["model_delta"])
        if order:
            one = np.load(tmp_path / "0" / "evaluator_only" / "original.npz")
            two = np.load(folder / "evaluator_only" / "original.npz")
            np.testing.assert_array_equal(one["x"], two["x"])
    assert any(not torch.equal(deltas[0][k], deltas[1][k]) for k in deltas[0])
    assert reports[0]["observation_sha256"] != reports[1]["observation_sha256"]


def test_one_step_update_equals_exact_initial_full_gradient(tmp_path):
    source = tmp_path / "site.npz"
    make_digits(source)
    report = create_minibatch_update_observation(source, tmp_path / "one", batch_size=8,
                                                 local_steps=1, seed=4)
    assert report["gradient_proxy_relative_l2_error_vs_exact_initial_full_dataset_gradient"] < 5e-6


def test_invalid_and_unknown_labels_fail_closed(tmp_path):
    source = tmp_path / "site.npz"
    make_digits(source)
    for args in ({"batch_size": 32, "local_steps": 4}, {"batch_size": 3},
                 {"learning_rate": float("nan")}, {"local_steps": 0}):
        with pytest.raises(ValueError):
            create_minibatch_update_observation(source, tmp_path / "bad", **args)
    folder = tmp_path / "unknown"
    create_minibatch_update_observation(source, folder, batch_size=8, local_steps=2,
                                        disclose_labels=False)
    obs = torch.load(folder / "attacker" / "observation.pt", weights_only=True)
    assert "known_labels" not in obs
    with pytest.raises(ValueError, match="requires explicitly observed labels"):
        reconstruct(obs, iterations=1)


def test_deterministic_update_for_same_data_and_schedule_seeds(tmp_path):
    source = tmp_path / "site.npz"
    make_digits(source)
    observations = []
    for name in ("a", "b"):
        folder = tmp_path / name
        create_minibatch_update_observation(source, folder, batch_size=8, local_steps=4,
                                            seed=15, schedule_seed=9)
        observations.append(torch.load(folder / "attacker" / "observation.pt", weights_only=True))
    for key in observations[0]["model_delta"]:
        assert torch.equal(observations[0]["model_delta"][key], observations[1]["model_delta"][key])
