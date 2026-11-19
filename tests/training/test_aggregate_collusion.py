import numpy as np
import pytest
import torch

from feddisclosure.evaluation.aggregate_observer import create_aggregate_observations
from feddisclosure.attacks.gradient_inversion import reconstruct


def site_files(tmp_path):
    files = []
    for i in range(3):
        path = tmp_path / f"site{i}.npz"
        rng = np.random.default_rng(30 + i)
        np.savez_compressed(path, x=rng.random((32, 64), dtype=np.float32),
                            y=rng.integers(0, 10, size=32), source_idx=np.arange(32) + i * 100)
        files.append(path)
    return files


def test_aggregate_and_collusion_privileges(tmp_path):
    result = create_aggregate_observations(site_files(tmp_path), tmp_path / "study", seed=2)
    assert result["collusion_isolation_max_tensor_error"] < 2e-6
    observations = {}
    for case in result["cases"]:
        folder = tmp_path / "study" / case / "attacker"
        obs = torch.load(folder / "observation.pt", weights_only=True)
        assert "original" not in obs and "source_idx" not in obs and "x" not in obs
        assert not (folder / "target.npz").exists()
        assert obs["known_labels"] == torch.load(tmp_path / "study" / "individual" / "attacker" / "observation.pt", weights_only=True)["known_labels"]
        observations[case] = obs
    for key in observations["individual"]["model_delta"]:
        torch.testing.assert_close(observations["individual"]["model_delta"][key],
                                   observations["all_but_one_collusion"]["model_delta"][key],
                                   rtol=0, atol=2e-6)
    assert result["cases"]["aggregate_only"]["other_client_updates_visible_to_attack"] is False
    assert result["cases"]["all_but_one_collusion"]["other_client_updates_visible_to_attack"] is True
    # Explicitly permit aggregate-only target-proxy attack but never mislabel the role.
    _, report = reconstruct(observations["aggregate_only"], iterations=1)
    assert report["observer"] == "server_aggregate_model_delta_target_probe"


def test_bad_clients_and_overlap_fail_closed(tmp_path):
    files = site_files(tmp_path)
    with pytest.raises(ValueError, match="three distinct"):
        create_aggregate_observations([files[0]] * 3, tmp_path / "bad")
    copy = tmp_path / "copied.npz"
    copy.write_bytes(files[0].read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        create_aggregate_observations([files[0], files[1], copy], tmp_path / "overlap")
