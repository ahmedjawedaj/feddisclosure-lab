import numpy as np
import pytest
import torch

from feddisclosure.evaluation.aggregate_observer import create_aggregate_observations
from feddisclosure.attacks.gradient_inversion import reconstruct
from scripts.run_joint_aggregate_attack import score_joint


def _sites(root):
    files = []
    for index in range(3):
        rng = np.random.default_rng(index + 100)
        filename = root / f'site{index}.npz'
        np.savez_compressed(filename, x=rng.random((32, 64), dtype=np.float32),
                            y=rng.integers(0, 10, size=32),
                            source_idx=np.arange(32) + index * 100)
        files.append(filename)
    return files


def test_joint_oracle_contract_and_tensor_identity(tmp_path):
    setup = tmp_path / 'setup'
    report = create_aggregate_observations(_sites(tmp_path), setup, seed=5, joint_oracle=True)
    oracle = report['cases']['aggregate_joint_oracle']
    assert oracle['pooled_initial_gradient_max_error'] < 1e-5
    assert oracle['privileged_all_contributor_labels'] is True
    folder = setup / 'aggregate_joint_oracle' / 'attacker'
    assert not (folder / 'pooled.npz').exists()
    observation = torch.load(folder / 'observation.pt', map_location='cpu', weights_only=True)
    assert len(observation['known_labels']) == 24
    assert not any(k in observation for k in ('x','images','source_idx','ground_truth','site_ids'))
    estimated, metadata = reconstruct(observation, iterations=1)
    assert estimated.shape == (24,64)
    assert metadata['observer'] == 'server_aggregate_model_delta_joint_oracle'


def test_evaluator_only_matching_and_validation(tmp_path):
    pooled = tmp_path / 'pooled.npz'
    target = tmp_path / 'target.npz'
    recovered = tmp_path / 'recovered.npz'
    x = np.arange(24*64, dtype=np.float32).reshape(24,64) / (24*64)
    y = np.arange(24)%10
    np.savez_compressed(pooled, x=x, y=y)
    np.savez_compressed(target, x=x[:8], y=y[:8])
    np.savez_compressed(recovered, x=x[::-1])
    # Order changes require label alignment. A permutation without permuted
    # labels must not receive an artificial zero reconstruction score.
    score = score_joint(pooled, target, recovered)
    assert score['reconstruction_count'] == 24
    assert score['pooled_label_matched_mse'] >= 0
    np.savez_compressed(recovered, x=np.full((24,64), np.inf))
    with pytest.raises(ValueError, match='Nonfinite'):
        score_joint(pooled,target,recovered)
