import numpy as np
import pytest
import torch
from feddisclosure.training.datasets import prepare_partitions
from feddisclosure.evaluation.update_observer import create_update_observation
from feddisclosure.attacks.gradient_inversion import reconstruct, run_attack


def test_multirecord_label_privilege_and_truth_boundary(tmp_path):
    manifest = prepare_partitions(tmp_path / 'data', limit=240)
    site = tmp_path / 'data' / manifest['client_partitions'][0]['file']
    contract = create_update_observation(site, tmp_path / 'case', batch_size=8, local_steps=4, seed=19)
    obs = torch.load(tmp_path / 'case' / 'attacker' / 'observation.pt', weights_only=True)
    assert contract['distinct_training_records'] == 8
    assert contract['fixed_local_batch_repeated']
    assert 'known_labels' not in obs
    assert not any(k in obs for k in ('x', 'original', 'source_idx', 'y', 'ground_truth'))
    assert (tmp_path / 'case' / 'evaluator_only' / 'original.npz').exists()
    with pytest.raises(ValueError, match='observed labels'):
        reconstruct(obs, iterations=1)
    privileged = create_update_observation(site, tmp_path / 'known', batch_size=8, local_steps=1,
                                             seed=19, disclose_labels=True)
    assert privileged['labels_disclosed_to_attacker']
    attack = run_attack(tmp_path / 'known' / 'attacker' / 'observation.pt', tmp_path / 'known' / 'attacker', iterations=1)
    assert attack['batch_size'] == 8
    with np.load(tmp_path / 'known' / 'attacker' / 'recovered.npz') as recovery:
        assert recovery['x'].shape == (8, 64)


def test_multirecord_invalid_batch_size(tmp_path):
    manifest = prepare_partitions(tmp_path / 'data', limit=150)
    site = tmp_path / 'data' / manifest['client_partitions'][0]['file']
    with pytest.raises(ValueError):
        create_update_observation(site, tmp_path / 'case', batch_size=7)
