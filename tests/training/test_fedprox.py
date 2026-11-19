from pathlib import Path
import numpy as np
import pytest
import torch
from feddisclosure.training.task import DigitMLP, train_local
from feddisclosure.training.offline_runner import load_config


def test_fedprox_zero_exactly_matches_fedavg_and_positive_mu_reduces_drift():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(128, 64)).astype(np.float32)
    y = rng.integers(0, 10, size=128)
    torch.manual_seed(400)
    models = [DigitMLP() for _ in range(3)]
    for m in models[1:]:
        m.load_state_dict(models[0].state_dict())
    start = {k: v.clone() for k, v in models[0].state_dict().items()}
    kwargs = dict(epochs=4, batch_size=16, lr=0.15, seed=6)
    train_local(models[0], x, y, **kwargs)
    train_local(models[1], x, y, **kwargs, proximal_mu=0.0)
    train_local(models[2], x, y, **kwargs, proximal_mu=3.0)
    assert all(torch.equal(models[0].state_dict()[k], models[1].state_dict()[k]) for k in start)
    distance = lambda m: sum(float((m.state_dict()[k] - start[k]).square().sum()) for k in start)
    assert distance(models[2]) < distance(models[0])
    for invalid in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonnegative'):
            train_local(models[0], x, y, **kwargs, proximal_mu=invalid)


def test_bad_fedprox_configuration(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('''id: test\nseed: 1\ndataset: sklearn_digits\npartition: iid\nclients: 3\nrounds: 2\nlocal_epochs: 1\nbatch_size: 16\nlearning_rate: 0.1\nalgorithm: fedavg\nproximal_mu: 0.3\n''')
    with pytest.raises(ValueError, match='Invalid federation algorithm'):
        load_config(path)
    path.write_text(path.read_text().replace('algorithm: fedavg', 'algorithm: fedprox'))
    assert load_config(path)['proximal_mu'] == 0.3
