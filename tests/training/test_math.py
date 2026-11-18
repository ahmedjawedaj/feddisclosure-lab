import numpy as np
import pytest
pytest.importorskip("torch")
import torch
from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local, evaluate_local, model_bytes


def test_weighted_fedavg_against_independent_numpy():
    states = [{"layer": torch.tensor([1., 3.])}, {"layer": torch.tensor([5., 11.])},
              {"layer": torch.tensor([13., 23.])}]
    sizes = [2, 3, 5]
    actual = fedavg_weighted(states, sizes)["layer"].numpy()
    expected = np.average(np.stack([s["layer"].numpy() for s in states]), axis=0, weights=sizes)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    with pytest.raises(ValueError):
        fedavg_weighted(states, [2, 0, 5])
    with pytest.raises(ValueError):
        fedavg_weighted([states[0], {"other": torch.tensor([1., 3.])}], [1, 1])


def test_pytorch_training_reproducible_and_changes_parameters():
    x = np.random.default_rng(3).normal(size=(20, 64)).astype(np.float32)
    y = np.random.default_rng(3).integers(0, 10, size=20).astype(np.int64)
    torch.manual_seed(4)
    m1 = DigitMLP()
    initial = {k: v.clone() for k, v in m1.state_dict().items()}
    torch.manual_seed(4)
    m2 = DigitMLP()
    a = train_local(m1, x, y, epochs=2, batch_size=5, lr=0.1, seed=4)
    b = train_local(m2, x, y, epochs=2, batch_size=5, lr=0.1, seed=4)
    assert a == b
    assert all(torch.equal(m1.state_dict()[k], m2.state_dict()[k]) for k in initial)
    assert any(not torch.equal(initial[k], m1.state_dict()[k]) for k in initial)
    assert 0 <= evaluate_local(m1, x, y)["accuracy"] <= 1
    assert model_bytes(m1.state_dict()) > 0
