import importlib.util
import numpy as np
import pytest

from feddisclosure.defenses.opacus_training import validate_dp_config, run_local_dp_sgd, OpacusUnavailable


def valid():
    return dict(records=96, epochs=2, batch_size=32, learning_rate=.15,
                target_epsilon=5.0, target_delta=1e-5, max_grad_norm=1.0)


def test_invalid_dp_adjacency_config_fails_closed():
    validate_dp_config(**valid())
    invalid = ({"target_delta": 0}, {"target_delta": .05}, {"target_epsilon": float("nan")},
               {"max_grad_norm": -1}, {"batch_size": 96}, {"records": 5},
               {"epochs": 0}, {"learning_rate": float("inf")})
    for updates in invalid:
        with pytest.raises(ValueError):
            validate_dp_config(**(valid() | updates))


def test_native_opacus_absence_never_fakes_result():
    if importlib.util.find_spec("opacus") is not None:
        pytest.skip("Native Opacus available; run the native integration test")
    with pytest.raises(OpacusUnavailable):
        run_local_dp_sgd(np.zeros((96, 64), dtype=np.float32),
                         np.zeros(96, dtype=np.int64))


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("opacus") is None, reason="Needs real Opacus runtime")
def test_native_opacus_accountant_one_site():
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    state, meta = run_local_dp_sgd((x[:96] / 16).astype(np.float32), y[:96], epochs=1,
                                   batch_size=16, target_epsilon=7.0,
                                   target_delta=1e-5, max_grad_norm=1.0)
    assert meta["epsilon_accounted"] <= 7.0 + 1e-5
    assert meta["steps"] > 0 and meta["noise_multiplier"] > 0
    assert state["network.0.weight"].shape == (32, 64)
