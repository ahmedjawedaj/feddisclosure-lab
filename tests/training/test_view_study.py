import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("flwr")
from feddisclosure.attacks.view_study import (client_step, flatten, secagg_aggregate, unflatten)  # noqa: E402
from feddisclosure.training.task import DigitMLP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def _vectors(n=4, seed=0):
    torch.manual_seed(seed)
    return [flatten(DigitMLP(64, 10).state_dict()) for _ in range(n)]


def test_flatten_roundtrip():
    state = DigitMLP(64, 10).state_dict()
    back = unflatten(flatten(state), state)
    assert all(torch.equal(back[k], state[k]) for k in state)


@pytest.mark.parametrize("max_weight", [1000.0, 16.0])
def test_secagg_quantiser_error_stays_inside_worst_case_bound(max_weight):
    vecs, sizes = _vectors(), [8, 8, 8, 8]
    exact = sum(v * n / sum(sizes) for v, n in zip(vecs, sizes))
    bound = 4 * (2 * 8.0 / 2 ** 22) * max_weight / sum(sizes)
    for seed in range(5):
        assert np.abs(secagg_aggregate(vecs, sizes, max_weight=max_weight, rng_seed=seed) - exact).max() <= bound


def test_collusion_recovers_victim_to_quantisation_precision():
    vecs, sizes = _vectors(), [8, 8, 8, 8]
    agg = secagg_aggregate(vecs, sizes, max_weight=16.0, rng_seed=1)
    others = sum(vecs[j] * sizes[j] / sum(sizes) for j in range(1, 4))
    victim = (agg - others) * sum(sizes) / sizes[0]
    assert np.abs(victim - vecs[0]).max() < 1e-3       # server alone sees only the mean, which is far off
    assert np.abs(agg - vecs[0]).max() > 1e-2


def test_dp_step_adds_noise_that_grows_with_sigma():
    pytest.importorskip("opacus")
    torch.manual_seed(0)
    init = DigitMLP(64, 10).state_dict()
    x = np.random.default_rng(0).random((8, 64)).astype(np.float32); y = np.arange(8) % 10
    plain = flatten(client_step(init, x, y, lr=0.15, sigma=None))
    diffs = [np.linalg.norm(flatten(client_step(init, x, y, lr=0.15, sigma=s, seed=1)) - plain) for s in (0.05, 1.0)]
    assert diffs[1] > diffs[0] > 0


def test_replay_gate_runs_reference_existing_configs():
    import verify_flower_replay_gate as gate
    assert all((ROOT / cfg).is_file() for cfg, _ in gate.RUNS.values())


def test_workflows_are_statically_consistent():
    import check_workflows_static as chk
    assert all(v["ok"] for v in chk.check().values())
