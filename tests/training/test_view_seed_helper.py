"""The shared record selection must reproduce the data selection of the v0.14.2 run_view_comparison.py."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from feddisclosure.attacks.view_seed import client_step_seed, select_view_records
from feddisclosure.evaluation.intervals import client_seed

ROOT = Path(__file__).resolve().parents[2]
N, B = 4, 8


def original_selection(n_total, seed):
    """The logic of scripts/run_view_comparison.py one_seed before the refactor, copied verbatim."""
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(n_total)
    idx = perm[: N * B]
    return idx, perm[N * B:], rng


@pytest.mark.parametrize("seed", range(10))
def test_selection_matches_the_original_code_including_later_draws(seed):
    sel = select_view_records(1797, N, B, seed)
    idx, pool, rng = original_selection(1797, seed)
    assert np.array_equal(sel.idx, idx) and np.array_equal(sel.pool_idx, pool)
    assert np.array_equal(sel.rng.choice(100, 8, replace=False), rng.choice(100, 8, replace=False))


def test_pool_is_disjoint_from_client_records_and_covers_everything():
    sel = select_view_records(1797, N, B, 3)
    assert len(sel.idx) == 32 and not set(sel.idx) & set(sel.pool_idx)
    assert len(set(sel.idx) | set(sel.pool_idx)) == 1797


def test_too_many_records_requested_fails():
    with pytest.raises(ValueError):
        select_view_records(30, N, B, 0)


def test_client_seed_legacy_and_new_scheme():
    assert [client_step_seed(2, c) for c in range(3)] == [20, 21, 22]
    assert client_step_seed(2, 3, legacy=False) == client_seed(2, 3) == 2051
    assert client_step_seed(1, 10) == client_step_seed(2, 0)                 # the old collision
    assert client_step_seed(1, 10, legacy=False) != client_step_seed(2, 0, legacy=False)


def test_selection_reproduces_saved_v0142_mean_image_floor():
    """End to end: the saved per-seed mean-image floor can only be reproduced with the exact same pool."""
    saved = ROOT / "reports/results/privacy/v014/view_comparison"
    if not (saved / "seed0.json").is_file():
        pytest.skip("saved v0.14.2 evidence not present")
    sys.path.insert(0, str(ROOT / "scripts"))
    from compute_label_floor import floor_for_seed
    from feddisclosure.training.datasets import _source
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    for seed in (0, 1):
        want = json.loads((saved / f"seed{seed}.json").read_text())["floor"]["mean_image_mse"]
        assert floor_for_seed(x, y, seed, repeats=2)["mean_image_mse"] == pytest.approx(want, abs=1e-12)
