import numpy as np
import pytest

from feddisclosure.evaluation.intervals import (TUNING_SEED_BASE, client_seed, clopper_pearson_interval,
                                                rate_summary, tuning_seed, wilson_interval)


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-3)
    lo, hi = clopper_pearson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.6915, abs=1e-3)
    lo, hi = clopper_pearson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.1871, 0.8129), abs=1e-3)


def test_wilson_known_value_and_range():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    for k in range(0, 31):
        lo, hi = wilson_interval(k, 30)
        assert 0.0 <= lo <= k / 30 <= hi <= 1.0


def test_clopper_pearson_contains_wilson_centre_and_is_wider():
    for k in (1, 7, 15, 29):
        w, c = wilson_interval(k, 30), clopper_pearson_interval(k, 30)
        assert c[0] <= w[0] + 1e-9 and c[1] >= w[1] - 1e-9


def test_rate_summary():
    s = rate_summary([True] * 3 + [False] * 7)
    assert s["successes"] == 3 and s["trials"] == 10 and s["rate"] == pytest.approx(0.3)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        clopper_pearson_interval(1, 0)


def test_client_seeds_unique_up_to_limit_and_old_scheme_collided():
    seeds = {client_seed(s, c) for s in range(20) for c in range(64)}
    assert len(seeds) == 20 * 64
    old = {s * 10 + c for s in range(20) for c in range(16)}
    assert len(old) < 20 * 16        # documents the collision that the new scheme removes
    with pytest.raises(ValueError):
        client_seed(0, 1024)


def test_tuning_seeds_do_not_overlap_evaluation_seeds():
    assert tuning_seed(0) == TUNING_SEED_BASE
    assert all(tuning_seed(i) > 999 for i in range(100))
