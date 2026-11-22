"""Scaling study helpers (v0.15, WP9, rows S1 and S2). No Docker and no network."""
import json
from pathlib import Path

import numpy as np
import pytest

from feddisclosure.attacks.view_seed import select_view_records
from feddisclosure.evaluation import scaling_study as ss
from feddisclosure.evaluation.intervals import wilson_interval

ROOT = Path(__file__).resolve().parents[2]


def test_collusion_view_recovers_the_victim_exactly_without_quantisation():
    rng = np.random.default_rng(0)
    for n in (4, 8, 16):
        states = [rng.normal(size=50) for _ in range(n)]
        sizes = [8] * n
        aggregate = sum(s * w for s, w in zip(states, sizes)) / sum(sizes)
        np.testing.assert_allclose(ss.collusion_view(aggregate, states, sizes), states[0], atol=1e-12)


def test_collusion_view_handles_unequal_sizes_and_validates_input():
    rng = np.random.default_rng(1)
    states, sizes = [rng.normal(size=10) for _ in range(3)], [5, 11, 8]
    aggregate = sum(s * w for s, w in zip(states, sizes)) / sum(sizes)
    np.testing.assert_allclose(ss.collusion_view(aggregate, states, sizes), states[0], atol=1e-12)
    with pytest.raises(ValueError):
        ss.collusion_view(aggregate, states, [5, 11])
    with pytest.raises(ValueError):
        ss.collusion_view(aggregate, states[:1], [5])


def test_victim_scoring_uses_the_victim_rows_of_a_joint_reconstruction():
    rng = np.random.default_rng(2)
    originals = rng.random((24, 64))
    recovered = originals[rng.permutation(24)].copy()
    recovered[:] = recovered + 0.0
    assert np.allclose(ss.victim_per_image_mse(originals, recovered, 8), 0.0)
    # An individual view holds only the victim batch and is scored against the first rows.
    assert np.allclose(ss.victim_per_image_mse(originals, originals[:8][::-1].copy(), 8), 0.0)
    assert ss.victim_per_image_mse(originals, rng.random((24, 64)), 8).shape == (8,)


def test_victim_images_do_not_depend_on_the_number_of_clients():
    for seed in (0, 1, 7):
        a, b = select_view_records(1797, 4, 8, seed), select_view_records(1797, 8, 8, seed)
        assert np.array_equal(a.idx[:8], b.idx[:8])
        assert not set(b.idx.tolist()) & set(b.pool_idx.tolist())
        assert len(b.pool_idx) == 1797 - 64


def test_bootstrap_is_deterministic_ordered_and_checked():
    v = np.random.default_rng(3).random(30)
    first = ss.bootstrap_ci(v)
    assert first == ss.bootstrap_ci(v)
    assert first[1] <= first[0] <= first[2]
    flags = np.array([1.0] * 10 + [0.0] * 20)
    lo, hi = ss.bootstrap_ci(flags)[1:]
    assert 0.0 <= lo <= hi <= 1.0
    with pytest.raises(ValueError):
        ss.bootstrap_ci(v, resamples=100)
    with pytest.raises(ValueError):
        ss.bootstrap_ci([0.5])


def test_pooled_rate_counts_images_below_the_threshold_and_uses_wilson():
    per_seed = [[0.01] * 6 + [0.2] * 2, [0.04999] * 8, [0.05] * 4 + [0.5] * 4]    # exactly 0.05 is not recognisable
    r = ss.pooled_rate(per_seed)
    assert (r["successes"], r["images"]) == (6 + 8 + 0, 24)
    assert r["wilson"] == list(wilson_interval(14, 24))
    assert 0 <= r["wilson"][0] <= r["rate"] <= r["wilson"][1] <= 1


def _fake_run(n, floor, **views):
    return {"clients": n, "label_floor_per_image_mse": floor, "mean_image_per_image_mse": [0.07] * len(floor),
            "views": views}


def _eight(good):      # eight images, `good` of them below 0.05
    return [0.01] * good + [0.2] * (8 - good)


def test_summarise_rejects_mixed_client_counts():
    with pytest.raises(ValueError):
        ss.summarise([_fake_run(4, _eight(6), v=_eight(6)), _fake_run(8, _eight(6), v=_eight(6))])


def _summary(n, floor_good, agg_good, col_good, seeds=10):
    runs = [_fake_run(n, _eight(floor_good), agg=_eight(agg_good), col=_eight(col_good)) for _ in range(seeds)]
    return ss.summarise(runs)


def test_s1_boundary_is_floor_plus_five_points_and_a_failure_is_recorded_as_fail():
    assert ss.check_s1(_summary(8, 6, 6, 8), "agg")["pass"] is True               # aggregate equals the floor
    assert ss.check_s1(_summary(8, 4, 5, 8), "agg")["pass"] is False              # +12.5 points
    # Floor 40 of 80 images. Aggregate 44 of 80 is exactly +5 points and passes.
    runs = [_fake_run(8, _eight(4), agg=_eight(4)) for _ in range(9)] + [_fake_run(8, _eight(4), agg=_eight(8))]
    r = ss.check_s1(ss.summarise(runs), "agg")
    assert r["observed"]["gap"] == pytest.approx(0.05) and r["pass"] is True
    # One more image above the floor is +6.25 points and fails.
    runs[0] = _fake_run(8, _eight(4), agg=_eight(5))
    assert ss.check_s1(ss.summarise(runs), "agg")["pass"] is False


def test_s2_needs_every_client_count_at_or_above_85_percent():
    ok4, ok8 = _summary(4, 6, 3, 7, seeds=20), _summary(8, 6, 3, 7, seeds=20)    # 87.5 percent
    assert ss.check_s2({4: ok4, 8: ok8}, "col")["pass"] is True
    low8 = _summary(8, 6, 3, 6, seeds=20)                                       # 75 percent
    r = ss.check_s2({4: ok4, 8: low8}, "col")
    assert r["pass"] is False and r["observed"]["by_clients"]["8"]["rate"] == pytest.approx(0.75)


def test_run_seed_is_deterministic_and_shaped_for_both_client_counts():
    pytest.importorskip("flwr")      # the SecAgg+ quantiser comes from Flower
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    x, y = x.astype(np.float32) / 16.0, y.astype(np.int64)
    out = {}
    for n in (4, 8):
        a = ss.run_seed(x, y, 0, n, iterations=3)
        assert a == ss.run_seed(x, y, 0, n, iterations=3)
        assert set(a["views"]) == {"plaintext_individual", "secagg_aggregate_maxw1000", "secagg_collusion_n-1_maxw1000"}
        assert all(len(v) == 8 for v in a["views"].values()) and len(a["label_floor_per_image_mse"]) == 8
        assert a["public_pool_size"] == 1797 - 8 * n
        out[n] = a
    # Same seed, same victim records, same initial model and same victim step, so plaintext is identical.
    assert out[4]["views"]["plaintext_individual"] == out[8]["views"]["plaintext_individual"]


def test_legacy_seeds_change_the_client_steps_but_not_the_data():
    pytest.importorskip("flwr")      # the SecAgg+ quantiser comes from Flower
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    x, y = x.astype(np.float32) / 16.0, y.astype(np.int64)
    new = ss.run_seed(x, y, 1, 4, iterations=3, legacy_seeds=False)
    old = ss.run_seed(x, y, 1, 4, iterations=3, legacy_seeds=True)
    assert new["label_floor_per_image_mse"] == old["label_floor_per_image_mse"]


def test_saved_summary_matches_a_recomputation_from_the_per_seed_files():
    run_dir = ROOT / "reports/results/privacy/v0150/scaling/runs"
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        pytest.skip("scaling results are not in this checkout")
    saved = json.loads(summary_path.read_text())
    for n, s in saved["summaries"].items():
        runs = [json.loads((run_dir / f"n{n}_seed{i}.json").read_text()) for i in range(saved["seeds"])]
        fresh = ss.summarise(runs)
        assert fresh["label_floor"]["successes"] == s["label_floor"]["successes"]
        for name, v in s["views"].items():
            assert fresh["views"][name]["successes"] == v["successes"]
            assert fresh["views"][name]["wilson"] == pytest.approx(v["wilson"], abs=1e-12)
    assert saved["S1"]["pass"] is (saved["S1"]["observed"]["gap"] <= 0.05)
