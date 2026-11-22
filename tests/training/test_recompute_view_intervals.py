"""recompute_view_intervals on inputs whose answers can be worked out by hand."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import recompute_view_intervals as r  # noqa: E402


def test_bootstrap_of_a_constant_vector_is_degenerate():
    assert r.bootstrap_mean_ci(np.full(10, 0.25)) == [0.25, 0.25, 0.25]


def test_bootstrap_uses_the_2_5_and_97_5_quantile_levels(monkeypatch):
    levels = []
    real = np.quantile
    monkeypatch.setattr(r.np, "quantile", lambda a, q, *x, **k: (levels.append(q), real(a, q, *x, **k))[1])
    out = r.bootstrap_mean_ci(np.array([0.0, 1.0, 0.5, 0.25]))
    assert levels == [pytest.approx(0.025), pytest.approx(0.975)]
    assert out[1] <= out[0] <= out[2]


def test_bootstrap_interval_of_two_points_spans_both_values():
    # Resampled means of [0, 1] can only be 0, 0.5 or 1. Mass at each end is 25%, so the 2.5% and 97.5% quantiles are the ends.
    assert r.bootstrap_mean_ci(np.array([0.0, 1.0]), resamples=4000) == [0.5, 0.0, 1.0]


def test_pooled_rate_counts_eight_images_per_seed():
    out = r.pooled_rate(np.array([0.5, 0.25, 1.0]))          # 4 + 2 + 8 successes out of 24
    assert (out["successes"], out["trials"]) == (14, 24)
    assert out["rate"] == pytest.approx(14 / 24)


def test_pooled_rate_rejects_a_rate_that_is_not_a_whole_number_of_images():
    with pytest.raises(ValueError, match="not an integer"):
        r.pooled_rate(np.array([0.3, 0.5]))


def test_wilson_interval_for_54_of_80():
    per_seed = np.array([7, 7, 5, 6, 5, 5, 6, 5, 5, 3]) / 8      # 54 successes in 80 images
    out = r.pooled_rate(per_seed)
    assert (out["successes"], out["trials"], out["rate"]) == (54, 80, 0.675)
    assert out["wilson"] == [pytest.approx(0.566, abs=1e-3), pytest.approx(0.768, abs=1e-3)]
    lo, hi = out["clopper_pearson"]
    assert 0.5 < lo < 0.675 < hi < 0.85


FLOOR_RATE, FLOOR_MSE = 0.5, 0.05


def res_with(agg_rate, agg_mse):
    """Smallest result dict that checks() reads. Only the aggregate row varies."""
    def view(rate, mse):
        return {"recognisable": {"rate": rate, "wilson": [0.0, 1.0], "clopper_pearson": [0.0, 1.0]},
                "mse_mean": mse, "mse_bootstrap95": [mse, mse, mse]}
    return {"label_floor": {"mse_mean": FLOOR_MSE, "mse_bootstrap95": [FLOOR_MSE] * 3,
                            "recognisable": {"rate": FLOOR_RATE}},
            "views": {"secagg_aggregate_maxw1000": view(agg_rate, agg_mse),
                      "plaintext_individual": view(0.9, 0.01), "secagg_collusion_n-1_maxw1000": view(0.9, 0.02),
                      "dp0.2_individual": view(0.0, 0.2)},
            "mean_image_mse_saved": 0.07, "max_abs_diff_to_saved_summary": 0.0}


def c2(agg_rate, agg_mse):
    return next(c for c in r.checks(res_with(agg_rate, agg_mse), {}) if c["id"] == "C2")["pass"]


BOUNDARY_RATE = FLOOR_RATE + 0.05


@pytest.mark.parametrize("rate, expected", [(BOUNDARY_RATE, True), (np.nextafter(BOUNDARY_RATE, 1.0), False)])
def test_c2_rate_threshold_is_floor_plus_five_points(rate, expected):
    assert c2(float(rate), FLOOR_MSE) is expected


@pytest.mark.parametrize("mse, expected", [(FLOOR_MSE - 0.005, True), (np.nextafter(FLOOR_MSE - 0.005, 0.0), False)])
def test_c2_mse_threshold_is_floor_minus_0_005(mse, expected):
    assert c2(FLOOR_RATE, mse) is expected


def test_c2_fails_when_the_attack_clearly_beats_the_floor():
    assert c2(0.9, 0.01) is False


def test_c5_does_not_count_skipped_tests_as_passed():
    ok = {"returncode": 0, "failed": 0, "skipped": 0, "passed": 40, "label_floor_passed": 7, "label_floor_skipped": 0}
    assert r.unit_tests_ok(ok)
    assert not r.unit_tests_ok({**ok, "skipped": 1})
    assert not r.unit_tests_ok({**ok, "label_floor_passed": 0, "label_floor_skipped": 1})
    assert not r.unit_tests_ok({**ok, "failed": 1})
    assert not r.unit_tests_ok({"returncode": 0, "summary": "all passed"})       # no counts recorded


def test_c5_rejects_a_mse_interval_that_does_not_contain_its_mean():
    res = res_with(0.5, 0.05)
    res["views"]["plaintext_individual"]["mse_bootstrap95"] = [0.01, 0.02, 0.03]
    ok = {"returncode": 0, "failed": 0, "skipped": 0, "passed": 1, "label_floor_passed": 1, "label_floor_skipped": 0}
    assert next(c for c in r.checks(res, ok) if c["id"] == "C5")["pass"] is False


def test_pytest_verbose_parser_counts_skips_separately():
    out = ("tests/training/test_label_floor.py::test_a PASSED [ 10%]\n"
           "tests/training/test_label_floor.py::test_b SKIPPED (no flwr) [ 20%]\n"
           "tests/training/test_intervals.py::test_c PASSED [ 30%]\n"
           "tests/training/test_intervals.py::test_d FAILED [ 40%]\n")
    assert r.parse_pytest_verbose(out) == {"passed": 2, "failed": 1, "skipped": 1,
                                           "label_floor_passed": 1, "label_floor_skipped": 1}
