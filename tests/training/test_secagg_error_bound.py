"""Property tests for the one-round SecAgg+ quantisation error bound (v0.15 E1, E2 and E3).

Every check runs the real upstream flwr quantize and dequantize through
feddisclosure.attacks.view_study.secagg_aggregate. The statements are in
docs/proofs/one-round-secagg-error.md. The long search (20,000 cases) is
scripts/run_secagg_bound_search.py. These tests are the smaller versions.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flwr")
pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_secagg_bound_search as rsb  # noqa: E402
from feddisclosure.attacks.view_study import secagg_aggregate  # noqa: E402

C, Q, LD = rsb.C, rsb.Q, rsb.LD
FAST = settings(max_examples=1500, deadline=None, derandomize=True, database=None,
                suppress_health_check=list(HealthCheck))


def _check(vectors, sizes, max_weight, seed):
    assert rsb.no_clipping(vectors, sizes, max_weight)
    n = len(vectors)
    agg = secagg_aggregate(vectors, sizes, max_weight=max_weight, rng_seed=seed)
    mu_q, mu_true, total_q = rsb.exact_means(vectors, sizes, max_weight)
    err_q = np.abs(agg.astype(LD) - mu_q).astype(np.float64)
    err_t = np.abs(agg.astype(LD) - mu_true).astype(np.float64)
    assert err_q.max() <= rsb.stated_bound(n, total_q)
    assert err_q.max() <= rsb.worst_case_bound(n, total_q)          # the float64 only bound also holds
    assert np.all(err_t <= rsb.stated_bound(n, total_q) + rsb.weight_term(vectors, total_q))


# ---------------------------------------------------------------------------------------
# Facts about the upstream path that the proof relies on
# ---------------------------------------------------------------------------------------
def test_float32_inputs_stay_float32_until_rounding_and_land_on_a_grid_of_one_eighth():
    rng = np.random.default_rng(0)
    v = (rng.uniform(-8, 8, 50000)).astype(np.float32)
    x32 = rsb.pre_quantised(v, 1.0)
    assert x32.dtype == np.float32
    assert np.all(np.mod(x32.astype(np.float64) * 8, 1) == 0)


def test_quantised_levels_include_the_upper_end_Q():
    import flwr.common.secure_aggregation.quantization as qz
    k = qz.quantize([np.array([8.0, -8.0, 0.0])], C, Q)[0]
    assert k.tolist() == [Q, 0, Q // 2]                              # the upstream docstring says Q - 1


def test_cell_lemma_and_bias_size_on_random_float32_arrays():
    study = rsb.float32_path_study(batches=4, batch_size=250_000, seed=11)
    assert study["cell_lemma_violations"] == 0
    assert study["x32_not_multiple_of_one_eighth"] == 0
    assert study["integer_cells_with_bias_above_3_16"] == 0
    assert study["worst_single_client_error_units_either_rounding"] < 1.0
    table = study["bias_table_units"]
    assert table["all"]["max_abs_bias_units"] <= 3.0 / 16.0
    assert abs(table["all"]["mean_bias_units"]) < 5e-4                  # no systematic sign over random arrays
    assert 0.125 <= table["abs_wv_lt_1"]["max_abs_bias_units"] <= 0.141   # the reviewer's 0.137 holds for small values
    assert table["abs_wv_4_to_8"]["max_abs_bias_units"] > 0.18            # the general supremum is 3/16


def test_bound_assumption_on_n():
    lim = rsb.sum_range_check()
    assert lim["largest_n_without_possible_wrap"] == 1023
    assert lim["n_1023_all_at_plus_c"]["wrapped"] is False


def test_clipped_inputs_are_detected_and_leave_the_bound():
    v = [np.full(4, 6.0, dtype=np.float32), np.full(4, 6.0, dtype=np.float32)]
    sizes = [1000, 1000]
    assert rsb.no_clipping(v, sizes, 1000.0)
    big = [np.full(4, 9.0, dtype=np.float32)] * 2
    assert not rsb.no_clipping(big, sizes, 1000.0)
    agg = secagg_aggregate(big, sizes, max_weight=1000.0, rng_seed=0)
    mu_q, _, total_q = rsb.exact_means(big, sizes, 1000.0)
    assert np.abs(agg - mu_q.astype(np.float64)).max() > 100 * rsb.stated_bound(2, total_q)


# ---------------------------------------------------------------------------------------
# E1: counterexample search
# ---------------------------------------------------------------------------------------
@FAST
@given(data=st.data(), n=st.integers(2, 12), d=st.integers(1, 4),
       max_weight=st.sampled_from([16.0, 200.0, 1000.0, 5000.0]), seed=st.integers(0, 2 ** 31 - 1))
def test_e1_explicit_float32_values_unequal_sizes(data, n, d, max_weight, seed):
    sizes = [data.draw(st.integers(1, int(max_weight))) for _ in range(n)]
    dqs = [q / Q for q in rsb.quantised_weights(sizes, max_weight)]
    vectors = []
    for dq in dqs:
        lim = rsb.limit_for(dq, np.float32)
        frac = data.draw(st.lists(st.one_of(st.floats(-1, 1, width=32),
                                            st.sampled_from([-1.0, 1.0, 0.0, 0.5, -0.5, 0.999999, -0.999999])),
                                  min_size=d, max_size=d))
        v = (np.asarray(frac) * float(lim)).astype(np.float32)
        while np.any(np.abs(v * dq) > C):
            v = np.where(np.abs(v * dq) > C, np.nextafter(v, np.float32(0)), v).astype(np.float32)
        vectors.append(v)
    _check(vectors, sizes, max_weight, seed)


@FAST
@given(seed=st.integers(0, 2 ** 31 - 1), family=st.sampled_from(rsb.FAMILIES))
def test_e1_generated_cases_n_up_to_64(seed, family):
    vectors, sizes, max_weight = rsb.make_case(np.random.default_rng(seed), family, np.float32)
    assert 2 <= len(vectors) <= 64
    _check(vectors, sizes, max_weight, seed % 1000003)


@FAST
@given(seed=st.integers(0, 2 ** 31 - 1), family=st.sampled_from(rsb.FAMILIES))
def test_e1_float64_inputs_control(seed, family):
    vectors, sizes, max_weight = rsb.make_case(np.random.default_rng(seed), family, np.float64)
    _check(vectors, sizes, max_weight, seed % 1000003)


# ---------------------------------------------------------------------------------------
# E2: the float32 path adds a deterministic bias but does not break 2 c n / T
# ---------------------------------------------------------------------------------------
def test_e2_float32_bias_is_real_and_the_worst_case_bound_still_holds():
    study = rsb.run_e2(batches=4, batch_size=200_000, seed=5)
    ex = study["identical_clients_example"]
    assert abs(ex["per_element_bias_units"]) > 0.12                          # a deterministic offset, no randomness
    assert ex["aggregate_error_units"] > ex["hoeffding_term_units_d1_delta_0.01"]   # pure Hoeffding term fails
    assert ex["aggregate_error_units"] < ex["worst_case_units"]              # 2cn/T holds
    assert study["adversarial_rounding_tightness"]["float64_always_down"] > 0.9999
    assert study["adversarial_rounding_tightness"]["float64_always_up"] > 0.9999
    assert study["worst_single_client_error_units_either_rounding"] < 1.0   # no float32 counterexample to 2cn/T


# ---------------------------------------------------------------------------------------
# E3: Hoeffding with independent rounding, failure with a shared stream
# ---------------------------------------------------------------------------------------
def test_stream_helper_matches_secagg_aggregate_for_one_client():
    rng = np.random.default_rng(3)
    vecs = [rng.uniform(-1, 1, 20)]
    one = secagg_aggregate(vecs, [450], max_weight=1000.0, rng_seed=9)
    assert np.array_equal(rsb.aggregate_with_client_streams(vecs, [450], 1000.0, [9]), one)


@pytest.mark.parametrize("n", [4, 16, 64, 256])
def test_e3_independent_rounding_p99_below_hoeffding(n):
    row = rsb.run_e3([n], trials=150, d=32, seed=100 + n)[0]
    assert row["independent"]["below_bound"]
    assert row["independent"]["below_single_bound"]
    assert row["independent"]["fraction_trials_above_bound"] <= 0.01 + 0.04   # 150 trials, allow binomial noise


def test_e3_shared_seed_violates_at_n_256_and_error_does_not_shrink():
    row = rsb.run_e3([256], trials=150, d=32, seed=777)[0]
    assert not row["shared_seed_identical_data"]["below_bound"]
    assert row["shared_seed_identical_data"]["p99_max_over_bound"] > 2
    assert not row["shared_seed_different_data"]["below_bound"]            # common draws correlate the clients
    # identical data and shared draws: error in units is proportional to n, not to sqrt(n)
    assert 0.8 < row["shared_seed_identical_data"]["p99_max_error_units_over_n"] < 1.0
    # independent draws: proportional to sqrt(n)
    assert 1.2 < row["independent"]["p99_max_error_units_over_sqrt_n"] < 1.9


def test_shared_stream_variance_formula():
    """Var(sum of errors) = sum over i,k of (min(f_i, f_k) - f_i f_k) for one shared uniform draw."""
    f = np.array([0.2, 0.5, 0.9, 0.5])
    u = (np.arange(400000) + 0.5) / 400000                       # exact quadrature of a uniform draw
    err = (u[:, None] <= f[None, :]).astype(float) - f[None, :]
    assert abs(err.sum(1).var() - sum(min(a, b) - a * b for a in f for b in f)) < 1e-4
    assert abs(err.sum(1).mean()) < 1e-9                         # still unbiased
