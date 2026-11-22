"""M1: multi-round note. Convex L2 bound in 1,000 random trials and the non-convex ReLU counterexample.

Statements are in docs/proofs/multi-round-note.md. Each trial runs FedAvg with local full batch gradient
descent on a convex smooth model (least squares or ridge logistic regression), once with exact weighted
averaging and once with the real upstream SecAgg+ quantiser path (secagg_aggregate), and checks
    L2 distance after T rounds <= T * sqrt(d) * (largest one-round per coordinate bound).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flwr")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_secagg_bound_search as rsb  # noqa: E402


def test_convex_l2_bound_never_violated_in_1000_trials():
    res = rsb.run_m1_convex(trials=1000, seed=424242)
    assert res["trials"] == 1000
    assert res["trials_with_step_exactly_two_over_beta"] >= 200       # the boundary step size is well covered
    assert res["kinds"]["ls"] > 300 and res["kinds"]["logreg"] > 300
    assert res["one_round_bound_violations_inside_trajectories"] == 0
    assert res["telescoping_violations"] == 0
    assert res["l2_bound_violations"] == 0


def test_step_exactly_two_over_beta_is_non_expansive_for_one_client():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (30, 6))
    beta = float(np.linalg.eigvalsh(X.T @ X / 30)[-1])
    eta = 2.0 / beta
    y = rng.normal(0, 1, 30)

    def step(t):
        return t - eta * X.T @ (X @ t - y) / 30
    for _ in range(200):
        a, b = rng.normal(0, 1, 6), rng.normal(0, 1, 6)
        assert np.linalg.norm(step(a) - step(b)) <= np.linalg.norm(a - b) * (1 + 1e-12)


def test_step_above_two_over_beta_breaks_the_bound_so_the_check_can_fail():
    res = rsb.run_m1_negative_control(trials=100, seed=5, kappa=1.3)
    assert res["l2_bound_violations"] > 10


def test_two_round_relu_counterexample_is_reproduced():
    res = rsb.relu_counterexample(range(100))
    assert res["exact_round1_model"][0] == 0.0                       # exactly at the ReLU kink
    assert res["round1_inside_bound"]
    assert res["seeds_with_round1_w_off_the_kink"] > 10
    assert res["min_round2_over_one_round_bound"] > 1e4              # round 2 difference is O(0.1), bound is O(1e-5)
    assert res["min_round2_difference_off_kink"] > 2 * res["naive_two_round_tolerance_2x"]


def test_relu_counterexample_on_repo_digits_data_matches_saved_evidence():
    out = rsb.relu_digits_reproduction(trials=20)
    if "unavailable" in out:
        pytest.skip(out["unavailable"])
    # saved: reports/results/secagg/v014_dropout_matrix/error_propagation.json, 200 trials, 39.5% above 2x bound
    assert out["fraction_above_2x_bound"] >= 0.2
    assert out["one_round_bound"] == pytest.approx(3.9736429850260414e-05)


def test_tolerance_limit_arithmetic():
    t = rsb.tolerance_limit_note()
    assert t["p_all_300_below_true_p99"] == pytest.approx(0.049, abs=5e-4)
    assert t["confidence_of_max_covering_p99"] > 0.95
