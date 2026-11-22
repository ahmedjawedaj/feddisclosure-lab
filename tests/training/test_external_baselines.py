"""WP11 helpers: matched data selection, LiRA scoring on toy inputs, verdict logic including failing cases."""
import importlib.util

import numpy as np
import pytest

from feddisclosure.attacks.view_seed import select_view_records
from feddisclosure.evaluation import external_baselines as eb
from feddisclosure.training.lira_study import lira_scores, roc_summary


@pytest.mark.parametrize("seed", [0, 3, 9])
def test_matched_records_equal_repo_selection(seed):
    n = 1797
    x = np.random.default_rng(0).random((n, 64)).astype(np.float32)
    y = np.arange(n) % 10
    rec = eb.matched_records(x, y, seed)
    sel = select_view_records(n, 4, 8, seed)
    assert np.array_equal(rec["idx"], sel.idx)
    assert np.array_equal(rec["victim_x"], x[sel.idx[:8]])
    assert np.array_equal(rec["victim_y"], y[sel.idx[:8]])
    assert not set(sel.idx) & set(sel.pool_idx) and len(rec["pool_x"]) == n - 32


def test_victim_gradient_divides_by_lr_in_parameter_order():
    import torch
    delta = {k: torch.full((2,), 0.3) for k in eb.PARAM_NAMES}
    g = eb.victim_gradient({}, delta, lr=0.15)
    assert len(g) == 4 and all(torch.allclose(t, torch.full((2,), 2.0)) for t in g)


def test_mse_summary_uses_strict_recognisable_threshold():
    s = eb.mse_summary([0.0, 0.049, 0.05, 0.2])
    assert s["recognisable"] == 0.5 and s["mse"] == pytest.approx(0.07475)


def test_b1_passes_at_exactly_the_ratio_and_fails_above():
    assert eb.b1_verdict([0.15, 0.15], [0.10, 0.10])["verdict"] == "PASS"       # ratio 1.5
    assert eb.b1_verdict([0.151, 0.151], [0.10, 0.10])["verdict"] == "FAIL"
    assert eb.b1_verdict([0.01], [0.10])["verdict"] == "PASS"                    # repo stronger


def test_b1_failing_case_when_external_is_near_exact():
    v = eb.b1_verdict([0.0132] * 5, [0.0001] * 5)
    assert v["verdict"] == "FAIL" and v["ratio"] > 100


def test_b1_rejects_bad_input():
    with pytest.raises(ValueError):
        eb.b1_verdict([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        eb.b1_verdict([float("nan")], [0.1])


def test_b2_boundary_and_failure():
    assert eb.b2_verdict(0.72, 0.75)["verdict"] == "PASS"       # exactly the margin
    assert eb.b2_verdict(0.7199, 0.75)["verdict"] == "FAIL"
    assert eb.b2_verdict(0.80, 0.75)["verdict"] == "PASS"
    with pytest.raises(ValueError):
        eb.b2_verdict(float("nan"), 0.7)


def _toy_shadows(seed=0, m=40, n=30, gap=1.5):
    rng = np.random.default_rng(seed)
    mem = rng.random((m, n)) < 0.5
    mem[:4] = True
    mem[4:8] = False
    phi = rng.normal(0, 1, (m, n)) + gap * mem
    return phi, mem


def test_independent_lira_matches_closed_form_on_one_example():
    phi = np.array([[1.0], [3.0], [0.0], [2.0]])
    mem = np.array([[True], [True], [False], [False]])
    t = np.array([2.5])
    s = eb.independent_lira_scores(t, phi, mem)
    sd_in = sd_out = 1.0                                   # population std of {1,3} and {0,2}
    expected = (-0.5 * ((2.5 - 2.0) / sd_in) ** 2 - np.log(sd_in)) - (-0.5 * ((2.5 - 1.0) / sd_out) ** 2 - np.log(sd_out))
    assert s[0] == pytest.approx(expected)


def test_independent_lira_separates_members_and_scores_in_the_right_direction():
    phi, mem = _toy_shadows(gap=3.0)
    rng = np.random.default_rng(5)
    target_mem = rng.random(30) < 0.5
    target = rng.normal(0, 1, 30) + 3.0 * target_mem
    for kw in ({}, {"fix_variance": True}, {"use_median": True}):
        scores = eb.independent_lira_scores(target, phi, mem, **kw)
        assert eb.auc(scores, target_mem) > 0.8
    assert eb.auc(-scores, target_mem) < 0.2


def test_independent_lira_no_signal_gives_chance_level():
    rng = np.random.default_rng(1)
    mem = rng.random((60, 200)) < 0.5
    phi = rng.normal(size=(60, 200))
    tmem = rng.random(200) < 0.5
    s = eb.independent_lira_scores(rng.normal(size=200), phi, mem, fix_variance=True)
    assert abs(eb.auc(s, tmem) - 0.5) < 0.1


def test_independent_lira_requires_enough_shadows():
    phi = np.zeros((3, 2))
    mem = np.array([[True, True], [True, False], [False, False]])
    with pytest.raises(ValueError):
        eb.independent_lira_scores(np.zeros(2), phi, mem)


def test_repo_lira_wrapper_equals_direct_call():
    phi, mem = _toy_shadows()
    t = np.random.default_rng(2).normal(size=30)
    ins = [phi[mem[:, i], i] for i in range(30)]
    outs = [phi[~mem[:, i], i] for i in range(30)]
    assert np.allclose(eb.repo_lira_scores(t, phi, mem), lira_scores(t, ins, outs))


def test_auc_helper_matches_repo_roc_summary():
    s = np.array([0.1, 0.9, 0.4, 0.6, 0.5])
    m = np.array([0, 1, 0, 1, 1])
    assert eb.auc(s, m) == roc_summary(s, m)["auc"] == 1.0


def test_bootstrap_auc_interval_contains_point_and_paired_zero_for_identical_scores():
    phi, mem = _toy_shadows()
    rng = np.random.default_rng(7)
    tm = [rng.random(30) < 0.5 for _ in range(4)]
    sc = [eb.independent_lira_scores(rng.normal(size=30) + 1.5 * m, phi, mem) for m in tm]
    ci = eb.bootstrap_auc(sc, tm, n_boot=200, seed=0)
    assert ci["low"] <= ci["mean"] <= ci["high"]
    d = eb.bootstrap_auc(sc, tm, n_boot=100, seed=0, paired_with=sc)
    assert d["mean"] == 0 and d["low"] == 0 and d["high"] == 0


def test_bootstrap_auc_paired_detects_a_weaker_attack():
    phi, mem = _toy_shadows()
    rng = np.random.default_rng(8)
    tm = [rng.random(30) < 0.5 for _ in range(4)]
    strong = [eb.independent_lira_scores(rng.normal(size=30) + 1.5 * m, phi, mem) for m in tm]
    weak = [rng.normal(size=30) for _ in tm]
    d = eb.bootstrap_auc(strong, tm, n_boot=200, seed=0, paired_with=weak)
    assert d["mean"] > 0.2 and d["low"] > 0


def test_bootstrap_mean_ci_is_deterministic_and_needs_two_values():
    a = eb.bootstrap_mean_ci([1, 2, 3, 4], n_boot=500, seed=1)
    assert a == eb.bootstrap_mean_ci([1, 2, 3, 4], n_boot=500, seed=1) and a[1] <= a[0] <= a[2]
    with pytest.raises(ValueError):
        eb.bootstrap_mean_ci([1.0])


def test_split_and_evaluate_use_first_shadows_then_targets():
    phi, mem = _toy_shadows(m=20)
    models = {"phi": phi, "member": mem, "model_id": np.arange(20)}
    sh, tg = eb.split_models(models, 16)
    assert len(sh["phi"]) == 16 and len(tg["phi"]) == 4 and tg["model_id"][0] == 16
    scores, aucs = eb.evaluate_lira(models, 16, lambda p, sp, sm: eb.independent_lira_scores(p, sp, sm, fix_variance=True))
    assert len(scores) == 4 and all(0 <= a <= 1 for a in aucs)


@pytest.mark.skipif(importlib.util.find_spec("mialibrary") is None, reason="mialibrary is only in the isolated environment")
@pytest.mark.parametrize("fix,med", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_lira_agrees_with_mialibrary(fix, med):
    phi, mem = _toy_shadows(seed=4)
    t = np.random.default_rng(3).normal(size=30)
    a = eb.independent_lira_scores(t, phi, mem, fix_variance=fix, use_median=med)
    b = eb.mialibrary_scores(t, phi, mem, fix_variance=fix, use_median=med)
    assert np.allclose(a, b, atol=1e-8)
