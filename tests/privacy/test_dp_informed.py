import json
import math

import numpy as np
import pytest
from scipy import stats

from feddisclosure.attacks import dp_informed as dpi

pytest.importorskip("opacus")
Phi, Phiinv = stats.norm.cdf, stats.norm.ppf


# ---------------------------------------------------------------- bound formula
def test_bound_known_values():
    assert dpi.rero_bound(1, 1.0) == 1.0
    assert dpi.rero_bound(2, 1.0) == pytest.approx(Phi(1.0), abs=1e-12)           # 0.8413447
    # independently computed: Phi(Phi^-1(0.1) + 1/sigma)
    assert dpi.rero_bound(10, 1.0) == pytest.approx(0.3892, abs=2e-4)
    assert dpi.rero_bound(10, 0.5) == pytest.approx(0.7637, abs=2e-4)
    assert dpi.rero_bound(10, 4.0) == pytest.approx(0.1511, abs=2e-4)
    assert dpi.rero_bound(10, 0.2) > 0.9998
    # huge noise: success approaches the chance level 1/u from above
    assert 0.1 < dpi.rero_bound(10, 1e4) < 0.1002


def test_bound_monotone_and_rejects_bad_input():
    s = [dpi.rero_bound(10, x) for x in (0.2, 0.5, 1, 2, 4)]
    assert all(a > b for a, b in zip(s, s[1:]))
    u = [dpi.rero_bound(x, 1.0) for x in (2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(u, u[1:]))
    with pytest.raises(ValueError):
        dpi.rero_bound(0, 1.0)
    with pytest.raises(ValueError):
        dpi.rero_bound(10, 0.0)


def test_bound_is_the_neyman_pearson_power_with_sensitivity_C():
    # Test N(0,1) against N(mu,1) with false-positive rate kappa = 1/u. Power of the best test.
    for u in (2, 10, 100):
        for sigma in (0.5, 1.0, 3.0):
            mu = 1.0 / sigma
            threshold = Phiinv(1 - 1.0 / u)
            power = 1 - Phi(threshold - mu)
            assert dpi.rero_bound(u, sigma) == pytest.approx(power, abs=1e-12)


def test_power_curve_is_concave_which_justifies_the_jensen_step():
    q = np.linspace(0.001, 0.999, 999)
    h = Phi(Phiinv(q) + 2.0)
    assert (np.diff(h, 2) < 1e-12).all()


def test_orthogonal_reference_values_and_order():
    # two orthogonal unit-norm candidates: P(Z0 + mu > Z1) = Phi(mu / sqrt 2)
    assert dpi.orthogonal_reference(2, 1.0) == pytest.approx(Phi(1 / math.sqrt(2)), abs=1e-7)
    for u in (10, 100, 1000):
        for sigma in dpi.SIGMAS:
            assert 1.0 / u <= dpi.orthogonal_reference(u, sigma) <= dpi.rero_bound(u, sigma) + 1e-9


# ---------------------------------------------------------------- optimal rule on toy candidates
def test_nearest_rule_picks_closest_and_differs_from_inner_product_when_norms_differ():
    cands = np.array([[1.0, 0.0], [0.3, 0.3]])
    r = np.array([0.5, 0.5])
    assert dpi.ip_choice(r, cands) == 0          # inner product likes the longer vector
    assert dpi.nearest_choice(r, cands) == 1     # the Bayes rule takes the closer one


def test_exact_optimal_antipodal_pair_reaches_the_bound():
    # two candidates at +C and -C along one axis, the worst case of the bound for u = 2
    c = np.zeros((2, 5))
    c[0, 0], c[1, 0] = 1.0, -1.0
    for sigma in (0.5, 1.0, 2.0):
        n = 200_000
        k = dpi.mc_optimal_successes(c, 0, sigma=sigma, draws=n, rng=np.random.default_rng(1))
        lo, hi = stats.beta.ppf(0.0005, k, n - k + 1), stats.beta.ppf(0.9995, k + 1, n - k)
        assert lo <= Phi(1 / sigma) <= hi
        assert dpi.rero_bound(2, sigma) == pytest.approx(Phi(1 / sigma), abs=1e-12)


def test_exact_optimal_limits_far_apart_and_identical_and_orthogonal():
    rng = np.random.default_rng(0)
    far = np.eye(4) * 1.0
    assert dpi.mc_optimal_successes(far, 2, sigma=0.01, draws=500, rng=rng) == 500
    # orthogonal unit candidates match the closed form reference
    n = 100_000
    k = dpi.mc_optimal_successes(np.eye(10), 3, sigma=1.0, draws=n, rng=rng)
    ref = dpi.orthogonal_reference(10, 1.0)
    assert abs(k / n - ref) < 4 * math.sqrt(ref * (1 - ref) / n)
    # huge noise: close to chance
    k = dpi.mc_optimal_successes(np.eye(10), 3, sigma=1e3, draws=n, rng=rng)
    assert abs(k / n - 0.1) < 0.01


def _success_rates(cands, true_idx, sigma, n, seed):
    rng = np.random.default_rng(seed)
    ip_hits = nn_hits = 0
    for _ in range(n):
        r = cands[true_idx] + sigma * rng.standard_normal(cands.shape[1])
        ip_hits += int(dpi.ip_choice(r, cands) == true_idx)
        nn_hits += int(dpi.nearest_choice(r, cands) == true_idx)
    return ip_hits / n, nn_hits / n


def test_exact_optimal_is_not_beaten_by_inner_product_on_clipped_candidates():
    # All candidates have norm C. Inner product and nearest candidate then pick the same candidate on every
    # draw, so their success rates are equal, and both match the Monte Carlo optimum of the same candidates.
    rng = np.random.default_rng(2)
    g = rng.standard_normal((20, 30))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    sigma, n = 0.3, 2000
    ip_rate, nn_rate = _success_rates(g, 4, sigma, n, seed=5)
    assert ip_rate == nn_rate
    assert 0.05 < nn_rate < 0.98                                   # a rate that can move, not 0 or 1
    draws = 20000
    opt = dpi.mc_optimal_successes(g, 4, sigma=sigma, draws=draws, rng=np.random.default_rng(6)) / draws
    se = math.sqrt(opt * (1 - opt) * (1 / n + 1 / draws))
    assert abs(ip_rate - opt) < 4 * se


def test_nearest_rule_beats_inner_product_in_success_rate_when_norms_differ():
    # Control for the test above: with unequal norms the two rules differ and the Bayes rule wins on rate.
    cands = np.zeros((2, 4))
    cands[0, 0], cands[1, 0] = 1.0, 0.3
    ip_rate, nn_rate = _success_rates(cands, 1, 0.3, 2000, seed=3)
    assert nn_rate > 0.8 and ip_rate < 0.3


# ---------------------------------------------------------------- candidate sets
def test_candidates_disjoint_from_client_records_and_well_formed():
    _, y = dpi.load_public_digits()
    for seed in range(60):
        inst = dpi.build_instance(seed, y)
        client = set(inst["client"].tolist())
        assert len(client) == dpi.BATCH and inst["true_id"] in client
        for name, ids in inst["candidates"].items():
            mode, u = name.split("_u")
            assert len(ids) == int(u) == len(set(ids.tolist()))
            assert int((ids == inst["true_id"]).sum()) == 1
            others = [i for i in ids.tolist() if i != inst["true_id"]]
            assert not (set(others) & client)                       # public pool is disjoint
            if mode == "same":
                assert (y[ids] == y[inst["true_id"]]).all()


def test_candidate_sets_are_nested_and_deterministic():
    _, y = dpi.load_public_digits()
    a, b = dpi.build_instance(7, y), dpi.build_instance(7, y)
    assert all((a["candidates"][k] == b["candidates"][k]).all() for k in a["candidates"])
    small = set(a["candidates"]["any_u10"].tolist()) - {a["true_id"]}
    big = set(a["candidates"]["any_u1000"].tolist())
    assert small <= big


def test_tuning_and_evaluation_seeds_are_kept_apart():
    _, y = dpi.load_public_digits()
    with pytest.raises(ValueError):
        dpi.build_instance(9000, y)
    with pytest.raises(ValueError):
        dpi.build_instance(5, y, tuning=True)
    dpi.build_instance(9000, y, tuning=True)


# ---------------------------------------------------------------- real Opacus matches clipped sum + Gaussian noise
def test_opacus_with_zero_noise_equals_the_hand_computed_clipped_sum():
    r = dpi.crosscheck_opacus_noise(sigma=0.0, trials=2)
    assert r["max_abs_difference"] < 1e-4


def test_clipping_is_active_on_the_study_model():
    x, y = dpi.load_public_digits()
    inst = dpi.build_instance(0, y, configs=())
    init = dpi.init_state(0)
    g = dpi.clipped_gradients(init, x[inst["client"]], y[inst["client"]])
    assert np.linalg.norm(g, axis=1).max() <= 1.0 + 1e-6
    # the unclipped norms are above the clip, so clipping changes the gradients
    big = dpi.clipped_gradients(init, x[inst["client"]], y[inst["client"]], clip=1e6)
    assert np.linalg.norm(big, axis=1).min() > 1.05


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_opacus_noise_is_gaussian_with_std_sigma_times_clip(sigma):
    r = dpi.crosscheck_opacus_noise(sigma=sigma, trials=200)
    n = r["trials"] * r["coordinates"]
    assert abs(r["mean"]) < 5 / math.sqrt(n)
    assert 0.98 < r["variance"] < 1.02
    assert r["ks_pvalue"] > 1e-3


# ---------------------------------------------------------------- resume
def test_resume_skips_finished_seeds_and_is_deterministic(tmp_path):
    kw = dict(sigmas=(1.0,), draws=3, configs=(("same", 10), ("any", 10)))
    assert dpi.run_trials(tmp_path, range(2), **kw)["new_records"] == 2
    first = (tmp_path / "trials_sigma1.jsonl").read_text()
    assert dpi.run_trials(tmp_path, range(3), **kw)["new_records"] == 1
    text = (tmp_path / "trials_sigma1.jsonl").read_text()
    assert text.startswith(first)                                # earlier lines untouched
    assert dpi.completed_seeds(tmp_path / "trials_sigma1.jsonl") == {0, 1, 2}
    assert dpi.run_trials(tmp_path, range(3), **kw)["new_records"] == 0
    # a fresh directory with all three seeds in one go gives identical records
    other = tmp_path / "fresh"
    dpi.run_trials(other, range(3), **kw)
    assert (other / "trials_sigma1.jsonl").read_text() == text


def test_resume_ignores_a_half_written_last_line_and_refuses_changed_settings(tmp_path):
    kw = dict(sigmas=(1.0,), draws=3, configs=(("any", 10),))
    dpi.run_trials(tmp_path, range(2), **kw)
    p = tmp_path / "trials_sigma1.jsonl"
    p.write_text(p.read_text() + '{"seed": 2, "sigma"')
    assert dpi.completed_seeds(p) == {0, 1}
    assert dpi.run_trials(tmp_path, range(3), **kw)["new_records"] == 1
    with pytest.raises(RuntimeError):
        dpi.run_trials(tmp_path, range(3), sigmas=(1.0,), draws=4, configs=(("any", 10),))


# ---------------------------------------------------------------- summary and pre-registered checks
def _fake_summary(attack_rate_scale=1.0, trials=1000):
    cells = {}
    for mode, u in dpi.CONFIGS:
        for s in dpi.SIGMAS:
            b = dpi.rero_bound(u, s)
            k = int(round(min(1.0, b * attack_rate_scale) * trials))
            rate = lambda kk, nn: dpi._rate(kk, nn)
            opt_n = 40000
            cells[f"{mode}_u{u}|sigma={s:g}"] = {
                "config": f"{mode}_u{u}", "label_mode": mode, "u": u, "sigma": s, "chance": 1 / u,
                "bound": b, "orthogonal_reference": dpi.orthogonal_reference(u, s),
                "ip_attack": rate(k, trials), "dist_attack": rate(k, trials),
                "exact_optimal": rate(int(round(min(1.0, b * attack_rate_scale) * opt_n)), opt_n)}
    return {"cells": cells}


def test_d1_flags_an_attack_that_beats_the_bound():
    ok = dpi.prereg_check(_fake_summary(0.9))
    assert ok["D1"]["pass"] and ok["D1"]["observed"]["breaches"] == []
    at_bound = dpi.prereg_check(_fake_summary(1.0))     # attack exactly at the bound with 1000 trials is fine
    assert at_bound["D1"]["pass"]
    # a cell whose attack rate is far above the bound at high noise is a breach
    s = _fake_summary(0.9)
    key = "any_u100|sigma=4"
    s["cells"][key]["ip_attack"] = dpi._rate(300, 1000)
    chk = dpi.prereg_check(s)
    assert not chk["D1"]["pass"] and chk["D1"]["observed"]["breaches"][0]["cell"] == key


def test_d5_requires_optimal_between_attack_and_bound():
    s = _fake_summary(0.9)
    assert dpi.prereg_check(s)["D5"]["pass"]
    key = "any_u10|sigma=1"
    s["cells"][key]["exact_optimal"] = dpi._rate(30000, 40000)       # optimum far above the bound
    assert not dpi.prereg_check(s)["D5"]["pass"]
    s = _fake_summary(0.9)
    s["cells"][key]["ip_attack"] = dpi._rate(380, 1000)               # attack far above the optimum
    s["cells"][key]["exact_optimal"] = dpi._rate(8000, 40000)
    assert not dpi.prereg_check(s)["D5"]["pass"]


def test_summarize_collapses_duplicate_seed_lines(tmp_path):
    kw = dict(sigmas=(1.0,), draws=3, configs=(("any", 10),))
    dpi.run_trials(tmp_path, range(2), **kw)
    p = tmp_path / "trials_sigma1.jsonl"
    p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n")
    s = dpi.summarize(tmp_path, sigmas=(1.0,), configs=(("any", 10),))
    assert s["cells"]["any_u10|sigma=1"]["ip_attack"]["trials"] == 2
    json.dumps(s)


# ---------------------------------------------------------------- D2, D3 and D4 verdict logic
def test_d2_needs_both_candidate_modes_with_wilson_lower_limit_above_0_60():
    s = _fake_summary(0.9)
    out = dpi.prereg_check(s)["D2"]
    assert out["pass"] and all(v > 0.60 for v in out["observed"]["wilson_lower_by_candidate_mode"].values())
    # same-label fails alone: the rule needs both modes
    s = _fake_summary(0.9)
    s["cells"]["same_u10|sigma=0.2"]["ip_attack"] = dpi._rate(600, 1000)
    out = dpi.prereg_check(s)["D2"]
    assert not out["pass"] and out["observed"]["wilson_lower_by_candidate_mode"]["same_u10"] < 0.60
    # any-label fails alone
    s = _fake_summary(0.9)
    s["cells"]["any_u10|sigma=0.2"]["ip_attack"] = dpi._rate(600, 1000)
    assert not dpi.prereg_check(s)["D2"]["pass"]


def test_d2_threshold_is_strictly_above_0_60():
    s = _fake_summary(0.9)
    s["cells"]["same_u10|sigma=0.2"]["ip_attack"]["wilson"] = [0.60, 0.70]
    assert not dpi.prereg_check(s)["D2"]["pass"]
    s["cells"]["same_u10|sigma=0.2"]["ip_attack"]["wilson"] = [0.6001, 0.70]
    assert dpi.prereg_check(s)["D2"]["pass"]


def test_d3_fails_when_same_label_attack_is_more_than_0_05_below_the_bound():
    # The registered outcome: any-label gap 0.022 is within the limit, same-label gap 0.055 is not.
    s = _fake_summary(0.9)
    s["cells"]["same_u10|sigma=4"]["ip_attack"] = dpi._rate(96, 1000)
    s["cells"]["any_u10|sigma=4"]["ip_attack"] = dpi._rate(129, 1000)
    chk = dpi.prereg_check(s)
    d3 = chk["D3"]
    assert not d3["pass"]
    assert d3["observed"]["any_u10"]["gap"] == pytest.approx(dpi.rero_bound(10, 4.0) - 0.129)
    assert abs(d3["observed"]["any_u10"]["gap"]) <= 0.05 < abs(d3["observed"]["same_u10"]["gap"])
    assert d3["observed"]["same_u10"]["gap"] == pytest.approx(0.055, abs=1e-3)
    # D3 is separate from the other checks, the rest still pass on this summary
    assert chk["D1"]["pass"] and chk["D4"]["pass"] and chk["D5"]["pass"]


def test_d3_passes_only_when_both_modes_are_within_0_05_and_uses_the_absolute_gap():
    s = _fake_summary(1.0)                                   # attack at the bound, gap 0 in both modes
    assert dpi.prereg_check(s)["D3"]["pass"]
    bound = dpi.rero_bound(10, 4.0)
    for mode in ("same", "any"):
        s = _fake_summary(1.0)
        s["cells"][f"{mode}_u10|sigma=4"]["ip_attack"]["rate"] = bound - 0.0499
        assert dpi.prereg_check(s)["D3"]["pass"]
        s["cells"][f"{mode}_u10|sigma=4"]["ip_attack"]["rate"] = bound - 0.0501
        assert not dpi.prereg_check(s)["D3"]["pass"]
        s["cells"][f"{mode}_u10|sigma=4"]["ip_attack"]["rate"] = bound + 0.0501   # above the bound also fails
        assert not dpi.prereg_check(s)["D3"]["pass"]


def test_d4_needs_a_lower_rate_with_non_overlapping_wilson_intervals_for_every_candidate_set():
    s = _fake_summary(0.9)
    out = dpi.prereg_check(s)["D4"]
    assert out["pass"] and len(out["observed"]) == len(dpi.CONFIGS)
    # lower but overlapping at one candidate set: fails
    s = _fake_summary(0.9)
    s["cells"]["same_u100|sigma=0.5"]["ip_attack"] = dpi._rate(53, 1000)
    s["cells"]["same_u100|sigma=4"]["ip_attack"] = dpi._rate(45, 1000)
    out = dpi.prereg_check(s)["D4"]
    assert not out["pass"] and out["observed"]["same_u100"]["lower_and_non_overlapping"] is False
    assert all(v["lower_and_non_overlapping"] for k, v in out["observed"].items() if k != "same_u100")
    # higher at sigma 4 than at sigma 0.5 for one candidate set: fails
    s = _fake_summary(0.9)
    s["cells"]["any_u1000|sigma=4"]["ip_attack"] = dpi._rate(200, 1000)
    s["cells"]["any_u1000|sigma=0.5"]["ip_attack"] = dpi._rate(50, 1000)
    out = dpi.prereg_check(s)["D4"]
    assert not out["pass"] and out["observed"]["any_u1000"]["rate_sigma4"] > out["observed"]["any_u1000"]["rate_sigma0.5"]

