"""WP8 fault matrix: detector logic, controls, intended detections and evidence-copy safety.

Everything here is small and seeded. The full 200-audit runs are in
reports/results/privacy/v0150/fault_matrix/ and are produced by scripts/run_fault_matrix.py.
"""
import json

import numpy as np
import pytest

from feddisclosure.attacks import dp_informed as dpi
from feddisclosure.evaluation import fault_matrix as fm

pytest.importorskip("opacus")
pytest.importorskip("flwr")


def _skip_if_missing(paths, what):
    """The prepared partitions under data/ and the saved model checkpoints (*.pt) are git ignored, so a fresh
    checkout does not have them. Skip with the reason instead of failing on a missing file."""
    missing = [str(p.relative_to(fm.ROOT)) for p in paths if not p.is_file()]
    return pytest.mark.skipif(bool(missing), reason=f"{what} needs inputs that are git ignored and absent in this "
                              f"checkout: {', '.join(missing)}. Regenerate them as described in docs/v0.14-reproduce.md")


MATRIX_INPUTS = [fm.DATA4 / "manifest.json", fm.DATA3 / "manifest.json",
                 fm.SECAGG_SAVED / "v014_n4" / "n4-t2-baseline" / "server" / "final_model.pt",
                 fm.SECAGG_SAVED / "v014_dropout_matrix" / "n4-t2-kill1-r2" / "server" / "final_model.pt"]
PARITY_INPUTS = [fm.DATA3 / "manifest.json", fm.NATIVE_SAVED / "run-iid-d" / "server" / "final_model.pt",
                 fm.NATIVE_SAVED / "run-iid-d" / "checkpoints" / "digits-iid-smoke-v1_final.pt"]


@pytest.fixture(scope="module")
def digits():
    return dpi.load_public_digits()


# ---------------------------------------------------------------- audit logic on synthetic data
def test_chi2_audit_has_the_stated_level_and_sees_scale_errors():
    rng = np.random.default_rng(0)
    n = 5000
    honest = [fm.chi2_variance_audit(float((rng.standard_normal(n) ** 2).sum()), n)["flagged"] for _ in range(400)]
    assert np.mean(honest) <= 0.03                      # level 0.01, so a few percent at most over 400 audits
    for scale in (0.25, 0.5, 2.0, 1.1):
        flags = [fm.chi2_variance_audit(float(((scale * rng.standard_normal(n)) ** 2).sum()), n)["flagged"]
                 for _ in range(50)]
        assert all(flags), scale
    est = fm.chi2_variance_audit(float(((0.5 * rng.standard_normal(n)) ** 2).sum()), n)["std_ratio"]
    assert est == pytest.approx(0.5, abs=0.02)


def test_chi2_audit_is_two_sided_and_validates_input():
    assert fm.chi2_variance_audit(0.4 * 1000, 1000)["flagged"]
    assert fm.chi2_variance_audit(2.5 * 1000, 1000)["flagged"]
    assert not fm.chi2_variance_audit(1000.0, 1000)["flagged"]
    with pytest.raises(ValueError):
        fm.chi2_variance_audit(1.0, 0)


def test_clip_audit_flags_only_norms_above_the_claimed_clip():
    assert not fm.clip_audit([0.9999996, 1.0, 0.5])["flagged"]
    assert fm.clip_audit([1.0, 2.4, 1.0])["flagged"]
    assert fm.clip_audit([1.0, 1.0 + 1e-3])["flagged"]


def test_black_box_rule_uses_one_sided_lower_limit_against_the_claimed_bound():
    bound = dpi.rero_bound(fm.ATTACK_U, fm.CLAIMED_SIGMA)
    assert fm.bb_lower_limit(36, 60) > bound            # 0.60 success, well above the bound for the claimed sigma
    assert fm.bb_lower_limit(24, 60) < bound            # 0.40 is not enough evidence
    assert fm.bb_lower_limit(0, 60) == 0.0


def test_condition_parameters_keep_noise_std_at_claimed_value_when_clipping_is_skipped():
    nm, c = fm.dp_actual_params("control")
    assert (nm, c) == (1.0, 1.0)
    assert fm.dp_actual_params("noise_x0.5") == (0.5, 1.0)
    assert fm.dp_actual_params("noise_x2") == (2.0, 1.0)
    nm, c = fm.dp_actual_params("clip_skipped")
    assert c == 1e6 and nm * c == pytest.approx(1.0)


def test_tuning_and_evaluation_audit_indices_are_separated():
    fm.check_audit_index(0)
    fm.check_audit_index(199)
    with pytest.raises(ValueError):
        fm.check_audit_index(9000)
    with pytest.raises(ValueError):
        fm.check_audit_index(5, tuning=True)
    fm.check_audit_index(9000, tuning=True)


# ---------------------------------------------------------------- real Opacus: control and intended detections
SHORT = {"wb_runs": 3, "block": 20, "probes": 2}


def test_no_fault_control_is_not_flagged_by_any_dp_detector(digits):
    x, y = digits
    for audit in range(4):
        d = fm.dp_decisions(fm.run_dp_audit(audit, "control", x, y, **SHORT))
        assert not d["wb_noise"]["flagged"] and not d["wb_clip"]["flagged"] and not d["bb"]["flagged"], audit
        assert 0.97 < d["wb_noise"]["std_ratio"] < 1.03


@pytest.mark.parametrize("cond,factor", [("noise_x0.25", 0.25), ("noise_x0.5", 0.5), ("noise_x2", 2.0)])
def test_wrong_noise_scale_is_flagged_by_the_whitebox_noise_audit(digits, cond, factor):
    x, y = digits
    for audit in range(3):
        d = fm.dp_decisions(fm.run_dp_audit(audit, cond, x, y, **SHORT))
        assert d["wb_noise"]["flagged"], (cond, audit)
        assert d["wb_noise"]["std_ratio"] == pytest.approx(factor, rel=0.05)
        assert not d["wb_clip"]["flagged"]


def test_skipped_clipping_is_flagged_by_the_clip_audit_and_not_by_the_noise_audit(digits):
    x, y = digits
    for audit in range(3):
        d = fm.dp_decisions(fm.run_dp_audit(audit, "clip_skipped", x, y, **SHORT))
        assert d["wb_clip"]["flagged"] and d["wb_clip"]["max_norm_over_clip"] > 1.5
        assert not d["wb_noise"]["flagged"]


def test_black_box_attack_flags_much_less_noise_than_claimed(digits):
    x, y = digits
    flags = [fm.dp_decisions(fm.run_dp_audit(a, "noise_x0.25", x, y, wb_runs=1, block=40, probes=1))["bb"]["flagged"]
             for a in range(3)]
    assert all(flags)


# ---------------------------------------------------------------- SecAgg+ simulation with the real quantiser
def test_secagg_control_is_inside_the_proven_bound_and_has_the_expected_error_distribution(digits):
    x, y = digits
    for audit in range(4):
        r = fm.run_sim_trial(audit, "control", x, y)
        assert not r["d1"] and not r["d2"], audit
        assert all(q["err_over_bound"] < 1.0 for q in r["rounds"])


def test_plaintext_fallback_is_inside_the_bound_but_not_a_secagg_error_distribution(digits):
    x, y = digits
    for audit in range(3):
        r = fm.run_sim_trial(audit, "plaintext_fallback", x, y)
        assert not r["d1"]                               # zero error is inside any error bound
        assert r["d2"] and all(q["z_stat"] < -10 for q in r["rounds"])


@pytest.mark.parametrize("cond", ["stale_model", "dropout_cached_update"])
def test_wrong_contribution_breaks_the_error_bound(digits, cond):
    x, y = digits
    r = fm.run_sim_trial(0, cond, x, y)
    assert r["d1"] and r["d2"]
    assert r["rounds"][0]["err_over_bound"] < 1.0       # the first round has no fault, so no false alarm
    assert r["rounds"][1]["err_over_bound"] > 10


# ---------------------------------------------------------------- repository gates on tampered copies
def test_tampered_copies_are_refused_inside_saved_evidence(tmp_path):
    with pytest.raises(PermissionError):
        fm.safe_tamper_root(fm.SECAGG_SAVED / "v014_copy")
    with pytest.raises(PermissionError):
        fm.safe_tamper_root(fm.NATIVE_SAVED / "run-iid-d" / "server")
    with pytest.raises(PermissionError):
        fm.safe_tamper_root(fm.ROOT / "somewhere_in_the_repo")
    assert fm.safe_tamper_root(tmp_path / "x") == tmp_path / "x"
    assert fm.safe_tamper_root(fm.OUT_DIR / "tmp") == fm.OUT_DIR / "tmp"
    target = fm.SECAGG_SAVED / "v014_should_not_exist"
    with pytest.raises(PermissionError):
        fm.copy_tree(fm.SECAGG_SAVED / "v014", target)
    assert not target.exists()


@_skip_if_missing(MATRIX_INPUTS, "the matrix gate test")
def test_gates_flag_the_intended_faults_and_saved_evidence_is_untouched(tmp_path):
    before = fm.saved_evidence_digest()
    ctl = fm.run_matrix_gate("control", tmp_path)
    assert all(ctl["channels"][k] for k in ctl["channels"] if k.startswith(("rule:", "tolerance:")))
    for variant, channel in (("wrong_threshold_low", "rule:n4-t3-kill2-r2"),
                             ("wrong_threshold_strict", "rule:n4-t2-kill2-r2"),
                             ("stale_model", "tolerance:n4-t2-kill1-r2"),
                             ("dropout_cached_update", "tolerance:n4-t2-kill1-r2")):
        res = fm.run_matrix_gate(variant, tmp_path)
        v = fm.flagged_vs_control(res["channels"], ctl["channels"])
        assert v["flagged"] and channel in v["newly_failing_channels"], variant
        for e in res["tamper_log"]:
            assert str(tmp_path) in e["file"]
    # the plaintext fallback is NOT seen by the numerical verifier: the plain mean is closer to the reference
    pt = fm.run_matrix_gate("plaintext_fallback", tmp_path)
    assert not fm.flagged_vs_control(pt["channels"], ctl["channels"])["flagged"]
    assert fm.saved_evidence_digest() == before


@_skip_if_missing(PARITY_INPUTS, "the parity gate test")
def test_parity_gate_flags_a_stale_client_and_passes_the_honest_copy(tmp_path):
    assert fm.run_parity_gate("control", tmp_path)["channels"]["numerical_parity"] is True
    stale = fm.run_parity_gate("stale_model", tmp_path)
    assert stale["channels"]["numerical_parity"] is False and stale["max_abs_diff"] > 1e-3


def test_fault_level_counts_a_fault_only_if_all_variants_are_caught():
    cell = lambda yes: {"detected": yes}
    matrix = {"a": {"fault": "F1", "B_whitebox_any": cell(True), "A_existing_gates": None, "C_informed_attack": None,
                    "D1_error_bound": None, "D2_error_distribution": None},
              "b": {"fault": "F1", "B_whitebox_any": cell(False), "A_existing_gates": None, "C_informed_attack": None,
                    "D1_error_bound": None, "D2_error_distribution": None},
              "c": {"fault": "F2", "B_whitebox_any": cell(True), "A_existing_gates": None, "C_informed_attack": None,
                    "D1_error_bound": None, "D2_error_distribution": None}}
    fpr = {k: {"rate": 0.0} for k in ("B_whitebox_any", "C_informed_attack", "D1_error_bound", "D2_error_distribution")}
    res = fm.fault_level(matrix, fpr, include_d2=True)
    assert res["detected"] == {"F1": False, "F2": True} and res["evading_variants"] == ["b"]
    # a detector with a false positive rate above 0.05 does not count
    bad = dict(fpr, B_whitebox_any={"rate": 0.2})
    assert fm.fault_level(matrix, bad, include_d2=True)["n_detected"] == 0
