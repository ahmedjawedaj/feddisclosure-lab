"""Tests for the TensorFlow Federated distributed DP aggregator measurement (v0.16.11).

The helper and evidence tests need only numpy and the standard library. The tests that call the real
aggregators skip when tensorflow_federated is not importable, as in the CI environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_tff_ddp.py"
EVIDENCE = ROOT / "reports" / "results" / "privacy" / "v0160" / "tff_ddp" / "tff-0.87.0" / "tff_ddp_audit.json"
HAS_TFF = importlib.util.find_spec("tensorflow_federated") is not None
ROW_IDS = ["V", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "T"]

_spec = importlib.util.spec_from_file_location("audit_tff_ddp", SCRIPT)
ddp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ddp)


def test_scale_and_local_stddev_at_the_default_configuration():
    sc, sl, k = ddp.local_scaled(10, 1.0, 1.0, 16, 1e-4)
    assert k == pytest.approx(3.8906, abs=1e-4)
    assert sc == pytest.approx(8223.98, rel=1e-5)
    assert sl == pytest.approx(2600.651, rel=1e-5)


def test_scale_is_capped_at_one_million():
    assert ddp.scale_fn(20, 20, 0.1, 0.0, 3.89) == 1e6


@pytest.mark.parametrize("bits", [4, 5, 8, 12])
def test_scale_and_noise_ratio_match_the_preregistered_table(bits):
    scale, sl, sk, dg, integer = ddp.PRE_A2[bits]
    sc, loc, _ = ddp.local_scaled(10, 1.0, 1.0, bits, 1e-4)
    assert sc == pytest.approx(scale, rel=1e-4)
    assert loc == pytest.approx(sl, rel=1e-4)
    assert math.ceil(loc) == integer
    assert ddp.noise_ratio_pred(ddp.SKELLAM, 10, 1.0, 1.0, bits, 1e-4) == pytest.approx(sk, rel=5e-5)
    assert ddp.noise_ratio_pred(ddp.DGAUSS, 10, 1.0, 1.0, bits, 1e-4) == pytest.approx(dg, rel=5e-5)


def test_noise_ratio_for_fewer_and_more_clients_matches_the_preregistered_values():
    # m 20 is slightly below the square root of 2 because the larger sum wraps a little more often
    assert ddp.noise_ratio_pred(ddp.SKELLAM, 10, 1.0, 1.0, 16, 1e-4, 5) == pytest.approx(0.70711, abs=2e-5)
    assert ddp.noise_ratio_pred(ddp.SKELLAM, 10, 1.0, 1.0, 16, 1e-4, 20) == pytest.approx(1.40273, abs=2e-5)


def test_rounding_growth_for_a_dense_input_matches_the_preregistered_values():
    for bits, pred in ddp.PRE_A3.items():
        assert ddp.dense_ratio_pred(10, 1.0, 1.0, bits, 1e-4, 0.99) == pytest.approx(pred, abs=1e-4)


def test_rho_is_one_sixth_for_a_wide_input_and_the_mean_distance_for_a_narrow_one():
    assert ddp.rho_exact(60.0) == pytest.approx(1 / 6)
    assert ddp.rho_exact(0.05) == pytest.approx(0.05 * math.sqrt(2 / math.pi), rel=0.1)


def test_sensitivity_bound_matches_the_preregistered_values():
    k = ddp.kstd(1e-4)
    for (bits, flag), (_, _, pbound) in ddp.PRE_D.items():
        beta = ddp.BETA if flag else 0.0
        sc = ddp.scale_fn(bits, 10, 1.0, 0.0, k)
        assert ddp.inflated_bound(1.0, sc, beta) == pytest.approx(pbound, rel=5e-5)


def test_conditional_bound_never_exceeds_the_deterministic_bound():
    for bits in (5, 8, 16):
        sc = ddp.scale_fn(bits, 10, 1.0, 0.0, ddp.kstd(1e-4))
        assert ddp.inflated_bound(1.0, sc, ddp.BETA) <= ddp.inflated_bound(1.0, sc, 0.0) + 1e-12


def test_wrap_and_saturation_values_match_the_preregistered_ones():
    for p, (wrap, clip) in ddp.PRE_E2.items():
        assert ddp.noise_ratio_pred(ddp.SKELLAM, 10, 1.0, 1.0, 16, p) == pytest.approx(wrap, rel=5e-5)
        assert ddp.clip_alt_ratio(10, 1.0, 1.0, 16, p) == pytest.approx(clip, rel=5e-5)


def test_modular_overflow_norms_match_the_preregistered_values():
    for m, (pred, tol) in ddp.PRE_E3.items():
        assert ddp.e3_wrap_norm(4, 8, m) == pytest.approx(pred, abs=tol)
    assert ddp.e3_wrap_norm(4, 8, 64, clip=True) > ddp.e3_wrap_norm(4, 8, 64)


def test_smallest_accepted_bits_follow_the_constructor_rule():
    expected = {1: 2, 2: 3, 4: 3, 10: 4, 100: 7, 1000: 10, 10000: 14}
    assert {n: ddp.min_bits(n) for n in expected} == expected


def test_discrete_tables_are_normalised_and_have_the_stated_variance():
    xs, pm = ddp.dg_pmf(11)
    assert pm.sum() == pytest.approx(1.0)
    assert float(np.sum(pm * xs.astype(float) ** 2)) == pytest.approx(121.0, rel=2e-3)
    xs10, pm10 = ddp.conv_n(xs, pm, 10)
    assert float(np.sum(pm10 * xs10.astype(float) ** 2)) == pytest.approx(1210.0, rel=2e-3)
    ks, ps = ddp.skellam_sum_pmf(50.0, 200)
    assert ps.sum() == pytest.approx(1.0)
    assert float(np.sum(ps * ks.astype(float) ** 2)) == pytest.approx(100.0, rel=1e-3)


def test_wrap_leaves_small_noise_unchanged_and_shrinks_large_noise():
    assert ddp.gauss_wrap_rms2(2.0, 128)[0] == pytest.approx(4.0, rel=1e-3)
    wrapped, raw = ddp.gauss_wrap_rms2(100.0, 128)
    assert wrapped < raw


def test_value_multiplier_matches_the_documented_formula():
    assert ddp.zdelta(1.0, 5.0) == pytest.approx(1.005037815259212, rel=1e-12)
    zv = ddp.zdelta(1.0, 1.0)
    assert (zv ** -2 + 2.0 ** -2) ** -0.5 == pytest.approx(1.0, rel=1e-12)


def test_gaussian_trade_off_matches_the_preregistered_values():
    assert ddp.gdelta(1.0, 1.0) == pytest.approx(0.126937, abs=1e-6)
    assert ddp.geps(1.0, 1e-5) == pytest.approx(4.37718, abs=1e-4)
    assert ddp.geps(2.0, 1e-5) == pytest.approx(9.99726, abs=1e-3)
    assert ddp.gdelta(0.0, 1.0) == 0.0
    assert ddp.geps(0.2, 1e-5) < ddp.geps(1.0, 1e-5)


def test_power_figures_match_the_preregistration():
    assert ddp.mde_rms(245_760, 0.007) == pytest.approx(0.0082, abs=2e-5)
    assert ddp.mde_rms(163_840, 0.007) == pytest.approx(0.00847, abs=2e-5)


def test_moments_corr_and_trapz_on_known_inputs():
    x = np.random.RandomState(1).standard_normal(300_000)
    m = ddp.moments(x)
    assert abs(m["skew"]) < 0.02 and abs(m["excess_kurtosis"]) < 0.04
    assert ddp.rms(x) == pytest.approx(1.0, abs=0.005)
    assert ddp.corr(x, x) == pytest.approx(1.0)
    t = np.linspace(0.0, 1.0, 1001)
    assert ddp.trapz(t ** 2, t) == pytest.approx(1 / 3, abs=1e-6)


def test_flat_find_key_and_walk_names_read_nested_results():
    r = [np.ones(3, np.float32), [np.zeros(2, np.float32)]]
    assert ddp.flat(r).shape == (5,)
    assert ddp.find_key({"a": {"clip": 0.5}}, "clip") == 0.5
    assert ddp.find_key({"a": [1, 2]}, "clip") is None
    names = {nm for _, nm, _ in ddp.walk_names({"a": {"b": 1}, "c": [2]})}
    assert {"a", "b", "c"} <= names


def test_epsilon_names_ignores_docstrings_and_comments(tmp_path):
    f = tmp_path / "m.py"
    f.write_text('"""epsilon in a docstring"""\n# epsilon in a comment\nx = 1\n')
    assert ddp.epsilon_names(f) == []
    f.write_text("def f(epsilon):\n    return epsilon\n")
    assert len(ddp.epsilon_names(f)) >= 2


def test_deviation_list_is_a_list_of_dicts():
    assert isinstance(ddp.DEVIATIONS, list) and all({"date", "item", "reason"} <= set(d) for d in ddp.DEVIATIONS)


def test_quick_mode_requires_an_output_directory():
    r = subprocess.run([sys.executable, str(SCRIPT), "--quick"], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "--out" in r.stderr


@pytest.mark.skipif(not EVIDENCE.exists(), reason="evidence file absent")
def test_saved_evidence_matches_the_script_and_the_preregistration():
    doc = json.loads(EVIDENCE.read_text())
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert doc["preregistration_commit"] == "b32e31c"
    assert doc["quick"] is False
    assert [r["id"] for r in doc["rows"]] == ROW_IDS
    assert doc["summary"]["checks_total"] == sum(len(r["checks"]) for r in doc["rows"])
    assert doc["environment"]["tensorflow-federated"] == "0.87.0"


@pytest.mark.skipif(not EVIDENCE.exists(), reason="evidence file absent")
def test_saved_failures_are_listed_once_and_belong_to_a_row():
    doc = json.loads(EVIDENCE.read_text())
    failed = [(c["row"], c["check"]) for c in doc["summary"]["checks_failed"]]
    assert len(set(failed)) == len(failed)
    for row, _ in failed:
        assert any(r["id"] == row for r in doc["rows"])
    assert doc["summary"]["checks_failed"] == [{"row": r["id"], **c} for r in doc["rows"] for c in r["checks"] if not c["held"]]


@pytest.fixture(scope="module")
def quick_doc(tmp_path_factory):
    out = tmp_path_factory.mktemp("tff_ddp_quick")
    r = subprocess.run([sys.executable, str(SCRIPT), "--quick", "--out", str(out)], capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stderr[-500:]
    return json.loads((out / "tff_ddp_audit.json").read_text())


@pytest.mark.skipif(not HAS_TFF, reason="tensorflow_federated is not importable")
def test_quick_run_writes_the_structure(quick_doc):
    assert quick_doc["quick"] is True
    assert [x["id"] for x in quick_doc["rows"]] == ROW_IDS
    row = {x["id"]: x for x in quick_doc["rows"]}
    assert row["V"]["prediction_held"] is True and row["G"]["prediction_held"] is True


@pytest.mark.skipif(not HAS_TFF, reason="tensorflow_federated is not importable")
def test_quick_run_constructor_limits_and_exact_measurements(quick_doc):
    row = {x["id"]: x for x in quick_doc["rows"]}
    e1 = [c for c in row["E"]["checks"] if c["check"].startswith("E1")]
    assert e1 and all(c["held"] for c in e1)
    exact = [c for c in row["A"]["checks"] if c["kind"] == "exact" and "scale_factor" in c["check"]]
    assert exact and all(c["held"] for c in exact)
