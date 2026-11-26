"""Tests for the TensorFlow Federated DP aggregator measurement (v0.16.9).

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
SCRIPT = ROOT / "scripts" / "audit_tff_dp.py"
EVIDENCE = ROOT / "reports" / "results" / "privacy" / "v0160" / "tff_dp" / "tff-0.87.0" / "tff_dp_audit.json"
HAS_TFF = importlib.util.find_spec("tensorflow_federated") is not None

_spec = importlib.util.spec_from_file_location("audit_tff_dp", SCRIPT)
tffa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tffa)


def test_value_multiplier_matches_the_documented_formula():
    assert tffa.zdelta(1.0, 5.0) == pytest.approx(1.005037815259212, rel=1e-12)
    assert tffa.zdelta(0.5, 0.5) == pytest.approx(0.5773502691896257, rel=1e-12)
    # the composed multiplier of the value sum and the count returns the configured total
    for z, sb in [(1.0, 5.0), (1.0, 10.0), (0.3, 2.0), (1.0, 0.6)]:
        zv = tffa.zdelta(z, sb)
        assert (zv ** -2 + (2 * sb) ** -2) ** -0.5 == pytest.approx(z, rel=1e-12)


def test_recover_fraction_inverts_the_geometric_update():
    for f in (0.0, 0.25, 0.5, 0.98, 1.0):
        c1 = 0.1 * math.exp(-0.2 * (f - 0.5))
        assert tffa.recover_fraction(c1, 0.1) == pytest.approx(f, abs=1e-12)


def test_rms_standard_error_and_power_figures_match_the_preregistration():
    assert tffa.se_rms(300_000) == pytest.approx(0.0012910, rel=1e-3)
    assert tffa.mde_rms(300_000, 0.005) == pytest.approx(0.00609, abs=2e-5)
    assert tffa.mde_rms(1_000_000, 0.005) == pytest.approx(0.00560, abs=2e-5)


def test_moments_of_a_normal_sample_are_small():
    x = np.random.RandomState(1).standard_normal(300_000)
    m = tffa.moments(x)
    assert abs(m["skew"]) < 0.02 and abs(m["excess_kurtosis"]) < 0.04
    assert tffa.rms(x) == pytest.approx(1.0, abs=0.005)


def test_flat_and_find_key_read_nested_results():
    r = [np.ones(3, np.float32), [np.zeros(2, np.float32)]]
    assert tffa.flat(r).shape == (5,)
    assert tffa.find_key({"a": {"clip": 0.5}}, "clip") == 0.5
    assert tffa.find_key({"a": [1, 2]}, "clip") is None


def test_epsilon_names_ignores_docstrings_and_comments(tmp_path):
    f = tmp_path / "m.py"
    f.write_text('"""epsilon in a docstring"""\n# epsilon in a comment\nx = 1\n')
    assert tffa.epsilon_names(f) == []
    f.write_text("def f(epsilon):\n    return epsilon\n")
    assert len(tffa.epsilon_names(f)) >= 2


def test_a1_cases_agree_with_the_formula():
    for cid, z, n, sb, pred in tffa.A1_CASES:
        if pred[0] != "value" or z == 0.0:
            continue
        s = 0.05 * n if sb is None else sb
        assert tffa.zdelta(z, s) == pytest.approx(pred[1], rel=1e-12), cid


def test_deviation_list_is_a_list_of_dicts():
    assert isinstance(tffa.DEVIATIONS, list) and all({"date", "item", "reason"} <= set(d) for d in tffa.DEVIATIONS)


@pytest.mark.skipif(not EVIDENCE.exists(), reason="evidence file absent")
def test_saved_evidence_matches_the_script_and_the_preregistration():
    doc = json.loads(EVIDENCE.read_text())
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert doc["preregistration_commit"] == "7a22b12"
    assert doc["quick"] is False
    ids = [r["id"] for r in doc["rows"]]
    assert ids == ["V", "F1", "F2", "F3", "F4", "F5", "F6", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "T"]
    assert doc["summary"]["checks_total"] == sum(len(r["checks"]) for r in doc["rows"])
    assert doc["environment"]["tensorflow-federated"] == "0.87.0"


@pytest.mark.skipif(not EVIDENCE.exists(), reason="evidence file absent")
def test_saved_failures_are_the_recorded_ones():
    doc = json.loads(EVIDENCE.read_text())
    failed = {(c["row"], c["check"]) for c in doc["summary"]["checks_failed"]}
    assert len(failed) == len(doc["summary"]["checks_failed"])
    for row, check in failed:
        assert any(r["id"] == row for r in doc["rows"]), (row, check)


@pytest.fixture(scope="module")
def quick_doc(tmp_path_factory):
    out = tmp_path_factory.mktemp("tff_quick")
    r = subprocess.run([sys.executable, str(SCRIPT), "--quick", "--out", str(out)], capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stderr[-500:]
    return json.loads((out / "tff_dp_audit.json").read_text())


@pytest.mark.skipif(not HAS_TFF, reason="tensorflow_federated is not importable")
def test_quick_run_writes_the_structure(quick_doc):
    assert quick_doc["quick"] is True
    ids = [x["id"] for x in quick_doc["rows"]]
    assert ids == ["V", "F1", "F2", "F3", "F4", "F5", "F6", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "T"]
    f2 = next(x for x in quick_doc["rows"] if x["id"] == "F2")
    assert f2["prediction_held"] is True


@pytest.mark.skipif(not HAS_TFF, reason="tensorflow_federated is not importable")
def test_noise_free_clip_update_has_the_paper_direction(quick_doc):
    a3 = next(x for x in quick_doc["rows"] if x["id"] == "A3")
    assert a3["prediction_held"] is True
    assert a3["measured"]["u"]["1.0"]["next_clip"] < 0.1 < a3["measured"]["u"]["0.0"]["next_clip"]
