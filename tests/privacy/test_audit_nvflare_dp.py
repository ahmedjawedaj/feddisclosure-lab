"""Tests for the NVFlare DP audit harness (v0.16 T10).

The analytic tests need only numpy, and a comparison with scipy runs when scipy
is installed. The tests that touch NVFlare skip cleanly when nvflare is not
importable. They run the harness in quick mode in a subprocess and check the
structure and the deterministic rows.
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
SCRIPT = ROOT / "scripts" / "audit_nvflare_dp.py"
EVIDENCE = (ROOT / "reports" / "results" / "privacy" / "v0160" / "nvflare_dp_audit" / "nvflare-2.9.1"
            / "nvflare_dp_audit.json")
HAS_NVFLARE = importlib.util.find_spec("nvflare") is not None
needs_nvflare = pytest.mark.skipif(not HAS_NVFLARE, reason="nvflare is not importable")

ALL_ROWS = ["N1", "N2", "N3", "N4", "N5", "N6a", "N6b", "N7", "N8", "N9", "N10", "N11", "N12",
            "P0", "P1", "P2", "P3", "P4"]
DETERMINISTIC_ROWS = ["N1", "N2", "N3", "N8", "N9", "N10", "N11", "P0", "P1", "P2", "P3", "P4"]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_nvflare_dp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def quick_doc(tmp_path_factory):
    if not HAS_NVFLARE:
        pytest.skip("nvflare is not importable")
    out = tmp_path_factory.mktemp("nvflare_quick")
    subprocess.run([sys.executable, str(SCRIPT), "--quick", "--out", str(out)],
                   check=True, capture_output=True, timeout=600)
    return json.loads((out / "nvflare_dp_audit.json").read_text())


# ---------------------------------------------------------------- analytic
def test_incomplete_beta_matches_known_values(audit):
    # I_x(1, 1) is x. I_x(2, 1) is x^2. I_x(1, 2) is 1 - (1 - x)^2.
    assert audit.betainc(1, 1, 0.3) == pytest.approx(0.3, rel=1e-12)
    assert audit.betainc(2, 1, 0.3) == pytest.approx(0.09, rel=1e-12)
    assert audit.betainc(1, 2, 0.3) == pytest.approx(1 - 0.7**2, rel=1e-12)


def test_clopper_pearson_closed_forms(audit):
    # With zero events the upper bound is 1 - a^(1/n). With n events the lower bound is a^(1/n).
    assert audit.cp_upper(0, 1000, 0.025) == pytest.approx(1 - 0.025 ** (1 / 1000), rel=1e-9)
    assert audit.cp_lower(1000, 1000, 0.025) == pytest.approx(0.025 ** (1 / 1000), rel=1e-9)
    assert audit.cp_lower(0, 50, 0.025) == 0.0
    assert audit.cp_upper(50, 50, 0.025) == 1.0


def test_clopper_pearson_matches_scipy_when_installed(audit):
    stats = pytest.importorskip("scipy.stats")
    for k, n in ((0, 200_000), (3, 200_000), (1_000, 2_000), (99_000, 200_000), (7, 40)):
        assert audit.cp_upper(k, n, 0.025) == pytest.approx(
            1.0 if k >= n else float(stats.beta.ppf(0.975, k + 1, n - k)), rel=1e-8)
        assert audit.cp_lower(k, n, 0.025) == pytest.approx(
            0.0 if k <= 0 else float(stats.beta.ppf(0.025, k, n - k + 1)), rel=1e-8)


def test_empirical_bound_stays_below_truth_on_simulated_events(audit):
    rng = np.random.default_rng(0)
    p_a, p_b, n = 0.6, 0.2, 200_000
    k_a, k_b = int(rng.binomial(n, p_a)), int(rng.binomial(n, p_b))
    out = audit.empirical_eps_lower(k_a, n, k_b, n)
    assert 0 < out["eps_lower_bound"] <= math.log(p_a / p_b)
    assert out["eps_lower_bound"] > 0.9 * math.log(p_a / p_b)


def test_empirical_bound_takes_the_larger_direction(audit):
    out = audit.empirical_eps_lower(1_000, 10_000, 5_000, 10_000)
    assert out["eps_lower_a_over_b"] == 0.0 and out["eps_lower_b_over_a"] > 0.0
    assert out["eps_lower_bound"] == out["eps_lower_b_over_a"]


def test_quick_mode_requires_an_output_directory():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--quick"], capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0 and "--out" in proc.stderr


# ---------------------------------------------------------------- nvflare
@needs_nvflare
def test_recorder_passes_calls_through_and_restores(audit):
    F = audit.load_nvflare()
    original = np.random.laplace
    np.random.seed(5)
    expected = np.random.laplace(scale=2.0, size=3)
    np.random.seed(5)
    with audit.record_randomness(F) as rec:
        got = np.random.laplace(scale=2.0, size=3)
    assert np.array_equal(got, expected)
    assert rec.laplace == [{"scale": 2.0, "size": 3}]
    assert np.random.laplace is original


@needs_nvflare
def test_quick_run_structure(quick_doc):
    doc = quick_doc
    assert doc["schema"] == "feddisclosure.nvflare_dp_audit.v1"
    assert doc["quick"] is True
    for key in ("environment", "parameters", "script_sha256", "summary", "rows", "runtime_seconds"):
        assert key in doc
    assert doc["environment"]["nvflare"]
    assert {"python", "numpy", "nvflare", "platform"} <= set(doc["environment"])
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert [r["id"] for r in doc["rows"]] == ALL_ROWS
    for r in doc["rows"]:
        for key in ("id", "configuration", "prediction", "tolerance", "measured", "prediction_held", "note", "checks"):
            assert key in r, (r["id"], key)
        assert isinstance(r["prediction_held"], bool)
    s = doc["summary"]
    assert s["checks_total"] == sum(len(r["checks"]) for r in doc["rows"])
    assert s["checks_held"] + len(s["checks_failed"]) == s["checks_total"]
    assert s["rows_total"] == len(ALL_ROWS)


@needs_nvflare
def test_quick_run_deterministic_rows_hold(quick_doc):
    by = {r["id"]: r for r in quick_doc["rows"]}
    for row in DETERMINISTIC_ROWS:
        failed = [c for c in by[row]["checks"] if not c["held"]]
        assert not failed, (row, failed)
        assert by[row]["prediction_held"] is True


@needs_nvflare
def test_quick_run_headline_values(quick_doc):
    by = {r["id"]: r for r in quick_doc["rows"]}
    assert by["N3"]["measured"]["ledger_one_call"]["n_upload"] == 2558
    assert by["N3"]["measured"]["privacy_spent_ten_calls"] == pytest.approx(2559.0, abs=1e-9)
    assert by["N9"]["measured"]["ledger"]["n_upload"] == 11
    assert [c["outcome"] for c in by["N5"]["measured"]["cells"]][-1] == "timed out"
    assert by["N6a"]["measured"]["real_process_dxo_per_run"] is True


@needs_nvflare
def test_filter_source_hashes_match_the_preregistration(quick_doc):
    # The pre-registration compares these two files by hash across releases. 2.9.1 is the version run.
    env = quick_doc["environment"]
    if env["nvflare"] != "2.9.1":
        pytest.skip("measured version is not 2.9.1")
    assert env["nvflare_svt_privacy_sha256"].startswith("c543bbea2ce3bfe2")
    assert env["nvflare_percentile_privacy_sha256"].startswith("29bcf457e2c20ff8")


# ---------------------------------------------------------------- evidence
@pytest.mark.skipif(not EVIDENCE.exists(), reason="evidence file is not committed yet")
def test_committed_evidence_pins_the_check_counts():
    doc = json.loads(EVIDENCE.read_text())
    assert doc["quick"] is False
    assert doc["summary"]["checks_total"] == 63
    assert doc["summary"]["checks_held"] + len(doc["summary"]["checks_failed"]) == 63
    assert doc["summary"]["rows_total"] == 18
    assert [r["id"] for r in doc["rows"]] == ALL_ROWS
    assert doc["parameters"]["mc_runs_per_input_n6"] == 200_000
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
