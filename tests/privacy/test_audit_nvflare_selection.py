"""Tests for the SVTPrivacy selection check (v0.16.5).

The statistics tests need only numpy. The saved evidence tests read the JSON that the full run wrote.
The test that calls the real filter skips when nvflare is not importable.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_nvflare_selection.py"
EVIDENCE = ROOT / "reports" / "results" / "privacy" / "v0160" / "nvflare_selection" / "nvflare_selection.json"
HAS_NVFLARE = importlib.util.find_spec("nvflare") is not None

sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("audit_nvflare_selection", SCRIPT)
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)


def test_ceiling_of_the_old_design_is_below_its_claim():
    lo, _ = sel.interval(200_000, 200_000)
    _, hi = sel.interval(0, 200_000)
    assert 10.72 < math.log(lo / hi) < 10.74 < 20


def test_lower_bound_is_zero_for_equal_counts_and_positive_for_a_clear_gap():
    same = sel.lower_bound(1000, 10_000, 1000, 10_000)
    assert same["lb"] == 0.0
    gap = sel.lower_bound(2000, 10_000, 1000, 10_000)
    assert 0.5 < gap["lb"] < math.log(2.0)


def test_min_detectable_matches_the_preregistered_values():
    assert abs(sel.min_detectable(0.053209, 500_000, 1.0)["epsilon_80"] - 1.021) < 0.002
    assert abs(sel.min_detectable(0.036918, 500_000, 1.0)["epsilon_80"] - 1.026) < 0.002
    assert abs(sel.min_detectable(0.048909, 500_000, 0.1)["epsilon_80"] - 0.126) < 0.002


def test_power_is_one_half_near_the_critical_value_and_small_at_the_claim():
    pc, claim = 0.05, 1.0
    assert sel.power_at(claim, pc, 500_000, claim) < 0.02
    assert sel.power_at(1.1, pc, 500_000, claim) > 0.99


def test_chi2_df4_survival_and_block_agreement():
    assert abs(sel.chi2_df4_sf(13.2767) - 0.01) < 1e-4
    ba = sel.block_agreement([100, 100, 100, 100, 100], 1000)
    assert ba["chi2"] == 0.0 and ba["p_value"] == 1.0


def test_numeric_computation_gives_n_over_d_for_equal_inputs():
    et, eq = sel.split(1.0, 2)
    p, alive = sel.support_prob([0] * 6, (0, 1), 2, et, eq, False, grid=400, kmax=600)
    assert abs(p - 1 / math.comb(6, 2)) < 1e-4 and alive < 1e-6


def test_split_matches_the_documented_formula():
    et, eq = sel.split(1.0, 2)
    assert abs(et - 1 / (1 + 4 ** (2 / 3))) < 1e-15 and abs(et + eq - 1.0) < 1e-15


def test_deviation_list_is_a_list_of_dicts():
    assert isinstance(sel.DEVIATIONS, list)


@pytest.mark.skipif(not EVIDENCE.exists(), reason="evidence file absent")
def test_saved_evidence_matches_the_script_and_the_preregistration():
    doc = json.loads(EVIDENCE.read_text())
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert doc["preregistration_commit"] == "cdd2ce031f7da61d642f1afe5b20f950a90dab08"
    assert doc["quick"] is False and doc["parameters"]["runs_per_block"] == 100_000
    ids = [r["id"] for r in doc["rows"]]
    assert ids == ["S0", "S1", "S2", "S3", "C", "V", "T"]
    assert doc["summary"]["checks_held"] == doc["summary"]["checks_total"]


@pytest.mark.skipif(not HAS_NVFLARE, reason="nvflare is not importable")
def test_quick_run_writes_the_structure(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--quick", "--skip-versions", "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    doc = json.loads((tmp_path / "nvflare_selection.json").read_text())
    assert doc["quick"] is True
    s1 = next(r for r in doc["rows"] if r["id"] == "S1")
    assert s1["measured"]["worlds"]["a"]["calls"] == 10_000
    assert s1["measured"]["ledger_first_call"]["n_upload"] == 2
