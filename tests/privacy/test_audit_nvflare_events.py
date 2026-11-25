"""Tests for the SVTPrivacy events check (v0.16.8).

The statistics and numeric tests need only numpy. The saved evidence tests read the JSON that the full run wrote.
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
SCRIPT = ROOT / "scripts" / "audit_nvflare_events.py"
EVIDENCE = ROOT / "reports" / "results" / "privacy" / "v0160" / "nvflare_events" / "nvflare_events.json"
HAS_NVFLARE = importlib.util.find_spec("nvflare") is not None

sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("audit_nvflare_events", SCRIPT)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def test_ceiling_is_far_above_every_claim():
    n = 500_000
    lo, _ = ev.interval(n, n)
    _, hi = ev.interval(0, n)
    assert 11.63 < math.log(lo / hi) < 11.66
    assert max(c["claim"] for c in ev.ROWS.values()) < 2.0


def test_lower_bound_is_zero_for_equal_counts_and_positive_for_a_clear_gap():
    assert ev.lower_bound(1000, 10_000, 1000, 10_000)["lb"] == 0.0
    assert 0.5 < ev.lower_bound(2000, 10_000, 1000, 10_000)["lb"] < math.log(2.0)


def test_min_detectable_matches_the_preregistered_values():
    for rid, pc in (("E1", 0.003669), ("E2", 0.006340), ("E3", 0.004069), ("E4", 0.006267), ("E5", 0.107588), ("E6", 0.007041)):
        cfg = ev.ROWS[rid]
        assert abs(ev.min_detectable(pc, 500_000, cfg["claim"])["epsilon_80"] - cfg["mde"]) < 0.002, rid


def test_power_is_small_at_the_claim_and_high_above_it():
    assert ev.power_at(1.0, 0.05, 500_000, 1.0) < 0.02
    assert ev.power_at(1.1, 0.05, 500_000, 1.0) > 0.99


def test_chi2_df4_survival_and_block_agreement():
    assert abs(ev.chi2_df4_sf(13.2767) - 0.01) < 1e-4
    ba = ev.block_agreement([100, 100, 100, 100, 100], 1000)
    assert ba["chi2"] == 0.0 and ba["p_value"] == 1.0


def test_numeric_computation_gives_the_uniform_value_for_equal_inputs():
    et, eq = ev.split(1.0, 2)
    p, alive = ev.count_prob(0, 0, 2, 6, 2, et, eq, False, grid=400, kmax=600)
    assert abs(p - 1 / math.comb(6, 2)) < 1e-4 and alive < 1e-6


def test_numeric_computation_reproduces_the_v0165_values():
    et, eq = ev.split(1.0, 2)
    pa = ev.count_prob(1, 0, 2, 6, 2, et, eq, False, grid=3000)[0]
    pc = ev.count_prob(0, 1, 2, 6, 2, et, eq, False, grid=3000)[0]
    assert abs(pa - 0.08354) < 2e-5 and abs(pc - 0.05321) < 2e-5


def test_at_least_one_special_entry_has_probability_one_for_a_full_draw():
    # h 0 means every outcome counts, so the probability is the chance that the loop stops, which is 1
    et, eq = ev.split(1.0, 2)
    p, _ = ev.count_prob(1, 0, 2, 6, 0, et, eq, False, grid=400, kmax=600)
    assert abs(p - 1.0) < 1e-6


def test_release_closed_form_matches_a_direct_sum_for_one_entry():
    a = 0.05
    assert abs(ev.p_pos(a) - (1 - 0.5 * math.exp(-a))) < 1e-15
    cfg = ev.ROWS["E5"]
    pa, pc = ev.numeric_probs(cfg)
    assert abs(math.log(pa / pc) / cfg["claim"] - 0.9762) < 1e-3
    assert abs(pa - 0.1442) < 1e-4 and abs(pc - 0.1076) < 1e-4


def test_event_tests_on_hand_made_outputs():
    import numpy as np
    s = ev.event_test("set", 2, 2)
    assert s(np.array([1.0, -1.0, 0.0, 0.0])) and not s(np.array([1.0, 0.0, 0.0, 0.0])) and not s(np.array([1.0, 1.0, 1.0, 0.0]))
    a = ev.event_test("atleast", 3, 2)
    assert a(np.array([1.0, 1.0, 0.0, 5.0])) and not a(np.array([1.0, 0.0, 0.0, 5.0]))
    p = ev.event_test("allpos", 2, 2)
    assert p(np.array([0.0, 1.0, 2.0])) and not p(np.array([0.0, -1.0, 2.0])) and not p(np.array([1.0, 1.0, 2.0]))
    j = ev.event_test("joint", 2, 2)
    assert j(np.array([1.0, 2.0, 0.0])) and not j(np.array([1.0, -2.0, 0.0])) and not j(np.array([1.0, 2.0, 3.0]))


def test_rows_have_the_preregistered_sizes():
    n_of = {rid: math.ceil(c["d"] * c["fraction"]) for rid, c in ev.ROWS.items()}
    assert n_of == {r: ev.ROWS[r]["n"] for r in ev.ROWS} and [ev.ROWS[r]["n"] for r in ev.ROWS] == [4, 6, 4, 10, 3, 2]
    for c in ev.ROWS.values():
        assert len(c["a"]) == len(c["c"]) == c["d"]


def test_deviation_list_is_a_list_of_dicts():
    assert isinstance(ev.DEVIATIONS, list)


@pytest.mark.skipif(not EVIDENCE.exists(), reason="evidence file absent")
def test_saved_evidence_matches_the_script_and_the_preregistration():
    doc = json.loads(EVIDENCE.read_text())
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert doc["preregistration_commit"] == "a1d1ea5ecb8c774e68536b52d2d94596c56eb419"
    assert doc["quick"] is False and doc["parameters"]["runs_per_block"] == 100_000
    assert [r["id"] for r in doc["rows"]] == ["K0", "E1", "E2", "E3", "E4", "E5", "E6", "C", "P", "T"]
    assert doc["summary"]["checks_held"] == doc["summary"]["checks_total"] == 83
    assert doc["replication"] == [] and doc["summary"]["excess_found_rows"] == []
    for r in doc["rows"]:
        if r["id"].startswith("E"):
            m = r["measured"]
            assert m["interval_and_lb"]["lb"] <= m["claim"]
            assert m["worlds"]["a"]["calls"] == m["worlds"]["c"]["calls"] == 500_000


@pytest.mark.skipif(not HAS_NVFLARE, reason="nvflare is not importable")
def test_quick_run_writes_the_structure(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--quick", "--out", str(tmp_path)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-500:]
    doc = json.loads((tmp_path / "nvflare_events.json").read_text())
    assert doc["quick"] is True
    assert [r["measured"]["ledger_first_call"]["n_upload"] for r in doc["rows"] if r["id"].startswith("E")] == [4, 6, 4, 10, 3, 2]
    e1 = next(r for r in doc["rows"] if r["id"] == "E1")
    assert e1["measured"]["worlds"]["a"]["calls"] == 10_000
