"""Tests for the Flower adaptive clipping guarantee harness (v0.16 T8c)."""
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

pytest.importorskip("flwr")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_flower_adaptive_guarantee.py"
EVIDENCE = ROOT / "reports" / "results" / "privacy" / "v0160" / "flower_adaptive_guarantee"
PREREG = ROOT / "docs" / "preregistration" / "v0.16-adaptive-guarantee.md"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_flower_adaptive_guarantee", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- formulas
def test_composed_multiplier_formulas(audit):
    zd = (1.0 ** -2 - 10.0 ** -2) ** -0.5
    assert audit.zpaper(zd, 5.0) == pytest.approx(1.0, rel=1e-12)
    assert audit.zflower(zd, 5.0) == pytest.approx((1 + 0.75 * (1.0 / 5.0) ** 2) ** -0.5, rel=1e-12)
    assert audit.zflower(zd, 5.0) == pytest.approx(0.98533, abs=1e-5)


def test_flower_count_is_never_above_the_paper_count_reading(audit):
    for zeta in (0.2, 1.0, 3.0):
        for sb in (0.6, 5.0, 50.0):
            assert audit.zflower(zeta, sb) < audit.zpaper(zeta, sb) < zeta


def test_preregistered_table_values(audit):
    """The predicted cell means in the pre-registration come from these formulas."""
    for sb, lr, want in ((5.0, 0.2, 0.89476), (0.6, 1.0, 0.52644), (10.0, 0.05, 0.97192)):
        zd = (1.0 - (2 * sb) ** -2) ** -0.5
        er = math.exp(-lr * 0.5 + lr ** 2 * (sb / 100) ** 2 / 2)
        assert audit.zflower(zd * er, sb) == pytest.approx(want, abs=2e-5)


def test_labels_and_seed_rule(audit):
    assert audit.label(0.98) == "equal at this tolerance"
    assert audit.label(0.9799) == "differs from configured"
    seeds = {audit.cell_seed(a, b, c, j) for a in range(4) for b in range(3) for c in range(2) for j in range(10)}
    assert len(seeds) == 240
    assert audit.cell_seed(0, 0, 0, 0) == 20261201


def test_preregistration_exists_and_names_the_rows():
    text = PREREG.read_text()
    for token in ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "## Deviation log"):
        assert token in text


# ---------------------------------------------------------------- what Flower sends
V_ALL = ["V1", "V2", "V3", "V4"]


def variant(audit, name):
    return next(v for v in audit.ADAPTIVE + audit.FIXED if v[0] == name)


def test_draw_log_does_not_change_the_stream(audit):
    np.random.seed(7)
    a = [np.random.normal(0, 1) for _ in range(3)]
    np.random.seed(7)
    with audit.DrawLog() as log:
        b = [np.random.normal(0, 1) for _ in range(3)]
    assert a == b and len(log.calls) == 3
    assert np.random.normal is not log  # restored


@pytest.mark.parametrize("name", V_ALL)
def test_count_location_is_the_plain_number_of_clipped_updates(audit, name):
    v = variant(audit, name)
    r = audit.Runner(v, 1.0, 100, 1.0, clipped_count_stddev=5.0)
    with audit.DrawLog() as log:
        r.run(audit.NORMS, bits=audit.client_bits(audit.NORMS, 1.0))
    (loc, scale, _), = log.count_draws()
    assert loc == 50.0 and scale == 5.0


@pytest.mark.parametrize("name", ["V2", "V4"])
def test_real_client_mods_send_bits_zero_and_one(audit, name):
    v = variant(audit, name)
    r = audit.Runner(v, 1.0, 100, 1.0, clipped_count_stddev=5.0)
    norms = [0.5] * 50 + [1.5] * 50
    r.run(norms, mod=True)
    assert sorted({float(b) for b in r.sent_bits}) == [0.0, 1.0]
    assert [float(b) for b in r.sent_bits] == [0.0] * 50 + [1.0] * 50


def test_client_bits_helper_matches_the_clip_rule(audit):
    bits = audit.client_bits([0.4, 1.0, 1.0000001, 2.0], 1.0)
    assert bits == [0, 0, 1, 1]


@pytest.mark.parametrize("name", V_ALL)
def test_one_client_moving_outside_changes_the_location_by_one(audit, name):
    v = variant(audit, name)
    locs = []
    for outside in (False, True):
        norms = [0.5] * 100
        if outside:
            norms[3] = 1.5
        r = audit.Runner(v, 1.0, 100, 1.0, clipped_count_stddev=5.0)
        with audit.DrawLog() as log:
            r.run(norms, bits=audit.client_bits(norms, 1.0))
        locs.append(log.count_draws()[0][0])
    assert locs[1] - locs[0] == 1.0


# ---------------------------------------------------------------- trajectories
def test_noise_free_clip_used_is_the_clip_before_the_call(audit):
    for name in V_ALL + ["F1", "F2", "F3", "F4"]:
        v = variant(audit, name)
        r = audit.Runner(v, 0.0, 100, 0.2) if v[3] else audit.Runner(v, 0.0, 100, 0.2, clip_norm_lr=0.2)
        before = r.strategy.clipping_norm
        out = r.run(audit.NORMS, bits=audit.client_bits(audit.NORMS, 0.2) if (v[2] == "client" and not v[3]) else None,
                    mod=False) if v[2] == "server" else r.run(audit.NORMS, mod=True)
        want = float(np.mean([min(u, before) for u in audit.NORMS]))
        assert out[0] == pytest.approx(want, rel=1e-5)


def test_trajectory_identity_and_agreement_between_variants(audit):
    seqs = []
    for name in V_ALL:
        t = audit.run_trajectory(variant(audit, name), 5.0, 0.2, 0.1, 99, 4)
        seqs.append(t["C"])
        c = np.array(t["C"])
        ft = np.array(t["ret"]) / 100
        assert np.allclose(c[1:] / c[:-1], np.exp(-0.2 * (ft - 0.5)), rtol=1e-12, atol=0)
        assert all(x == 100.0 for x in t["loc"])  # every update is clipped from a start of 0.1
    for other in seqs[1:]:
        assert np.allclose(seqs[0], other, rtol=1e-9, atol=0)


def test_all_clipped_rounds_shrink_the_clip_by_the_preregistered_mean(audit):
    t = audit.run_trajectory(variant(audit, "V1"), 5.0, 0.2, 0.1, 5, 30)
    c = np.array(t["C"])
    r = c[1:] / c[:-1]
    pred = math.exp(-0.2 * 0.5 + 0.2 ** 2 * (5.0 / 100) ** 2 / 2)
    assert r.mean() == pytest.approx(pred, abs=4 * r.std(ddof=1) / math.sqrt(r.size) + 1e-12)


def test_fixed_variants_keep_the_clip_and_release_no_count(audit):
    for name in ("F1", "F2", "F3", "F4"):
        res = audit.run_fixed(variant(audit, name), 1.0, 0.1, 11, 3)
        assert res["clip_unchanged"] is True
        assert res["count_draws"] == 0
        assert res["zeta_ratio_max_error"] <= 1e-12


def test_constructor_grid_row_holds_for_every_variant(audit):
    for name in V_ALL:
        (r,) = audit.rows_g6(variant(audit, name))
        assert r["prediction_held"] is True
        assert r["measured"]["count"] == 21


def test_row_helper_records_both_columns(audit):
    r = audit.row("GX", variant(audit, "V1"), 1, 2, {"x": 1}, True, False, "exact", lr=0.2)
    assert r["prediction_held"] is True and r["agrees_with_paper"] is False and r["lr"] == 0.2
    json.dumps(r)


# ---------------------------------------------------------------- saved evidence
def _evidence(version):
    path = EVIDENCE / f"flwr-{version}" / "flower_adaptive_guarantee.json"
    if not path.exists():
        pytest.skip("evidence not present")
    return json.loads(path.read_text())


@pytest.mark.parametrize("version", ["1.27.0", "1.44.0"])
def test_saved_evidence_is_complete_and_matches_the_script(version):
    doc = _evidence(version)
    assert doc["environment"]["flwr"] == version
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert doc["quick"] is False and len(doc["rows"]) == 640
    assert doc["parameters"]["trajectories"] == 10 and doc["parameters"]["rounds"] == 30
    names = {r["row"] for r in doc["rows"]}
    assert {"G0a", "G0b", "G0c", "G0d", "G0e", "G1", "G2a", "G2b", "G2c", "G3a", "G3b", "G3c", "G4", "G5a", "G5b",
            "G6", "G7"} <= names
    assert len(doc["cells"]) == 96


@pytest.mark.parametrize("version", ["1.27.0", "1.44.0"])
def test_saved_rows_hold_and_the_default_cell_has_the_reported_ratios(version):
    doc = _evidence(version)
    assert all(r["prediction_held"] for r in doc["rows"])
    cell = next(c for c in doc["cells"] if c["variant"] == "V1" and c["sigma_b"] == 5.0 and c["lr"] == 0.2
                and c["c0"] == 0.1)
    assert cell["rho_delta"]["mean"] == pytest.approx(0.9054, abs=5e-4)
    assert cell["z_flower_over_z"]["mean"] == pytest.approx(0.8952, abs=5e-4)
    assert cell["z_paper_count_over_z"]["mean"] == pytest.approx(0.9062, abs=5e-4)


def test_cross_version_comparison_is_saved_and_held():
    path = EVIDENCE / "cross_version" / "cross_version.json"
    if not path.exists():
        pytest.skip("evidence not present")
    doc = json.loads(path.read_text())
    assert doc["prediction_held"] is True and doc["cells_compared"] == 96
    assert doc["versions"] == ["1.27.0", "1.44.0"]


@pytest.mark.slow
def test_quick_run_writes_every_row(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path), "--quick"],
                   check=True, capture_output=True, timeout=900)
    doc = json.loads((tmp_path / "flower_adaptive_guarantee.json").read_text())
    names = {r["row"] for r in doc["rows"]}
    assert {"G0b", "G1", "G2a", "G3a", "G4", "G5a", "G6", "G7"} <= names
    assert len(doc["script_sha256"]) == 64
    # G4 is a statistical row and can miss its tolerance at the small quick-run size
    assert all(r["prediction_held"] for r in doc["rows"] if r["row"] != "G4")
