"""Tests for the Flower adaptive clipping harness (v0.16 T8b)."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("flwr")
pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_flower_adaptive_clipping.py"
V1, V2, V3, V4 = (
    ("V1", "legacy", "server"),
    ("V2", "legacy", "client"),
    ("V3", "message", "server"),
    ("V4", "message", "client"),
)


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_flower_adaptive_clipping", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_counted_fraction_inverts_the_geometric_update(audit):
    for frac in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert audit.counted_fraction(0.1, audit.geo(0.1, frac)) == pytest.approx(frac, abs=1e-12)


def test_paper_values_used_in_the_preregistration(audit):
    assert audit.geo(0.1, 0.0) == pytest.approx(0.11051709180756478, rel=1e-12)
    assert audit.geo(0.1, 1.0) == pytest.approx(0.09048374180359596, rel=1e-12)
    assert 0.1 * math.exp(-2.3) == pytest.approx(0.010025884372280375, rel=1e-12)
    z, sb = 1.0, 5.0
    assert (z ** -2 - (2 * sb) ** -2) ** -0.5 == pytest.approx(1.005037815259212, rel=1e-12)


@pytest.mark.parametrize("variant", [V1, V2, V3, V4])
def test_constructor_adjustment_matches_the_paper_relation(audit, variant):
    s = audit.construct(variant, 1.0, 100)
    assert s.clipped_count_stddev == pytest.approx(5.0)
    assert s.noise_multiplier == pytest.approx(1.005037815259212, abs=1e-9)
    with pytest.raises(ValueError):
        audit.construct(variant, 1.0, 100, clipped_count_stddev=0.5)


@pytest.mark.parametrize("variant", [V1, V2, V3, V4])
def test_noise_free_clip_update_is_geometric(audit, variant):
    r = audit.Runner(variant, 0.0, 100, 0.1)
    r.run(audit.mixed(50, 50, 0.05, 0.2), mod=True)
    assert r.strategy.clipping_norm == pytest.approx(0.1, rel=1e-12)


def test_same_seed_gives_same_count_noise(audit):
    out = audit.row_a8(V1)[0]
    assert out["measured"]["same_seed_identical"] is True
    assert out["measured"]["different_seed_differs"] is True


def test_row_helper_records_both_columns(audit):
    r = audit.row("AX", V1, 1, 2, {"x": 1}, True, False, "exact")
    assert r["prediction_held"] is True and r["agrees_with_paper"] is False
    json.dumps(r)


@pytest.mark.slow
def test_quick_run_writes_every_row(tmp_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--quick"],
        check=True, capture_output=True, timeout=900,
    )
    doc = json.loads((tmp_path / "flower_adaptive_clipping.json").read_text())
    names = {r["row"] for r in doc["rows"]}
    assert {"A0", "A1a", "A2", "A3", "A4", "A4b", "A5", "A6", "A7", "A8", "A9", "A10", "A11"} <= names
    assert len(doc["script_sha256"]) == 64
