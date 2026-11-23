"""Tests for the Flower DP audit harness (v0.16 P1)."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("flwr")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_flower_dp.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_flower_dp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_delta_matches_known_value(audit):
    # For mu = 1 and eps = 0 the Gaussian trade-off gives delta = 2*Phi(0.5) - 1.
    assert audit.gauss_delta(1.0, 0.0) == pytest.approx(0.3829249, abs=1e-6)


def test_eps_and_delta_are_inverse(audit):
    for mu in (0.5, 1.0, 5.0):
        eps = audit.gauss_eps(mu, 1e-5)
        assert audit.gauss_delta(mu, eps) == pytest.approx(1e-5, rel=1e-6)


def test_classical_calibration_is_conservative_only_for_small_eps(audit):
    delta = 1e-5

    def exact(eps):
        sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps
        return audit.gauss_delta(1.0 / sigma, eps)

    assert exact(1.0) < delta
    assert exact(20.0) > delta


def test_clopper_pearson_bounds_contain_the_estimate(audit):
    assert audit.cp_lower(50, 1000, 0.025) < 0.05 < audit.cp_upper(50, 1000, 0.025)
    assert audit.cp_upper(0, 1000, 0.025) == pytest.approx(0.003682, abs=1e-5)


def test_empirical_bound_never_exceeds_truth_on_gaussian_data(audit):
    import numpy as np

    rng = np.random.default_rng(0)
    mu = 3.0
    p0, p1 = rng.normal(0, 1, 50_000), rng.normal(mu, 1, 50_000)
    out = audit.empirical_eps_lower(p0, p1, audit.planned_threshold(mu, 50_000))
    assert 0 < out["eps_lower_bound"] <= audit.gauss_eps(mu, audit.DELTA)


@pytest.mark.slow
def test_quick_run_reproduces_the_headline_rows(tmp_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--quick"],
        check=True, capture_output=True, timeout=600,
    )
    rows = json.loads((tmp_path / "flower_dp_audit.json").read_text())["rows"]
    by = {r["row"]: r for r in rows if r["row"] not in ("L1", "L2")}
    assert by["C1"]["measured"]["mu_zero_out"] == pytest.approx(1.0, rel=0.1)
    assert by["C3"]["measured"]["sensitivity_zero_out"] == pytest.approx(0.91, rel=1e-3)
    assert by["C4"]["measured"]["sensitivity_zero_out"] == pytest.approx(0.5, rel=1e-3)
    assert by["A1"]["measured"]["sigma_ratio_to_reference"] == pytest.approx(0.905, rel=0.05)
    local = [r for r in rows if r["row"] == "L1"]
    assert [r["meets_configured_guarantee"] for r in local] == [True] * 4 + [False] * 2


def test_reported_failure_releases_nothing(audit):
    # Row C4 is not reached through a client failure: the wrapper returns no aggregate.
    import numpy as np

    api = audit.OldApi(audit.DifferentialPrivacyServerSideFixedClipping, audit.N, audit.Z)
    glob = [np.zeros(audit.DIM, dtype=np.float32)]
    api.strategy.configure_fit(1, audit.ndarrays_to_parameters(glob), api.manager)
    res = audit.FitRes(audit.OK, audit.ndarrays_to_parameters([glob[0] + audit.e1()]), 1, {})
    results = [(api.proxies[0], res), (api.proxies[1], res)]
    params, _ = api.strategy.aggregate_fit(1, results, [RuntimeError("client failed")])
    assert params is None
