"""Tests for the FedML DP audit harness (v0.16 P2).

The analytic tests need only scipy. The tests that touch FedML skip cleanly when
fedml is not importable. They load only fedml.core.dp through the harness stubs
and never run fedml/__init__.py.
"""
from __future__ import annotations

import ast
import collections
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_fedml_dp.py"
FLOWER_SCRIPT = ROOT / "scripts" / "audit_flower_dp.py"
HAS_FEDML = importlib.util.find_spec("fedml") is not None
needs_fedml = pytest.mark.skipif(not HAS_FEDML, reason="fedml is not importable")


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_fedml_dp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def F(audit):
    pytest.importorskip("torch")
    if not HAS_FEDML:
        pytest.skip("fedml is not importable")
    before = set(sys.modules)
    loaded = audit.load_fedml()
    yield loaded
    for name in set(sys.modules) - before:
        if name == "fedml" or name.startswith("fedml."):
            del sys.modules[name]


def _function_sources(path: Path, names: set[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}


# ---------------------------------------------------------------- math
def test_math_copies_match_the_flower_script():
    names = {"gauss_delta", "gauss_eps", "cp_upper", "cp_lower", "planned_threshold", "empirical_eps_lower"}
    ours = _function_sources(SCRIPT, names)
    theirs = _function_sources(FLOWER_SCRIPT, names)
    assert set(ours) == names == set(theirs)
    for n in names:
        assert ours[n] == theirs[n], f"{n} differs from scripts/audit_flower_dp.py"


def test_exact_gaussian_delta_matches_known_value(audit):
    assert audit.gauss_delta(1.0, 0.0) == pytest.approx(0.3829249, abs=1e-6)


def test_laplace_profile_matches_numerical_integration(audit):
    from scipy import integrate

    shift, eps = 4.0, 1.0

    def f(x):
        return max(0.5 * math.exp(-abs(x - shift)) - math.exp(eps) * 0.5 * math.exp(-abs(x)), 0.0)

    numeric = integrate.quad(f, -60, 60 + shift, points=[0.0, shift], limit=400)[0]
    assert audit.laplace_delta(shift, eps) == pytest.approx(numeric, abs=1e-6)


def test_laplace_eps_and_delta_are_inverse(audit):
    for eps_pure, delta in ((1.0, 0.1), (4.0, 1e-3)):
        eps = audit.laplace_eps(eps_pure, delta)
        assert audit.laplace_delta(eps_pure, eps) == pytest.approx(delta, rel=1e-9)


def test_laplace_with_delta_in_the_scale_still_meets_its_claim(audit):
    # scale = 1 / (eps - ln(1 - delta)) is tighter than 1 / eps, and the exact delta is 1 - sqrt(1 - delta).
    for eps, delta in ((1.0, 1e-5), (0.1, 0.5)):
        eps_pure = eps - math.log(1 - delta)
        assert audit.laplace_delta(eps_pure, eps) == pytest.approx(1 - math.sqrt(1 - delta), rel=1e-9)
        assert audit.laplace_delta(eps_pure, eps) <= delta


def test_empirical_bound_never_exceeds_truth_on_gaussian_data(audit):
    import numpy as np

    rng = np.random.default_rng(0)
    mu = 3.0
    p0, p1 = rng.normal(0, 1, 50_000), rng.normal(mu, 1, 50_000)
    out = audit.empirical_eps_lower(p0, p1, audit.planned_threshold(mu, 50_000))
    assert 0 < out["eps_lower_bound"] <= audit.gauss_eps(mu, audit.DELTA)


def test_empirical_bound_never_exceeds_truth_on_laplace_data(audit):
    import numpy as np

    rng = np.random.default_rng(1)
    shift = np.full(audit.DIM, 0.25)
    b, n = 1.0, 50_000
    x0 = rng.laplace(0, b, (n, audit.DIM))
    x1 = rng.laplace(0, b, (n, audit.DIM)) + shift
    thr = audit.laplace_planned_threshold(b, shift, n)
    out = audit.empirical_eps_lower(audit.lr_stat(x0, shift, b), audit.lr_stat(x1, shift, b), thr)
    assert 0 < out["eps_lower_bound"] <= float(np.abs(shift).sum() / b)


def test_network_block_raises_in_a_subprocess():
    code = (
        "import importlib.util, socket\n"
        f"s = importlib.util.spec_from_file_location('a', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "m.block_network()\n"
        "try:\n    socket.socket()\nexcept RuntimeError:\n    print('blocked')\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "blocked"


# ---------------------------------------------------------------- fedml
@needs_fedml
def test_only_fedml_core_dp_is_loaded(F):
    assert "fedml" in sys.modules and not hasattr(sys.modules["fedml"], "__file__")
    assert not any("mlops" in m or m.endswith("fedml.arguments") for m in F.loaded_modules)
    assert all(m.startswith(("fedml.core.dp", "fedml.core.common", "fedml.core")) or m == "fedml" for m in F.loaded_modules)


@needs_fedml
def test_gaussian_scale_is_the_classical_formula_and_rejects_epsilon_above_one(F):
    delta = 1e-5
    g = F.DPMechanism("gaussian", 1.0, delta, sensitivity=2.0)
    assert g.dp.scale == pytest.approx(math.sqrt(2 * math.log(1.25 / delta)) * 2.0)
    for eps in (1.0000001, 2, 10):
        with pytest.raises(ValueError):
            F.DPMechanism("gaussian", eps, delta, sensitivity=1)


@needs_fedml
def test_laplace_scale_includes_delta(F):
    m = F.DPMechanism("laplace", 1.0, 0.1, sensitivity=3.0)
    assert m.dp.scale == pytest.approx(3.0 / (1.0 - math.log(0.9)))


@needs_fedml
def test_dp_clip_scale_is_divided_on_every_call(F):
    args = types.SimpleNamespace(clipping_norm=1.0, noise_multiplier=1.0, train_data_num_in_total=1000,
                                 client_num_per_round=10, client_num_in_total=100)
    frame = F.DP_Clip(args)
    frame.compute_noise((4,), 100.0)
    frame.compute_noise((4,), 100.0)
    assert frame._scale == pytest.approx(1.0 / 100.0**2)


@needs_fedml
def test_dp_clip_add_global_noise_returns_nothing_and_leaves_the_input_noise_free(F):
    import torch

    args = types.SimpleNamespace(clipping_norm=1.0, noise_multiplier=1.0, train_data_num_in_total=1000,
                                 client_num_per_round=10, client_num_in_total=100)
    model = collections.OrderedDict(a=torch.ones(5))
    assert F.DP_Clip(args).add_global_noise(model) is None
    assert torch.allclose(model["a"], torch.full((5,), 1.0 / 100.0))


@needs_fedml
def test_dp_clip_is_not_selected_by_the_local_or_global_predicates(F):
    args = types.SimpleNamespace(enable_dp=True, dp_solution_type="dp_clip", delta=1e-5, clipping_norm=1.0,
                                 noise_multiplier=1.0, train_data_num_in_total=1000,
                                 client_num_per_round=10, client_num_in_total=100)
    inst = F.FedMLDifferentialPrivacy()
    inst.init(args)
    assert inst.is_dp_enabled()
    assert not inst.is_local_dp_enabled() and not inst.is_global_dp_enabled()


@needs_fedml
def test_global_dp_cannot_construct_its_accountant_and_works_without_it(F):
    base = dict(mechanism_type="gaussian", epsilon=1.0, delta=1e-5, sensitivity=0.1,
                client_num_per_round=10, client_num_in_total=100)
    ok = F.GlobalDP(types.SimpleNamespace(**base))
    assert ok.is_rdp_accountant_enabled is False
    with pytest.raises(TypeError, match="dp_param"):
        F.GlobalDP(types.SimpleNamespace(enable_rdp_accountant=True, rdp_alpha=2.0, sigma=1.0,
                                         max_grad_norm=1.0, clipping=True, **base))


@needs_fedml
def test_accountant_get_epsilon_fails_after_a_step(F):
    args = types.SimpleNamespace(sigma=1.0, max_grad_norm=1.0, clipping=True)
    acc = F.RDP_Accountant(dp_param=1.0, alpha=2.0, dp_mechanism="gaussian", args=args)
    assert acc.get_epsilon(1e-5) == (0, 0)
    acc.step(noise_multiplier=1.0, sample_rate=1.0)
    with pytest.raises(TypeError):
        acc.get_epsilon(1e-5)


@needs_fedml
def test_conversion_function_is_an_upper_bound_on_exact_epsilon(F, audit):
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    for sigma in (1.0, 4.0):
        rdp = F.compute_rdp(q=1.0, noise_multiplier=sigma, steps=1, orders=alphas)
        eps = float(F.RDP_Accountant.get_privacy_spent(orders=alphas, rdp=rdp, delta=1e-5)[0])
        exact = audit.gauss_eps(1.0 / sigma, 1e-5)
        assert exact <= eps <= 1.12 * exact


@needs_fedml
@pytest.mark.slow
def test_quick_run_reproduces_the_headline_rows(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path), "--quick"],
                   check=True, capture_output=True, timeout=900)
    doc = json.loads((tmp_path / "fedml_dp_audit.json").read_text())
    assert doc["environment"]["fedml_init_executed"] is False
    assert doc["environment"]["network_blocked_during_rows"] is True
    by = {r["row"]: r for r in doc["rows"] if r["row"] not in ("LD1", "LD2")}
    assert by["K3"]["returned"] is None
    assert by["K4"]["flags"]["dp_clip"]["is_local_dp_enabled"] is False
    assert by["CD2"]["exception"].startswith("TypeError")
    assert by["RA3"]["worst_relative_difference"] <= 1e-6
    assert by["G2"]["cells"][0]["accepted"] is True
    # Statistical checks need the full sample size. Deterministic checks must hold in the quick run too.
    deterministic = {"G2", "LP3", "K1", "K2", "K3", "K4", "CD1", "CD2", "CD3", "RA2", "RA3", "RA4", "OB1"}
    for row in deterministic:
        failed = [c for c in by[row]["checks"] if not c["held"]]
        assert not failed, (row, failed)
    ld = [r for r in doc["rows"] if r["row"] == "LD1"]
    assert [r["configured"]["update_norm"] for r in ld] == [1.0, 5.0, 25.0]
