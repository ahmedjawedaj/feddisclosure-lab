"""Tests for the P2 server view harness (v0.16.13, rows S4, S6 and S6b).

Fast tests. They skip when flwr or scipy is missing. The one test that needs Docker skips when the
Docker CLI or the daemon is missing. The S4 checks are exercised on the saved evidence of the real run and
on altered copies that must fail (positive controls).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("flwr")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_p2_server_view.py"
EVIDENCE = ROOT / "reports" / "results" / "privacy" / "v0160" / "p2" / "flwr-1.27.0"
PREREG = ROOT / "docs" / "preregistration" / "v0.16-p2.md"


@pytest.fixture(scope="module")
def audit():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("audit_p2_server_view", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- formulas and constants
def test_preregistered_epsilon_values_come_from_the_formula(audit):
    assert audit.p1.gauss_eps(1.0, 1e-5) == pytest.approx(audit.PREDICTIONS["S6b_eps"]["M10"], abs=2e-6)
    assert audit.p1.gauss_eps(10 / 9, 1e-5) == pytest.approx(audit.PREDICTIONS["S6b_eps"]["M9"], abs=2e-6)


def test_power_numbers_of_the_s6_rule(audit):
    n = audit.ROUNDS
    kmin = next(k for k in range(n + 1) if audit.wilson_interval(k, n)[0] > 0.5)
    assert kmin == 1044
    from scipy import stats

    assert stats.binom.sf(kmin - 1, n, 0.531) == pytest.approx(0.80, abs=0.02)
    assert stats.binom.sf(kmin - 1, n, 0.5) == pytest.approx(0.0259, abs=1e-3)


def test_balanced_label_threshold(audit):
    lo, hi = audit.R_LO, audit.R_HI
    assert (hi - audit.THRESHOLD) / (hi - lo) == pytest.approx(0.5)


# ---------------------------------------------------------------- driving the real strategy
def test_reported_failure_releases_nothing_and_absence_releases_a_rescaled_aggregate(audit):
    import numpy as np

    drv = audit.Driver(0.0)
    rng = np.random.default_rng(0)
    ups = {i: (3 * audit.unit(rng)).astype("float32") for i in range(1, 10)}
    released, cids = drv.round(dict(ups), failed=True)
    assert released is None and len(cids) == 9
    released, cids = drv.round(dict(ups))
    assert released is not None and len(cids) == 9
    # Nine clipped updates of norm 1, equal weights, so the mean is their average and the shift is 1/9 for one client.
    mean = np.mean([u / np.linalg.norm(u) for u in ups.values()], axis=0)
    assert np.allclose(released, mean, atol=1e-5)


def test_clip_check_passes(audit):
    assert audit.s6_clip(50)["within_1e-5"] == 50


def test_dependent_arm_is_above_chance_and_control_is_not(audit):
    dep = audit.run_arm("bit-dep", 0, True, True, 600)
    ctl = audit.run_arm("bit-ctl", 1, True, False, 600)
    assert dep["wilson95"][0] > 0.5 and dep["exact_release_rule_rounds"] == 600
    assert ctl["wilson95"][0] <= 0.5 <= ctl["wilson95"][1]
    sdep = audit.run_arm("set-dep", 2, False, True, 600)
    assert sdep["wilson95"][0] > 0.5 and sdep["exact_release_rule_rounds"] == 600
    assert sdep["share_9_results"] == pytest.approx(1 - sdep["share_participated"])


def test_s6b_configuration_follows_the_result_count(audit):
    m10, m9 = audit.s6b_config(10, 400), audit.s6b_config(9, 400)
    assert m10["sensitivity_zero_out"] == pytest.approx(0.1, rel=1e-3)
    assert m9["sensitivity_zero_out"] == pytest.approx(1 / 9, rel=1e-3)
    assert m9["effective_noise_multiplier"] / m10["effective_noise_multiplier"] == pytest.approx(0.9, rel=1e-3)
    assert m9["eps_at_delta"] > m10["eps_at_delta"]


# ---------------------------------------------------------------- S4 on saved evidence
def _run_dirs():
    return [EVIDENCE / "s4_docker" / "runs" / f"seed{s}-secagg" for s in (0, 1)]


needs_evidence = pytest.mark.skipif(not _run_dirs()[0].is_dir(), reason="saved S4 run directories are missing")


@needs_evidence
def test_saved_s4_runs_pass_with_four_of_four_checks(audit):
    for d in _run_dirs():
        res = audit.check_s4_run(d)
        assert res["status"] == "RUN" and res["pass_"]
        assert res["checks_true"] == {"S4a": 4, "S4b": 4, "S4c": 4, "S4d": 4}
        assert all(set(x["metrics_at_grid"]) == set(audit.S4_KEYS) for x in res["replies"])
        assert all(sum(x["array_records"].values()) == 0 for x in res["replies"])


def _copy_run(tmp_path: Path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(_run_dirs()[0], dst, ignore=shutil.ignore_patterns("relay", "*.pt"))
    return dst


def _rewrite_messages(run: Path, fn) -> None:
    path = run / "server" / "wire" / "messages.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    for r in rows:
        fn(r)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _is_collect_reply(r) -> bool:
    return r["direction"] == "in" and r["stage"] == "collect_masked_vectors"


@needs_evidence
def test_s4_fails_when_the_metrics_are_missing_from_a_reply(audit, tmp_path):
    run = _copy_run(tmp_path)

    def drop(r):
        if _is_collect_reply(r):
            r["content"]["config_records"]["fitres.metrics"] = {}

    _rewrite_messages(run, drop)
    res = audit.check_s4_run(run)
    assert not res["pass_"] and res["checks_true"]["S4a"] == 0


@needs_evidence
def test_s4_fails_when_a_value_differs_from_the_receipt(audit, tmp_path):
    run = _copy_run(tmp_path)
    seen = False

    def bump(r):
        nonlocal seen
        if _is_collect_reply(r) and not seen:
            r["content"]["config_records"]["fitres.metrics"]["train_loss"] += 1e-9
            seen = True

    _rewrite_messages(run, bump)
    res = audit.check_s4_run(run)
    assert not res["pass_"] and res["checks_true"]["S4b"] == 3


@needs_evidence
def test_s4_fails_when_an_array_record_is_not_empty(audit, tmp_path):
    run = _copy_run(tmp_path)

    def fill(r):
        if _is_collect_reply(r):
            r["content"]["array_records"]["fitres.parameters"] = {"0": {"nbytes": 4}}

    _rewrite_messages(run, fill)
    res = audit.check_s4_run(run)
    assert not res["pass_"] and res["checks_true"]["S4d"] == 0


@needs_evidence
def test_s4_fails_when_the_strategy_saw_different_metrics(audit, tmp_path):
    run = _copy_run(tmp_path)
    path = run / "server" / "wire" / "fit_results_seen.json"
    doc = json.loads(path.read_text())
    doc["results"][0]["metrics"] = {}
    path.write_text(json.dumps(doc))
    res = audit.check_s4_run(run)
    assert not res["pass_"] and res["checks_true"]["S4c"] == 3
    path.unlink()
    res = audit.check_s4_run(run)
    assert not res["pass_"] and res["checks_true"]["S4c"] == 0


@needs_evidence
def test_incomplete_capture_is_not_run_and_not_a_pass(audit, tmp_path):
    run = _copy_run(tmp_path)
    meta = json.loads((run / "wire_run.json").read_text())
    meta["ok"] = False
    (run / "wire_run.json").write_text(json.dumps(meta))
    res = audit.check_s4_run(run)
    assert res["status"] == "NOT RUN" and not res["pass_"]
    verdict = audit.s4_verdict([run], 1)
    assert verdict["status"] == "NOT RUN" and not verdict["pass"]


def test_missing_directories_give_not_run(audit, tmp_path):
    verdict = audit.s4_verdict([tmp_path / "nothing"], 1)
    assert verdict["status"] == "NOT RUN" and not verdict["pass"]


# ---------------------------------------------------------------- metric client and server hook
def test_probe_metrics_have_the_registered_keys():
    pytest.importorskip("torch")
    from feddisclosure.training.p2_metrics_client import PROBE_BASE, probe_metrics

    m = probe_metrics(2, 2.25, 8)
    assert set(m) == {"probe_const", "train_loss", "n_local"}
    assert m["probe_const"] == PROBE_BASE + 2 and isinstance(m["n_local"], int) and isinstance(m["train_loss"], float)


def test_server_hook_writes_what_aggregate_fit_receives(tmp_path):
    from feddisclosure.training.p2_metrics_server import SEEN_NAME, make_metrics_strategy

    class Parent:
        def aggregate_fit(self, server_round, results, failures):
            return "parameters", {}

    class Proxy:
        def __init__(self, node_id):
            self.node_id = node_id
            self.cid = str(node_id)

    class Res:
        def __init__(self, n, metrics):
            self.num_examples, self.metrics = n, metrics

    cls = make_metrics_strategy(tmp_path / "wire", Parent)
    out = cls().aggregate_fit(1, [(Proxy(7), Res(8, {"probe_const": 1234.5, "n_local": 8}))], [])
    assert out == ("parameters", {})
    seen = json.loads((tmp_path / "wire" / SEEN_NAME).read_text())
    assert seen["results"] == [{"node_id": "7", "cid": "7", "num_examples": 8, "metrics": {"probe_const": 1234.5, "n_local": 8}}]


def test_metrics_app_points_at_the_p2_modules():
    text = (ROOT / "docker" / "wire_relay" / "app_metrics" / "pyproject.toml").read_text()
    assert 'serverapp = "feddisclosure.training.p2_metrics_server:app"' in text
    assert 'clientapp = "feddisclosure.training.p2_metrics_client:app"' in text


def test_dry_run_prints_the_compose_commands_with_the_metrics_app(tmp_path):
    pytest.importorskip("yaml")
    env = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_wire_capture.py"), "--seeds", "1", "--modes", "secagg",
                          "--app-dir", str(ROOT / "docker" / "wire_relay" / "app_metrics"), "--output", str(tmp_path), "--dry-run"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "app_metrics:/alt-app:ro" in out and "use-secagg=false" not in out and "dry run: nothing was executed" in out


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI missing")
@needs_evidence
def test_compose_files_validate_with_the_wire_override(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)}
    env_file = EVIDENCE / "s4_docker" / "env" / "seed0-secagg.env"
    r = subprocess.run(["docker", "compose", "--env-file", str(env_file), "-f", str(ROOT / "compose.flower.n4.yaml"), "-f", str(ROOT / "compose.wire.yaml"),
                        "config", "-q"], capture_output=True, text=True, env=env, cwd=ROOT)
    if "Cannot connect" in r.stderr or "docker daemon" in r.stderr.lower():
        pytest.skip("Docker daemon unavailable")
    assert r.returncode == 0, r.stderr[-400:]


# ---------------------------------------------------------------- saved results and the pre-registration
@pytest.mark.skipif(not (EVIDENCE / "p2_server_view.json").is_file(), reason="saved results are missing")
def test_saved_results_are_consistent_and_bound_to_the_script():
    doc = json.loads((EVIDENCE / "p2_server_view.json").read_text())
    assert doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert not doc["quick"] and doc["parameters"]["rounds_per_arm"] == 2000 and doc["parameters"]["noise_runs"] == 20000
    assert doc["environment"]["flwr"] == "1.27.0"
    assert doc["S6_verdicts"]["S6"]["pass"] == all(doc["S6_verdicts"][k]["pass"] for k in ("S6-bit", "S6-set", "S6-ctl", "S6-exact", "S6-clip"))
    assert doc["S6b_verdicts"]["S6b"]["pass"] == all(doc["S6b_verdicts"][k]["pass"] for k in ("S6b-sens", "S6b-sigma", "S6b-zeff", "S6b-eps", "S6b-mix"))
    for arm in doc["S6_arms"].values():
        assert arm["correct"] == round(arm["accuracy"] * 2000)
        lo, hi = arm["wilson95"]
        assert lo < arm["accuracy"] < hi


def test_preregistration_has_predictions_power_and_an_empty_start_of_the_deviation_log():
    text = PREREG.read_text()
    spec = importlib.util.spec_from_file_location("release_gate", ROOT / "scripts" / "release_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    heads = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert any("Predictions" in h for h in heads) and any("Power" in h for h in heads)
    assert text.index("## Deviation log") > text.index("## Power of each null result") > text.index("## Predictions")
    assert not gate.prereg_power_missing(text)
