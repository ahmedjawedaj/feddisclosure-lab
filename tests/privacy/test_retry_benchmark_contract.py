"""This exercises an injected FAKE measurement, never OpenDP guarantees."""
from pathlib import Path
import json
from scripts.benchmark_retry_safety import run


def test_benchmark_repeats_identical_noisy_reply(tmp_path):
    out = tmp_path / "benchmark.json"
    result = run(out)
    assert json.loads(out.read_text()) == result
    assert result["engine"].endswith("NOT_DP")
    assert result["legacy"]["http_statuses"] == [409, 200, 409]
    assert result["retry_safe"]["http_statuses"] == [409, 200, 200]
    assert result["retry_safe"]["gateway_noisy_values"][1] == result["retry_safe"]["gateway_noisy_values"][2]
    assert all(b["spent_epsilon"] == .25 for b in result["retry_safe"]["site_budgets"].values())
    assert all(b["spent_epsilon"] == .5 for b in result["legacy"]["site_budgets"].values())


def test_retry_benchmark_fails_if_result_looks_like_formal_dp():
    source = (Path(__file__).resolve().parents[2] / "scripts/benchmark_retry_safety.py").read_text()
    assert "NO DP guarantee" in source
    assert "_test_mechanism=TestOnlyMeasurement()" in source
