import json
from pathlib import Path


def test_checked_in_benchmark_matches_source_claims():
    artifact = Path(__file__).resolve().parents[2] / "reports/results/sprint1_analytics_verified.json"
    recorded = json.loads(artifact.read_text())
    naive, history = recorded["results"]["naive"], recorded["results"]["history"]
    assert [r["count"] for r in naive["releases"]] == [110, 107]
    assert naive["metrics"]["inferred_positive_count"] == 3
    assert naive["metrics"]["successful_small_group_disclosure"]
    assert history["releases"][1]["released"] is False
    assert history["metrics"]["inferred_positive_count"] is None
    assert history["metrics"]["legitimate_query_rejection_rate"] == 0.5
