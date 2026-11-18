"""Recorded reproducibility assertion ignores random run IDs/timing."""
from pathlib import Path
import json


def test_separately_executed_runs_match_on_scientific_outcomes():
    folder = Path(__file__).resolve().parents[2] / "reports/results"
    def outcomes(filename):
        raw = json.loads((folder / filename).read_text())
        return {
            policy: {
                "release_counts": [value["count"] for value in row["releases"]],
                "inference": row["metrics"]["inferred_positive_count"],
                "small_group_exposed": row["metrics"]["successful_small_group_disclosure"],
                "rejects": row["metrics"]["rejected_queries"],
            } for policy, row in raw["results"].items()
        }

    expected = outcomes("sprint1_analytics_verified.json")
    observed = outcomes("sprint1_analytics_repeat_verified.json")
    assert observed == expected
