"""The release gate workflow keeps its trigger, its checkout and its base argument."""
from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github/workflows/release-gate.yml"


def test_gate_workflow_shape():
    doc = yaml.safe_load(WF.read_text())
    triggers = doc.get(True, doc.get("on"))
    assert "pull_request" in triggers
    steps = doc["jobs"]["gate"]["steps"]
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["fetch-depth"] == 0
    assert "pull_request.head.sha" in checkout["with"]["ref"]
    run = next(s["run"] for s in steps if "release_gate.py" in s.get("run", ""))
    assert "pull_request.base.sha" in run
