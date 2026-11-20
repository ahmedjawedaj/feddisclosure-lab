"""Acceptance tests must fail CLOSED on absence of native artifacts."""
import json
import pytest
from scripts.verify_native_release import verify, GateIncomplete
from scripts.native_preflight import evaluate


def test_native_release_gate_does_not_accept_missing_or_fake_data(tmp_path):
    with pytest.raises(GateIncomplete, match="Missing native evidence"):
        verify(tmp_path)
    f = tmp_path / "reports/results/sprint1/docker_isolation.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"observed": True}))
    with pytest.raises(GateIncomplete, match="Missing native evidence"):
        verify(tmp_path)


def test_preflight_is_prerequisite_not_certification():
    status = evaluate()
    assert status["schema_version"] == 1
    assert "Native" not in status["disclaimer"]  # plain availability report
    assert status["all_local_prerequisites"] == all(status["local_gates"].values())
