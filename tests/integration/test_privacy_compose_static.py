"""Static overlay contract, explicitly NOT evidence of a Docker execution."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_protected_compose_requires_dp_only_for_all_sites():
    overlay = yaml.safe_load((ROOT / "compose.privacy.yaml").read_text())
    services = overlay["services"]
    for name in ("institution-a", "institution-b", "institution-c"):
        site = services[name]
        assert site["environment"]["FDL_DP_ENABLED"] == "1"
        assert site["build"]["dockerfile"] == "Dockerfile.privacy"
        assert "FDL_DP_TOTAL_BUDGET" in site["environment"]
    assert services["gateway"]["environment"]["FDL_DP_ENABLED"] == "1"
    assert "[privacy]" in (ROOT / "Dockerfile.privacy").read_text()
