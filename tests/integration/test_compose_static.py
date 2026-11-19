"""Static Compose safety contract only; NOT a native Docker execution test."""
from pathlib import Path
import yaml


def test_compose_static_network_and_volume_contract():
    file = Path(__file__).resolve().parents[2] / "compose.yaml"
    doc = yaml.safe_load(file.read_text())
    institutions = [doc["services"][f"institution-{letter}"] for letter in ("a", "b", "c")]
    assert all("ports" not in site for site in institutions)
    assert len({tuple(site["volumes"]) for site in institutions}) == 3
    assert doc["networks"]["internal"]["internal"] is True
    assert doc["services"]["gateway"]["ports"] == ["127.0.0.1:8080:8000"]
    assert all(site["read_only"] for site in institutions)
