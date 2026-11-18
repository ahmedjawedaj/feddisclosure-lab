from pathlib import Path
import yaml


def test_core_experiment_expected_110_minus_107():
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/analytics/differencing.yaml").read_text())
    sites = config["institutions"]
    assert len(sites) == 3
    assert sum(s["rows"] for s in sites.values()) == 110
    assert sum(s["rows"] - s["positive"] for s in sites.values()) == 107
    assert config["min_cell"] == 5
