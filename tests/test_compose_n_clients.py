import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_compose_clients import generate  # noqa: E402
from verify_flower_topology import client_count, static_certify  # noqa: E402


def test_three_client_generation_reproduces_certified_file():
    source = (ROOT / "compose.flower.yaml").read_text()
    assert generate(source, 3).split("\n", 1)[1] == source


def test_four_client_file_is_current_and_passes_static_isolation_checks():
    source = (ROOT / "compose.flower.yaml").read_text()
    assert (ROOT / "compose.flower.n4.yaml").read_text() == generate(source, 4)
    result = static_certify(ROOT / "compose.flower.n4.yaml")
    assert result["static_verified"] and result["clients"] == 4
    assert result["client_readonly_mounts"] == 4 and len(result["services"]) == 10


def test_static_check_rejects_shared_site_mount(tmp_path):
    text = (ROOT / "compose.flower.n4.yaml").read_text().replace("FDL_SITE_3_PATH", "FDL_SITE_2_PATH")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text)
    with pytest.raises(ValueError, match="not unique"):
        static_certify(bad)


def test_static_check_rejects_wrong_partition_count(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text((ROOT / "compose.flower.n4.yaml").read_text().replace("num-partitions=4", "num-partitions=3", 1))
    with pytest.raises(ValueError, match="SuperNode ID mismatch"):
        static_certify(bad)


def test_client_count_requires_three():
    with pytest.raises(ValueError):
        client_count({"clientexec-0": {}, "supernode-0": {}})
