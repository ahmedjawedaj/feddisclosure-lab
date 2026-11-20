"""Native-deployment contract tests; they do NOT execute Flower or Docker."""
import hashlib
import json
from pathlib import Path
import pytest

from feddisclosure.training.deployment_contract import (
    checked_private_files, require_site_identity, record_client_event)
from scripts.prepare_flower_deployment import prepare
from scripts.verify_flower_topology import static_certify
from scripts.verify_flower_attestations import verify

ROOT = Path(__file__).resolve().parents[2]


def test_static_flower_topology_is_three_separate_readonly_client_mounts():
    result = static_certify(ROOT / "compose.flower.yaml")
    assert result["static_verified"] and result["client_readonly_mounts"] == 3
    assert not result["native_verified"]


def test_static_topology_rejects_private_data_mounted_on_server(tmp_path):
    content = (ROOT / "compose.flower.yaml").read_text()
    content = content.replace('target: /artifacts', 'target: /site')
    path = tmp_path / "unsafe.yml"
    path.write_text(content)
    with pytest.raises(ValueError, match="private client data"):
        static_certify(path)


def test_deployment_prepare_disjoint_and_digest_checked(tmp_path):
    import yaml
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"seed": 11, "dataset": "sklearn_digits", "partition": "iid",
                                  "clients": 3, "source_limit": 120}))
    data = tmp_path / "data"
    out = tmp_path / "run"
    env = tmp_path / "generated.env"
    record = prepare(cfg, data, out, env)
    assert record["client_count"] == 3 and record["evaluator_not_mounted"]
    assert not record["native_execution_verified"]
    assert env.stat().st_mode & 0o077 == 0
    assert len(set(s["path"] for s in record["sites"])) == 3
    assert "evaluator_only" not in env.read_text()
    assert prepare(cfg, data, out, env) == record
    (data / "institution-01" / "train.npz").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        prepare(cfg, data, out, env)


def test_deployment_site_identity_and_hashes(tmp_path):
    (tmp_path / "train.npz").write_bytes(b"train")
    (tmp_path / "val.npz").write_bytes(b"val")
    sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert checked_private_files(tmp_path, sha("train.npz"), sha("val.npz"))[0].name == "train.npz"
    with pytest.raises(RuntimeError, match="digest"):
        checked_private_files(tmp_path, "0" * 64, sha("val.npz"))
    assert require_site_identity(deployment_site_id="1", node_partition=1,
                                 node_partitions=3, expected_clients=3) == 1
    with pytest.raises(RuntimeError, match="SuperNode"):
        require_site_identity(deployment_site_id="1", node_partition=0,
                              node_partitions=3, expected_clients=3)
    with pytest.raises(RuntimeError):
        require_site_identity(deployment_site_id="1", node_partition=1,
                              node_partitions=5, expected_clients=3)


def test_three_client_run_receipts_and_missing_round(tmp_path):
    sites = []
    for i in range(3):
        digest = f"{i + 1:064x}"
        val_digest = f"{i + 12:064x}"
        sites.append({"client_id": i, "sha256": digest, "val_sha256": val_digest,
                      "samples": 25, "val_samples": 5})
        directory = tmp_path / "clients" / f"site-{i}"
        directory.mkdir(parents=True)
        for _ in range(2):
            record_client_event(str(directory), event_type="train", site_id=i,
                                run_id=17, dataset_sha256=digest, samples=25,
                                weights_sha256="f" * 64)
        record_client_event(str(directory), event_type="evaluate", site_id=i,
                            run_id=17, dataset_sha256=val_digest, samples=5)
    report = verify(tmp_path, {"client_partitions": sites}, 2)
    assert report["client_attestations_verified"] and report["run_id"] == "17"
    with pytest.raises(ValueError, match="training round"):
        verify(tmp_path, {"client_partitions": sites}, 3)
    file = tmp_path / "clients/site-0/events.jsonl"
    first = file.read_text().splitlines()[0]
    entry = json.loads(first)
    entry["samples"] = 5000
    file.write_text("\n".join([json.dumps(entry), *file.read_text().splitlines()[1:]]) + "\n")
    with pytest.raises(ValueError, match="source partition"):
        verify(tmp_path, {"client_partitions": sites}, 2)


def test_docker_build_context_whitelists_only_code():
    ignore = (ROOT / ".dockerignore").read_text()
    lines = [line.strip() for line in ignore.splitlines() if line.strip() and not line.startswith("#")]
    assert lines[0] == "**"
    assert "!src/**" in lines and "!pyproject.toml" in lines
    assert "!data/**" not in lines and "!reports/**" not in lines
    assert "!docker/flower/SuperExec.Dockerfile" in lines
