"""Stage public digits for 3 or 4 isolated Flower Docker clients without copying rows.

This privileged *host* step may inspect every public partition. It must not be
run inside the coordinator, ServerApp, gateway, or ClientApp container.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import yaml
from feddisclosure.training.datasets import prepare_partitions

ROOT = Path(__file__).resolve().parents[1]
CONTAINER_UID = 49999  # flwr/superexec "app" user


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path: Path, data_dir: Path, run_dir: Path, env_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    n_clients = int(config["clients"])
    if n_clients not in (3, 4):
        raise ValueError("Docker deployment supports three or four clients (compose.flower.yaml, compose.flower.n4.yaml)")
    data_dir, run_dir, env_path = data_dir.resolve(), run_dir.resolve(), env_path.resolve()
    if data_dir == run_dir or data_dir in run_dir.parents or run_dir in data_dir.parents:
        raise ValueError("Client data and run artifacts must have disjoint roots")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = prepare_partitions(data_dir, dataset=config["dataset"],
                                      clients=config["clients"], partition=config["partition"],
                                      seed=config["seed"], limit=config.get("source_limit"))
    else:
        manifest = json.loads(manifest_path.read_text())
    for config_key, manifest_key in (("seed", "seed"), ("partition", "partition"),
                                     ("clients", "clients"), ("source_limit", "source_limit"),
                                     ("dataset", "source")):
        if manifest.get(manifest_key) != config.get(config_key):
            raise ValueError(f"Partition manifest mismatch: {config_key}")
    if manifest.get("schema_version") != 2 or len(manifest["client_partitions"]) != n_clients:
        raise ValueError("Expected v2 metadata-only manifest and one partition per client")
    seen = set()
    variables = {"FDL_RUN_DIR": str(run_dir)}
    checked_sites = []
    for i, site in enumerate(manifest["client_partitions"]):
        if site["client_id"] != i:
            raise ValueError("Client index mismatch")
        site_dir = (data_dir / f"institution-{i:02d}").resolve()
        if not site_dir.is_dir() or not site_dir.is_relative_to(data_dir):
            raise ValueError("Client directory does not exist within the public-data root")
        for key, filename in (("sha256", "train.npz"), ("val_sha256", "val.npz")):
            expected_path = (site_dir / filename).resolve()
            if expected_path.parent != site_dir or digest(expected_path) != site[key]:
                raise ValueError(f"Client {i} data integrity failed")
            with np.load(expected_path, allow_pickle=False) as arrays:
                ids = arrays["source_idx"]
                if seen.intersection(int(x) for x in ids):
                    raise ValueError("Record reused across client partitions")
                seen.update(int(x) for x in ids)
        variables[f"FDL_SITE_{i}_PATH"] = str(site_dir)
        variables[f"FDL_SITE_{i}_TRAIN_SHA256"] = site["sha256"]
        variables[f"FDL_SITE_{i}_VAL_SHA256"] = site["val_sha256"]
        checked_sites.append({"site_id": i, "path": str(site_dir),
                              "train_sha256": site["sha256"], "val_sha256": site["val_sha256"],
                              "train_records": site["samples"], "val_records": site["val_samples"]})
    test = (data_dir / manifest["evaluator_test"]["file"]).resolve()
    if test.parent != data_dir / "evaluator_only" or digest(test) != manifest["evaluator_test"]["sha256"]:
        raise ValueError("Evaluator-only cohort was altered")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server").mkdir(exist_ok=True)
    for i in range(n_clients):
        (run_dir / "clients" / f"site-{i}").mkdir(parents=True, exist_ok=True)
    # Flower's SuperExec image runs as uid/gid 49999 ("app"). The writable output
    # mounts (server artifacts and per-site receipts, NEVER the data mounts) must be
    # writable by that user. Prefer an exact chown; fall back to 0777 on hosts that
    # cannot chown (single-host research sandbox only).
    writable = [run_dir / "server"] + [run_dir / "clients" / f"site-{i}" for i in range(n_clients)]
    for folder in writable:
        try:
            os.chown(folder, CONTAINER_UID, CONTAINER_UID)
            os.chmod(folder, 0o750)
        except PermissionError:
            os.chmod(folder, 0o777)
    contents = "\n".join(f'{k}={json.dumps(v)}' for k, v in variables.items()) + "\n"
    if env_path.exists() and env_path.read_text() != contents:
        raise FileExistsError("Existing Compose environment points to different data; use a new run path")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(contents)
    os.chmod(env_path, 0o600)
    report = {"schema_version": 1, "preparation_only": True, "client_count": n_clients,
              "manifest_sha256": digest(manifest_path), "config": config,
              "sites": checked_sites, "run_dir": str(run_dir), "env_file": str(env_path),
              "evaluator_not_mounted": True,
              "native_execution_verified": False}
    (run_dir / "deployment_preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/training/fedavg_digits_iid.yaml")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/training/digits-docker-iid-v012")
    p.add_argument("--run-dir", type=Path, default=ROOT / "reports/results/native_flower/v012")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env.flower.generated")
    a = p.parse_args()
    result = prepare(a.config, a.data_dir, a.run_dir, a.env_file)
    print(json.dumps({"prepared": True, "sites": len(result["sites"]),
                      "environment_file": str(a.env_file), "native_executed": False}, indent=2))


if __name__ == "__main__":
    main()
