"""One 8-client SecAgg+ round in the real Docker deployment, with peak container memory (row S3).

This is a thin wrapper around `scripts/run_flower_docker.py`. It reuses that runner unchanged (compose
startup, topology certification, the Flower CLI call, receipt verification, teardown) and adds two
things: staging for eight institutions, and a background `docker stats` poll. The shared preparation
helper accepts only three or four clients, so staging for eight is done here.

    python scripts/run_scaling_n8.py --output reports/results/privacy/v0150/scaling/docker_n8

Needs a Docker daemon. Run it through .github/workflows/scaling-n8.yml. Use --dry-run to print the plan.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from feddisclosure.evaluation.scaling_memory import (LIMIT_GB, PROJECT_PREFIX, STATS_FORMAT, PeakTracker,
                                                      StatsSampler, expected_container_names)
from feddisclosure.training.datasets import prepare_partitions
from feddisclosure.training.secagg_params import validate_secagg_params

CLIENTS = 8
CONFIG = ROOT / "configs/training/fedavg_digits_iid_8clients_1round.yaml"
APP_DIR = ROOT / "experiments/flower_secagg"
COMPOSE = ROOT / "compose.flower.n8.yaml"


def stage_clients(config_path: Path, data_dir: Path, run_dir: Path, env_path: Path) -> dict:
    """Same job and output as prepare_flower_deployment.prepare, for the configured client count.

    Writes one read-only dataset directory per site, the per-site attestation folders and the Compose
    environment file. Checks the same integrity properties (hashes, no record reused across sites).
    """
    from prepare_flower_deployment import CONTAINER_UID, digest

    config = yaml.safe_load(config_path.read_text())
    n = int(config["clients"])
    data_dir, run_dir, env_path = data_dir.resolve(), run_dir.resolve(), env_path.resolve()
    if data_dir == run_dir or data_dir in run_dir.parents or run_dir in data_dir.parents:
        raise ValueError("Client data and run artifacts must have disjoint roots")
    manifest_path = data_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                prepare_partitions(data_dir, dataset=config["dataset"], clients=n, partition=config["partition"],
                                   seed=config["seed"], limit=config.get("source_limit")))
    for key, mkey in (("seed", "seed"), ("partition", "partition"), ("clients", "clients"),
                      ("source_limit", "source_limit"), ("dataset", "source")):
        if manifest.get(mkey) != config.get(key):
            raise ValueError(f"Partition manifest mismatch: {key}")
    if manifest.get("schema_version") != 2 or len(manifest["client_partitions"]) != n:
        raise ValueError("Expected v2 metadata-only manifest and one partition per client")
    seen, variables, sites = set(), {"FDL_RUN_DIR": str(run_dir)}, []
    for i, site in enumerate(manifest["client_partitions"]):
        site_dir = (data_dir / f"institution-{i:02d}").resolve()
        if site["client_id"] != i or not site_dir.is_dir() or not site_dir.is_relative_to(data_dir):
            raise ValueError(f"Client {i} directory is wrong")
        for key, name in (("sha256", "train.npz"), ("val_sha256", "val.npz")):
            if digest(site_dir / name) != site[key]:
                raise ValueError(f"Client {i} data integrity failed")
            with np.load(site_dir / name, allow_pickle=False) as arrays:
                ids = {int(v) for v in arrays["source_idx"]}
            if seen & ids:
                raise ValueError("Record reused across client partitions")
            seen |= ids
        variables.update({f"FDL_SITE_{i}_PATH": str(site_dir), f"FDL_SITE_{i}_TRAIN_SHA256": site["sha256"],
                          f"FDL_SITE_{i}_VAL_SHA256": site["val_sha256"]})
        sites.append({"site_id": i, "path": str(site_dir), "train_sha256": site["sha256"],
                      "val_sha256": site["val_sha256"], "train_records": site["samples"],
                      "val_records": site["val_samples"]})
    run_dir.mkdir(parents=True, exist_ok=True)
    writable = [run_dir / "server"] + [run_dir / "clients" / f"site-{i}" for i in range(n)]
    for folder in writable:
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.chown(folder, CONTAINER_UID, CONTAINER_UID)
            os.chmod(folder, 0o750)
        except PermissionError:
            os.chmod(folder, 0o777)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(f"{k}={json.dumps(v)}" for k, v in variables.items()) + "\n")
    os.chmod(env_path, 0o600)
    report = {"schema_version": 1, "preparation_only": True, "client_count": n,
              "manifest_sha256": digest(manifest_path), "config": config, "sites": sites,
              "run_dir": str(run_dir), "env_file": str(env_path), "evaluator_not_mounted": True,
              "native_execution_verified": False}
    (run_dir / "deployment_preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def runner_argv(a, data_dir: Path, run_dir: Path, env_file: Path) -> list[str]:
    """Arguments for scripts/run_flower_docker.py. The reconstruction threshold must be below num-shares."""
    extra = (f"use-secagg=true num-shares={a.num_shares} reconstruction-threshold={a.threshold} "
             f"timeout={float(a.round_timeout)}")
    return ["run_flower_docker.py", "--config", str(CONFIG), "--data-dir", str(data_dir), "--run-dir", str(run_dir),
            "--env-file", str(env_file), "--app-dir", str(a.app_dir), "--no-require-evaluation",
            "--timeout", str(a.timeout), "--extra-run-config", extra]


def docker_stats() -> str:
    import subprocess
    done = subprocess.run(["docker", "stats", "--no-stream", "--format", STATS_FORMAT],
                          capture_output=True, text=True, timeout=60, check=False)
    return done.stdout


def host_info() -> dict:
    info = {"cpus": os.cpu_count()}
    try:
        kb = int(next(line for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemTotal")).split()[1])
        info["mem_total_gb"] = round(kb * 1024 / 10 ** 9, 2)
    except (OSError, StopIteration, ValueError):
        pass
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-shares", type=int, default=CLIENTS, help="neighbourhood size, odd and below 8 for fewer shares")
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--timeout", type=int, default=1500, help="seconds for the flwr run")
    p.add_argument("--round-timeout", type=float, default=120.0, help="SecAgg+ stage timeout in seconds")
    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument("--limit-gb", type=float, default=LIMIT_GB)
    p.add_argument("--app-dir", type=Path, default=APP_DIR)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    try:
        validate_secagg_params(CLIENTS, a.num_shares, a.threshold)
    except ValueError as exc:
        p.error(str(exc))
    out = a.output.resolve()
    data_dir, run_dir, env_file = out / "data", out / "run", out / "run.env"
    argv_runner = runner_argv(a, data_dir, run_dir, env_file)
    if a.dry_run:
        print("compose file:", COMPOSE)
        print("docker stats poll every", a.poll_seconds, "s, project prefix", PROJECT_PREFIX)
        print("python scripts/" + " ".join(argv_runner))
        print("# dry run: nothing was executed")
        return 0
    if shutil.which("docker") is None:
        print("Docker CLI missing. Nothing was measured.", file=sys.stderr)
        return 2
    import run_flower_docker as rfd
    rfd.prepare = stage_clients                      # the shared helper refuses eight clients
    tracker = PeakTracker()
    sampler = StatsSampler(docker_stats, tracker, a.poll_seconds)
    saved_argv, code = sys.argv, 0
    sys.argv = argv_runner
    sampler.start()
    try:
        rfd.main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    finally:
        sys.argv = saved_argv
        sampler.stop()
    record = run_dir / "server" / "flower_run.json"
    verified = bool(record.is_file() and json.loads(record.read_text()).get("native_flower_verified"))
    report = tracker.report(limit_gb=a.limit_gb, run_completed=verified and code == 0,
                            expected_names=expected_container_names(CLIENTS), poll_seconds=a.poll_seconds)
    report.update({"row": "S3", "clients": CLIENTS, "num_shares": a.num_shares, "reconstruction_threshold": a.threshold,
                   "runner_exit_code": code, "poll_errors": sampler.errors[:5], "host": host_info()})
    (out / "scaling_n8_memory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("peak_total_gb", "peak_single_container_gb", "containers_seen",
                                             "outcome", "pass")}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
