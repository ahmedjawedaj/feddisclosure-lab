"""Namespace-only client launcher: executed AFTER unshare, never Docker/Flower.

Mounts exactly one site dataset read-only, then masks the source data root.
This is a process-level research fallback for environments without Docker.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def call(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=20)


def validate_mapping(root: Path, source: Path, target: Path, site_id: int) -> None:
    if not 0 <= site_id < 3:
        raise ValueError("Only three-site experiments are supported")
    if source != root / f"institution-{site_id:02d}" or root == target or root in target.parents:
        raise ValueError("Unsafe site mapping")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--site-source", type=Path, required=True)
    p.add_argument("--site-target", type=Path, required=True)
    p.add_argument("--site-id", type=int, required=True)
    p.add_argument("--proof", type=Path, required=True)
    p.add_argument("worker_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = a.data_root.resolve(strict=True)
    source = a.site_source.resolve(strict=True)
    target = a.site_target.resolve(strict=True)
    validate_mapping(root, source, target, a.site_id)
    if not a.worker_args or a.worker_args[0] != "--":
        raise SystemExit("Expected -- followed by worker arguments")
    worker_args = a.worker_args[1:]
    # These operations occur in a NEW user/mount/PID namespace. Do not call as host root.
    call("mount", "--make-rprivate", "/")
    call("mount", "--bind", str(source), str(target))
    call("mount", "-o", "remount,bind,ro", str(target))
    call("mount", "-t", "tmpfs", "tmpfs", str(root))
    if any((root / f"institution-{i:02d}" / "train.npz").exists() for i in range(3)):
        raise RuntimeError("Dataset root was not hidden from the isolated worker")
    expected = (target / "train.npz", target / "val.npz")
    if not all(x.is_file() for x in expected):
        raise RuntimeError("Own private train/val files not mounted")
    mounts = Path("/proc/self/mountinfo").read_text()
    matching = [line for line in mounts.splitlines() if line.split(" - ")[0].split()[4] == str(target)]
    if not matching or not any("ro" in line.split(" - ")[0].split()[5].split(",") for line in matching):
        raise RuntimeError("Client dataset mount is not read-only")
    namespace_ids = {"pid_namespace": os.readlink("/proc/self/ns/pid"),
                     "mount_namespace": os.readlink("/proc/self/ns/mnt"),
                     "user_namespace": os.readlink("/proc/self/ns/user")}
    # Existing /proc belongs to the parent PID namespace. Linux disallows
    # mounting a less-revealing procfs here, so hide it entirely instead.
    call("mount", "-t", "tmpfs", "tmpfs", "/proc")
    if Path("/proc/1").exists():
        raise RuntimeError("Host process filesystem remains exposed")
    proof = {
        "schema_version": 1,
        "runtime": "linux_user_mount_pid_namespace_NOT_DOCKER_NOT_FLOWER",
        "site_id": a.site_id,
        **namespace_ids,
        "host_proc_hidden": True,
        "own_site_readable": True,
        "all_source_partitions_masked": True,
        "own_mount_read_only": True,
        "own_train_sha256": hashlib.sha256(expected[0].read_bytes()).hexdigest(),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    a.proof.write_text(json.dumps(proof, indent=2) + "\n")
    os.execv(sys.executable, [sys.executable, "-m", "feddisclosure.training.worker", *worker_args])


if __name__ == "__main__":
    main()
