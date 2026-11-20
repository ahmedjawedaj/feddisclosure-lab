"""Runtime-independent boundary and attestation utilities for Flower deployment.

A receipt documents client activity. It is NOT a cryptographic proof of
isolation, nor is it a DP or secure aggregation guarantee.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_site_identity(*, deployment_site_id: str, node_partition: int,
                          node_partitions: int, expected_clients: int) -> int:
    if not deployment_site_id.isdecimal():
        raise RuntimeError("Deployment site ID must be an unsigned integer")
    identity = int(deployment_site_id)
    if (identity != node_partition or node_partitions != expected_clients
            or not 0 <= identity < expected_clients):
        raise RuntimeError("Mounted dataset identity does not match SuperNode partition")
    return identity


def checked_private_files(site_dir: Path, train_digest: str, val_digest: str) -> tuple[Path, Path]:
    train, val = site_dir / "train.npz", site_dir / "val.npz"
    if not train.is_file() or not val.is_file():
        raise FileNotFoundError("Client can only load its own train.npz and val.npz")
    for path, digest in ((train, train_digest), (val, val_digest)):
        if not digest or len(digest) != 64 or digest_file(path) != digest.lower():
            raise RuntimeError("Client-mounted dataset digest did not match prepared manifest")
    return train, val


def record_client_event(directory: str, *, event_type: str, site_id: int,
                        run_id: int | str, dataset_sha256: str,
                        samples: int, weights_sha256: str | None = None) -> None:
    if event_type not in ("train", "evaluate") or samples <= 0:
        raise ValueError("Invalid client event")
    destination = Path(directory)
    if not destination.is_dir():
        raise RuntimeError("Client attestation directory must be pre-mounted")
    event = {"event_type": event_type, "site_id": site_id, "run_id": str(run_id),
             "dataset_sha256": dataset_sha256, "samples": samples,
             "weights_sha256": weights_sha256,
             "recorded_utc": datetime.now(timezone.utc).isoformat()}
    # A small append-only event file belongs exclusively to this container's
    # attestation mount. No client data, labels or model arrays are stored.
    fd = os.open(destination / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(event, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
