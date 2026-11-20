"""Check real Linux-namespace PyTorch evidence against the standard reference.

Important: this validator is NOT a Docker or Flower certification mechanism.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch


def verify(output_dir: Path, data_root: Path) -> dict:
    output_dir = output_dir.resolve()
    data_root = data_root.resolve()
    run = json.loads((output_dir / "linux_namespace_training.json").read_text())
    reference = json.loads((output_dir / "torch_reference.json").read_text())
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (run.get("engine") != "linux_namespace_isolated_torch_NOT_DOCKER_NOT_FLOWER"
            or run.get("native_docker_verified") is not False or run.get("native_flower_verified") is not False):
        raise ValueError("Wrong engine or improper native certification claim")
    if (reference.get("engine") != "torch_independent_process_reference_NOT_Flower"
            or run["manifest_sha256"] != reference["manifest_sha256"]
            or run["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest()):
        raise ValueError("Data provenance is not matched")
    if run["config"] != reference["config"] or len(run["rounds"]) != len(reference["rounds"]):
        raise ValueError("Training configurations or round counts differ")
    all_ids = []
    receipt_hashes = []
    for rnd, r in enumerate(run["rounds"], 1):
        a = reference["rounds"][rnd - 1]["public_test"]
        if (r["round"] != rnd or r["participants"] != 3
                or abs(r["global_accuracy"] - a["accuracy"]) > 1e-6
                or abs(r["global_loss"] - a["loss"]) > 1e-6):
            raise ValueError("Global evaluation does not match reference")
        ids = []
        for i, site in enumerate(manifest["client_partitions"]):
            path = output_dir / f"round-{rnd}-site-{i}-namespace.json"
            receipt = json.loads(path.read_text())
            if (receipt["runtime"] != "linux_user_mount_pid_namespace_NOT_DOCKER_NOT_FLOWER"
                    or receipt["site_id"] != i
                    or receipt["own_train_sha256"] != site["sha256"]
                    or not all(receipt[k] is True for k in ("own_site_readable", "host_proc_hidden",
                        "all_source_partitions_masked", "own_mount_read_only"))):
                raise ValueError("Namespace data isolation proof is incomplete")
            ids.append((receipt["mount_namespace"], receipt["pid_namespace"], receipt["user_namespace"]))
            all_ids.append(ids[-1])
            receipt_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if any(len({t[j] for t in ids}) != 3 for j in range(3)):
            raise ValueError("Independent client namespaces were not observed")
    a = torch.load(output_dir / "namespace_final_model.pt", map_location="cpu", weights_only=True)
    reference_checkpoint = output_dir / "checkpoints" / f"{run['config']['id']}_final.pt"
    b = torch.load(reference_checkpoint, map_location="cpu", weights_only=True)
    if set(a) != set(b):
        raise ValueError("Different model parameter names")
    diffs = {}
    for name in sorted(a):
        if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
            raise ValueError(f"Parameter shape or type mismatch: {name}")
        if not (torch.isfinite(a[name]).all() and torch.isfinite(b[name]).all()):
            raise ValueError("Non-finite parameters")
        diffs[name] = float((a[name] - b[name]).abs().max())
        if not torch.allclose(a[name], b[name], atol=1e-6, rtol=1e-6):
            raise ValueError(f"Model parity failed: {name}")
    result = {"verified_namespace_reference_parity": True,
              "native_docker_verified": False,
              "native_flower_verified": False,
              "three_isolated_clients": True,
              "rounds": len(run["rounds"]), "receipts": len(all_ids),
              "checkpoint_max_abs_difference": max(diffs.values()),
              "per_layer_max_abs_difference": diffs,
              "reference_checkpoint_sha256": hashlib.sha256(reference_checkpoint.read_bytes()).hexdigest(),
              "isolated_checkpoint_sha256": hashlib.sha256((output_dir / "namespace_final_model.pt").read_bytes()).hexdigest(),
              "receipt_sha256": receipt_hashes,
              "limits": "Trusted host, mount/PID/user namespaces, not Docker or Flower. No formal privacy guarantee."}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.output_dir, a.data_dir)
    path = a.output_dir / "namespace_parity_verification.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
