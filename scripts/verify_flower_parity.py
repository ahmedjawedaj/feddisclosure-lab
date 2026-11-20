"""Validate native Flower checkpoint against matched independent-worker PyTorch FedAvg.

Only a *successfully executed* Flower run and an independently executed reference
qualify. A mock/test fixture can exercise the validator, not certify Flower.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def require(condition: bool, message: str) -> None:
    # Never use assert in certification code: python -O silently drops asserts.
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certify(flower_dir: Path, reference_json: Path, reference_checkpoint: Path,
            manifest: Path, *, atol: float = 1e-5, rtol: float = 1e-5,
            require_deployment: bool = False) -> dict:
    if not (math.isfinite(atol) and atol > 0 and math.isfinite(rtol) and rtol >= 0):
        raise ValueError("Tolerances must be nonnegative and finite (atol positive)")
    record = json.loads((flower_dir / "flower_run.json").read_text())
    summary = json.loads((flower_dir / "flower_summary.json").read_text())
    reference = json.loads(reference_json.read_text())
    require(record.get("engine") == "Flower ServerApp/ClientApp", "No native Flower run metadata")
    require(record.get("exit_code") == 0 and not record.get("timed_out", False), "Flower process did not succeed")
    require(record.get("final_model_exists") and record.get("summary_exists"), "Incomplete Flower run")
    require(summary.get("engine") == "Flower ServerApp/ClientApp", "Invalid Flower summary")
    require(reference.get("engine") == "torch_independent_process_reference_NOT_Flower", "Invalid reference")
    fc, rc = record["config"], reference["config"]
    required = ("seed", "dataset", "partition", "clients", "rounds", "local_epochs",
                "batch_size", "learning_rate", "source_limit")
    mismatches = {k: {"flower": fc.get(k), "reference": rc.get(k)}
                  for k in required if fc.get(k) != rc.get(k)}
    if (float(fc.get("proximal_mu", 0)) != float(rc.get("proximal_mu", 0))
            or rc.get("algorithm", "fedavg") not in ("fedavg", "fedprox")):
        mismatches["proximal_mu"] = {"flower": fc.get("proximal_mu", 0),
                                    "reference": rc.get("proximal_mu", 0)}
    require(not mismatches, f"Not matched training configurations: {mismatches}")
    if require_deployment:
        require(record.get("runtime_mode") == "deployment"
                and record.get("native_flower_verified") is True
                and record.get("docker_isolation_verified") is True
                and record.get("client_attestation_verified") is True,
                "Native Docker Deployment Runtime certification evidence missing")
        site_receipts = json.loads((flower_dir.parent / "client_attestations.json").read_text())
        topology = json.loads((flower_dir.parent / "docker_topology.json").read_text())
        require(site_receipts.get("client_attestations_verified") is True
                and site_receipts.get("rounds") == fc["rounds"]
                and len(site_receipts.get("sites", {})) == fc["clients"]
                and topology.get("native_verified") is True
                and len(set(topology.get("container_ids", {}).values())) == 8,
                "Missing multi-container participation or isolation evidence")
    require(reference["manifest_sha256"] == sha256(manifest), "Reference data provenance mismatch")
    require(reference["final_checkpoint"]["sha256"] == sha256(reference_checkpoint), "Reference checkpoint tampered")
    require(summary["clients_requested"] == fc["clients"] and summary["rounds"] == fc["rounds"], "Flower participants or rounds mismatch")
    require(len(reference["rounds"]) == fc["rounds"], "Reference round count mismatch")
    flower_checkpoint = flower_dir / "final_model.pt"
    flower_weights = torch.load(flower_checkpoint, map_location="cpu", weights_only=True)
    reference_weights = torch.load(reference_checkpoint, map_location="cpu", weights_only=True)
    require(set(flower_weights) == set(reference_weights), "Model parameter keys differ")
    diffs = {}
    all_match = True
    for name in sorted(flower_weights):
        a, b = flower_weights[name].detach().cpu(), reference_weights[name].detach().cpu()
        require(a.shape == b.shape and a.dtype == b.dtype, f"Model tensor mismatch: {name}")
        require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), f"Non-finite model tensor: {name}")
        max_abs = float((a - b).abs().max())
        ok = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
        diffs[name] = {"max_abs_diff": max_abs, "within_tolerance": ok}
        all_match &= ok
    result = {
        "verified_numerical_parity": all_match,
        "scope": "native Flower checkpoint versus same-config independent-worker reference",
        "config_match": True, "require_deployment": require_deployment, "manifest_sha256": sha256(manifest),
        "reference_sha256": sha256(reference_checkpoint),
        "flower_sha256": sha256(flower_checkpoint),
        "atol": atol, "rtol": rtol,
        "layers": diffs,
        "limitations": ["This checks weight equality only and does not establish Docker/container isolation",
                        "A Flower process exit code and recorded logs are necessary but not sufficient for threat-model validation",
                        "Do not claim parity if this report says verified_numerical_parity=false"],
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flower-dir", required=True)
    p.add_argument("--reference-json", required=True)
    p.add_argument("--reference-checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--require-deployment", action="store_true")
    a = p.parse_args()
    report = certify(Path(a.flower_dir), Path(a.reference_json),
                     Path(a.reference_checkpoint), Path(a.manifest), atol=a.atol, rtol=a.rtol,
                     require_deployment=a.require_deployment)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["verified_numerical_parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
