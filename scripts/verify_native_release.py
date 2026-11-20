"""Fail-closed cross-component native release-evidence checker.

This verifies reported experiment artifacts and consistency. It does NOT
cryptographically attest to their provenance; real hosted CI logs remain
necessary for native execution claims.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


class GateIncomplete(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateIncomplete(message)


def load(path: Path):
    if not path.is_file():
        raise GateIncomplete(f"Missing native evidence: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise GateIncomplete(f"Invalid JSON evidence: {path}") from exc


def verify(root: Path) -> dict:
    artifact_paths = {
        "docker": "reports/results/sprint1/docker_isolation.json",
        "docker_history": "reports/results/sprint1/docker_principal_history.json",
        "flower": "reports/results/native_flower/v012/server/flower_run.json",
        "flower_parity": "reports/results/native_flower/v012/flower_parity.json",
        "flower_topology": "reports/results/native_flower/v012/docker_topology.json",
        "flower_attestations": "reports/results/native_flower/v012/client_attestations.json",
        "opendp": "reports/results/sprint4/opendp_native_smoke.json",
        "opacus": "reports/results/sprint4/opacus_native/summary.json",
        "secagg": "reports/results/sprint4/secagg_native/secagg_run.json",
        "secagg_marker": "reports/results/sprint4/secagg_native/secagg_workflow_returned.json",
    }
    results = {k: load(root / filename) for k, filename in artifact_paths.items()}
    d = results["docker"]
    require(d.get("observed") is True and d.get("separate_data_volumes") is True
            and d.get("all_services_on_one_internal_network") is True
            and d.get("only_gateway_bound_to_host_loopback") is True,
            "Native Docker service isolation not established")
    h = results["docker_history"]
    require(h.get("backend") == "native_docker_compose_with_principal_history"
            and [x.get("status") for x in h.get("cases", [])] == [200, 409, 200, 403],
            "Native protected history sequence not established")
    f = results["flower"]
    p = results["flower_parity"]
    require(f.get("engine") == "Flower ServerApp/ClientApp"
            and f.get("runtime_mode") == "deployment"
            and f.get("native_flower_verified") is True
            and f.get("docker_isolation_verified") is True
            and f.get("client_attestation_verified") is True
            and f.get("exit_code") == 0 and not f.get("timed_out")
            and f.get("final_model_exists") and f.get("summary_exists")
            and p.get("require_deployment") is True
            and p.get("verified_numerical_parity") is True,
            "Three-site Flower Deployment Runtime or checkpoint parity not established")
    require(results["flower_topology"].get("native_verified") is True
            and len(set(results["flower_topology"].get("container_ids", {}).values())) == 8
            and results["flower_attestations"].get("client_attestations_verified") is True
            and len(results["flower_attestations"].get("sites", {})) == 3,
            "Multi-client Docker/participation evidence not verified")
    require(p.get("config_match") is True
            and p.get("flower_sha256") == hashlib.sha256(
                (root / "reports/results/native_flower/v012/server/final_model.pt").read_bytes()).hexdigest(),
            "Native Flower evidence/checkpoint digest inconsistent")
    od = results["opendp"]
    require(od.get("engine") == "native_opendp_three_site_http" and od.get("delta") == 0
            and od.get("local_count_sensitivity") == 1,
            "OpenDP native result does not describe add-remove-one adjacency count")
    require(any(x.get("status") == 200 for x in od.get("released", []))
            and any(x.get("status") == 409 for x in od.get("released", [])),
            "OpenDP run must include releases and budget exhaustion")
    op = results["opacus"]
    account = op.get("dp_accounting", {})
    require(op.get("backend") == "NATIVE Opacus only"
            and account.get("mechanism") == "native_opacus_poisson_dp_sgd"
            and account.get("poisson_sampling") is True
            and account.get("accountant") == "Opacus RDP",
            "Native Opacus accountant missing")
    eps, limit, delta = (account.get(x) for x in ("epsilon_accounted", "target_epsilon", "delta"))
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in (eps, limit, delta))
            and 0 < eps <= limit + 1e-5 and 0 < delta < 1,
            "Opacus native accountant parameters invalid")
    sg, marker = results["secagg"], results["secagg_marker"]
    require(sg.get("engine") == "flower_upstream_secaggplus_workflow"
            and sg.get("native_success") is True and sg.get("exit_code") == 0
            and sg.get("workflow_marker_exists") is True
            and not sg.get("timed_out")
            and marker.get("engine") == "native_flower_upstream_SecAggPlusWorkflow"
            and marker.get("workflow_returned_normally") is True
            and marker.get("clients_requested") >= 3
            and 2 <= marker.get("reconstruction_threshold", 0) <= marker.get("num_shares", 0),
            "Native upstream SecAgg+ workflow evidence incomplete")
    return {"native_gates_consistent": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checks": sorted(artifact_paths),
            "sha256": {k: hashlib.sha256((root / rel).read_bytes()).hexdigest()
                       for k, rel in artifact_paths.items()},
            "scope": "Evidence consistency, not independent runtime attestation or cryptographic validation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/results/native_release_certificate.json")
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (GateIncomplete, FileNotFoundError, TypeError, KeyError, ValueError) as exc:
        print(f"NOT CERTIFIED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
