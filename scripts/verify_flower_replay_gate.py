"""Replay parity gate (new). The old 1e-5 float64 weight parity gate is kept and reported beside it.

A native Flower Docker run passes the replay gate only when ALL of these hold:
  1. runner native_flower_verified and docker_topology native_verified are true,
  2. client attestations verified (every client trained every round, weight hashes in receipts),
  3. an independent plain-PyTorch replay using Flower's own float32 reply-order aggregation
     reproduces every per-client per-round trained-weight SHA-256 in the receipts AND the final
     model bit for bit.
This is a stronger statement than 1e-5 closeness to a float64 reference, but it is a different
claim. It does not replace the float64 parity result, which stays in the report.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "reports/results/native_flower/v014"
RUNS = {
    "run-iid-d": ("configs/training/fedavg_digits_iid.yaml", "data/training/digits-docker-iid-v012"),
    "run-iid-repeat": ("configs/training/fedavg_digits_iid.yaml", "data/training/digits-docker-iid-v012"),
    "run-noniid-labelskew": ("configs/training/fedavg_digits_non_iid.yaml", "data/training/digits-docker-noniid-v014"),
    "run-mnist-full-iid": ("configs/training/fedavg_mnist_full_native.yaml", "data/training/mnist-full-docker-iid-v014"),
    "run-mnist-full-iid-seed2": ("configs/training/fedavg_mnist_full_native_seed2.yaml", "data/training/mnist-full-docker-iid-seed2-v014"),
    "run-mnist-full-iid-seed3": ("configs/training/fedavg_mnist_full_native_seed3.yaml", "data/training/mnist-full-docker-iid-seed3-v014"),
    "run-mnist-full-labelskew": ("configs/training/fedavg_mnist_full_labelskew_native.yaml", "data/training/mnist-full-docker-labelskew-v014"),
}


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse an existing bitwise_replay.json")
    ap.add_argument("--output", type=Path, default=V / "replay_gate_summary.json")
    a = ap.parse_args()
    rows = {}
    for name, (cfg, data) in RUNS.items():
        run = V / name
        replay = run / "bitwise_replay.json"
        if not (a.reuse and replay.is_file()):
            subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_flower_bitwise.py"), "--run-dir", str(run),
                            "--config", str(ROOT / cfg), "--data-dir", str(ROOT / data), "--output", str(replay)],
                           check=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        rec = load(run / "server/flower_run.json") or {}
        topo = load(run / "docker_topology.json") or {}
        old = load(run / "flower_parity.json") or {}
        rp = load(replay) or {}
        checks = {"runner_native_verified": rec.get("native_flower_verified") is True,
                  "topology_native_verified": topo.get("native_verified") is True,
                  "client_attestation_verified": rec.get("client_attestation_verified") is True,
                  "client_hashes_all_rounds_reproduced": rp.get("all_client_hashes_match") is True,
                  "final_model_bitwise_reproduced": rp.get("bitwise_reproduction_of_flower_run") is True}
        rows[name] = {"replay_gate_pass": all(checks.values()), "checks": checks,
                      "float64_parity_1e-5_pass": old.get("verified_numerical_parity") is True,
                      "float64_parity_max_layer_abs_diff": max((l["max_abs_diff"] for l in old.get("layers", {}).values()), default=None)}
    summary = {"schema_version": 1, "runs": rows,
               "replay_gate_passes": sum(r["replay_gate_pass"] for r in rows.values()), "runs_total": len(rows),
               "float64_gate_passes": sum(r["float64_parity_1e-5_pass"] for r in rows.values()),
               "note": "The float64 gate is unchanged and still fails where it failed. The replay gate is an additional, different claim."}
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    for n, r in rows.items():
        print("%-28s replay %-5s float64 %-5s maxdiff %s" % (n, r["replay_gate_pass"], r["float64_parity_1e-5_pass"], r["float64_parity_max_layer_abs_diff"]))


if __name__ == "__main__":
    main()
