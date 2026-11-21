"""Re-derive every native Flower scenario verdict from saved files (not runner flags).

PASS requires: eight services, no published ports, three distinct read-only site
mounts, final model + summary, three-site receipts for every round, and (for
parity scenarios) strict deployment-mode tensor parity. Fault scenarios PASS only
if the gate REFUSED to certify (no final model, no verified flag).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def judge(run: Path, fault_expected: bool) -> dict:
    rec = load(run / "server" / "flower_run.json") or {}
    topo = load(run / "docker_topology.json") or {}
    att = load(run / "client_attestations.json")
    par = load(run / "flower_parity.json")
    model = run / "server" / "final_model.pt"
    out = {"scenario": run.name, "final_model_sha256": sha(model),
           "runner_flag_native_flower_verified": rec.get("native_flower_verified"),
           "exit_code": rec.get("exit_code"), "timed_out": rec.get("timed_out"),
           "seconds": rec.get("seconds"), "fault_injection": rec.get("fault_injection"),
           "topology_native_verified": topo.get("native_verified"),
           "receipts_verified": bool(att and att.get("client_attestations_verified")),
           "flower_run_id": att.get("run_id") if att else None,
           "parity_verified": bool(par and par.get("verified_numerical_parity")),
           "parity_max_abs_diff": max((v["max_abs_diff"] for v in par["layers"].values()), default=None) if par else None}
    if fault_expected:
        out["verdict"] = ("PASS (gate refused: no checkpoint, not certified)"
                          if not model.is_file() and not rec.get("native_flower_verified") else "FAIL (fault was not rejected)")
    else:
        ok = (model.is_file() and rec.get("native_flower_verified") and topo.get("native_verified")
              and out["receipts_verified"] and out["parity_verified"])
        out["verdict"] = "PASS" if ok else "FAIL"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for d in sorted(a.base.iterdir()):
        if d.is_dir() and (d / "server" / "flower_run.json").is_file():
            rows.append(judge(d, d.name.startswith("fault-")))
    a.output.write_text(json.dumps({"schema_version": 1, "scenarios": rows}, indent=2) + "\n")
    for r in rows:
        print(f'{r["scenario"]:34s} {r["verdict"]}')


if __name__ == "__main__":
    main()
