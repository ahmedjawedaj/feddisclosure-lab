"""Real three-site OpenDP-over-HTTP smoke benchmark. NO stand-in noise backend.

Use a clean temporary database, fixed site-wide privacy budget and fresh
OpenDP measurements. If OpenDP is absent, fail before starting the services.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import httpx
from run_local_stack import free_port, wait_ready

ROOT = Path(__file__).resolve().parents[1]


def run(output: Path, *, epsilon: float = .25, total_budget: float = 1.0) -> dict:
    if importlib.util.find_spec("opendp") is None:
        raise RuntimeError("Native OpenDP package required. No simulated DP result will be written.")
    if epsilon <= 0 or total_budget <= 0:
        raise ValueError("Privacy budgets must be positive")
    processes = []
    logs = []
    with tempfile.TemporaryDirectory(prefix="fdl-real-opendp-") as tmp:
        try:
            ports = [free_port() for _ in range(4)]
            for i, (name, n) in enumerate((("institution-a", 40), ("institution-b", 35), ("institution-c", 35))):
                env = dict(os.environ, FDL_INSTITUTION=name, FDL_ROWS=str(n), FDL_CASES="1",
                           FDL_DB_PATH=str(Path(tmp) / f"{name}.sqlite"), FDL_MIN_CELL="5",
                           FDL_INTERNAL_TOKEN="research-local-internal-token", FDL_DP_ENABLED="1",
                           FDL_DP_EPSILON=str(epsilon), FDL_DP_TOTAL_BUDGET=str(total_budget),
                           PYTHONPATH=f"{ROOT / 'src'}:{ROOT}")
                file = open(Path(tmp) / f"{name}.log", "w")
                logs.append(file)
                processes.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "services.client.app:app",
                    "--host", "127.0.0.1", "--port", str(ports[i]), "--no-access-log"], cwd=ROOT,
                    env=env, stdout=file, stderr=subprocess.STDOUT))
                wait_ready(f"http://127.0.0.1:{ports[i]}", processes)
            env = dict(os.environ, FDL_CLIENT_URLS=",".join(f"http://127.0.0.1:{port}" for port in ports[:3]),
                       FDL_INTERNAL_TOKEN="research-local-internal-token", FDL_RESEARCH_KEY="research-local-key",
                       FDL_DP_ENABLED="1", FDL_DP_EPSILON=str(epsilon), PYTHONPATH=f"{ROOT / 'src'}:{ROOT}")
            file = open(Path(tmp) / "gateway.log", "w")
            logs.append(file)
            processes.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "services.gateway.app:app",
                "--host", "127.0.0.1", "--port", str(ports[3]), "--no-access-log"], cwd=ROOT,
                env=env, stdout=file, stderr=subprocess.STDOUT))
            base = f"http://127.0.0.1:{ports[3]}"
            wait_ready(base, processes)
            sequence = ["all_records", "negative_records"] * int(total_budget / epsilon + .5)
            records = []
            start = time.perf_counter()
            with httpx.Client(timeout=10) as client:
                for i, query_id in enumerate(sequence):
                    r = client.post(f"{base}/v1/federated-dp-count", json={
                        "experiment_id": f"query_{i}", "query_id": query_id},
                        headers={"X-Research-Key": "research-local-key"})
                    if r.status_code not in (200, 409):
                        raise RuntimeError(f"Native OpenDP federated gateway failed: HTTP {r.status_code} {r.text}")
                    records.append({"query_id": query_id, "status": r.status_code,
                                    "noisy_count": r.json()["noisy_count"] if r.status_code == 200 else None})
            if not any(r["status"] == 200 for r in records) or not any(r["status"] == 409 for r in records):
                raise RuntimeError("Budget test did not exercise both permitted and denied requests")
            first_two = records[:2]
            inferred = (first_two[0]["noisy_count"] - first_two[1]["noisy_count"]
                        if all(x["status"] == 200 for x in first_two) else None)
            result = {"engine": "native_opendp_three_site_http", "datetime_utc": datetime.now(timezone.utc).isoformat(),
                      "mechanism": "OpenDP make_laplace per-site scalar count", "adjacency": "add_remove_one_record",
                      "local_count_sensitivity": 1, "epsilon_per_site_per_release": epsilon, "delta": 0,
                      "epsilon_budget_per_site": total_budget, "released": records,
                      "true_restricted_subgroup": 3, "noisy_difference_inference": inferred,
                      "difference_abs_error": abs(inferred - 3) if inferred is not None else None,
                      "wall_seconds": time.perf_counter() - start,
                      "limitations": ["Single deterministic synthetic population; not a privacy attack success rate",
                                     "One site-level principal budget; no credential-scoped multitenancy",
                                     "Gateway partial failures may consume some local budget; not two-phase atomic",
                                     "Separate unprotected experiment routes must be disabled and no past releases exist"]}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
            for log in logs:
                log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/results/sprint4/opendp_native_smoke.json")
    parser.add_argument("--epsilon", type=float, default=.25)
    parser.add_argument("--total-budget", type=float, default=1.0)
    args = parser.parse_args()
    try:
        print(json.dumps(run(ROOT / args.output, epsilon=args.epsilon,
                             total_budget=args.total_budget), indent=2))
    except RuntimeError as exc:
        print(f"NOT VERIFIED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
