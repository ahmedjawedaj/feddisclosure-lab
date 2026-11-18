"""Run three genuine isolated HTTP processes without Docker for smoke testing.

No secret shared files between institution processes; each receives its own DB.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(url: str, processes: list[subprocess.Popen], timeout: int = 20) -> None:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("service exited before it became ready")
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.15)
    raise TimeoutError(f"service did not become ready: {url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run and stop the local integration stack")
    parser.add_argument("--output", default="reports/results/analytics_local.json")
    args = parser.parse_args()
    from feddisclosure.evaluation.runner import run_analytics_benchmark

    processes = []
    logs = []
    token = "local-process-test-internal-token"
    research_key = "local-process-test-research-key"
    with tempfile.TemporaryDirectory(prefix="fdl-isolated-") as scratch:
        try:
            specs = [("institution-a", 40), ("institution-b", 35), ("institution-c", 35)]
            ports = [free_port() for _ in range(4)]
            for i, (name, rows) in enumerate(specs):
                env = os.environ.copy()
                env.update({
                    "FDL_INSTITUTION": name, "FDL_DB_PATH": str(Path(scratch) / f"{name}.sqlite"),
                    "FDL_ROWS": str(rows), "FDL_CASES": "1", "FDL_MIN_CELL": "5",
                    "FDL_INTERNAL_TOKEN": token, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
                })
                log = open(Path(scratch) / f"{name}.log", "w")
                logs.append(log)
                processes.append(subprocess.Popen([
                    sys.executable, "-m", "uvicorn", "services.client.app:app",
                    "--host", "127.0.0.1", "--port", str(ports[i]), "--no-access-log",
                ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
                wait_ready(f"http://127.0.0.1:{ports[i]}", processes)
            gateway_env = os.environ.copy()
            gateway_env.update({
                "FDL_CLIENT_URLS": ",".join(f"http://127.0.0.1:{p}" for p in ports[:3]),
                "FDL_INTERNAL_TOKEN": token, "FDL_RESEARCH_KEY": research_key,
                "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
            })
            log = open(Path(scratch) / "gateway.log", "w")
            logs.append(log)
            processes.append(subprocess.Popen([
                sys.executable, "-m", "uvicorn", "services.gateway.app:app",
                "--host", "127.0.0.1", "--port", str(ports[3]), "--no-access-log",
            ], cwd=ROOT, env=gateway_env, stdout=log, stderr=subprocess.STDOUT))
            gateway = f"http://127.0.0.1:{ports[3]}"
            wait_ready(gateway, processes)
            print(f"Running three independent institutional HTTP services; gateway={gateway}", flush=True)
            result = run_analytics_benchmark(
                gateway, research_key, ROOT / "configs/analytics/differencing.yaml", ROOT / args.output,
            )
            summary = {policy: value["metrics"] for policy, value in result["results"].items()}
            print(json.dumps(summary, indent=2), flush=True)
            print(f"Result file: {ROOT / args.output}", flush=True)
            if not args.once:
                print("Press Ctrl+C to stop the services.", flush=True)
                while True:
                    time.sleep(1)
        except Exception:
            for log in logs:
                log.flush()
            for log in Path(scratch).glob("*.log"):
                print(f"===== {log.name} =====\n{log.read_text()}", file=sys.stderr)
            raise
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
