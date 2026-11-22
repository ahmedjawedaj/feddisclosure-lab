"""Fail-closed native Flower SecAgg+ research smoke; not a cryptographic audit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.util import find_spec
import json
from pathlib import Path
import shutil
import sys
import time

from feddisclosure.training.flwr_cli import run_flwr_cli

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiments/flower_secagg"


def execute(*, data_root: Path, output: Path, timeout: int = 900):
    if find_spec("flwr") is None or shutil.which("flwr") is None:
        raise RuntimeError("Native Flower CLI not installed; SecAgg+ remains unverified")
    if not (data_root / "manifest.json").is_file():
        raise FileNotFoundError("Prepare the public partitions first")
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest.get("schema_version") != 2 or len(manifest["client_partitions"]) != 3:
        raise ValueError("SecAgg+ smoke expects exactly three version-2 site partitions")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("SecAgg+ output directory must start empty")
    config = {
        "data-root": str(data_root.resolve()),
        "artifact-dir": str(output.resolve()),
    }
    cmd = ["flwr", "run", ".", "--stream", "--run-config",
           " ".join(f"{k}={json.dumps(v)}" for k, v in config.items())]
    start = time.perf_counter()
    code, stdout, stderr, timed_out, cli_attempts = run_flwr_cli(cmd, cwd=PROJECT, timeout=timeout)
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    (output / "native_stdout.log").write_text(stdout)
    (output / "native_stderr.log").write_text(stderr)
    marker = output / "secagg_workflow_returned.json"
    evidence = {
        "engine": "flower_upstream_secaggplus_workflow",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "exit_code": code,
        "timed_out": timed_out,
        "cli_attempts": cli_attempts,
        "workflow_marker_exists": marker.exists(),
        "duration_seconds": round(time.perf_counter() - start, 3),
        "native_success": bool(code == 0 and marker.exists() and not timed_out),
        "caveat": "Not a cryptographic audit or an end-to-end deployment certification",
    }
    (output / "secagg_run.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if not evidence["native_success"]:
        tail = "\n".join((stderr + "\n" + stdout).strip().splitlines()[-40:])
        print(f"--- last lines of Flower output ---\n{tail}\n--- end ---", file=sys.stderr)
        raise RuntimeError("Native Flower SecAgg+ execution failed; inspect logs")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/training/digits-iid")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/results/sprint4/secagg_native")
    args = parser.parse_args()
    try:
        print(json.dumps(execute(data_root=args.data_root, output=args.output), indent=2))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"NOT VERIFIED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
