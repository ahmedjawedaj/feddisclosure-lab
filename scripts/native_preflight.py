"""Read-only native runtime availability report. Never marks a skipped gate passed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib import metadata
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command_ok(*cmd: str) -> bool:
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=8, check=False)
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def evaluate() -> dict:
    runtime = {}
    for name, distributions in (
        ("flower", ("flwr",)), ("opendp", ("opendp",)), ("opacus", ("opacus",)),
    ):
        pkg = distributions[0]
        installed = find_spec(pkg) is not None
        try:
            version = metadata.version(pkg) if installed else None
        except metadata.PackageNotFoundError:
            version = None
        runtime[name] = {"installed": installed, "version": version}
    runtime["docker"] = {"installed": shutil.which("docker") is not None,
                         "compose_usable": command_ok("docker", "compose", "version"),
                         "daemon_usable": command_ok("docker", "info")}
    runtime["flower"]["cli_available"] = shutil.which("flwr") is not None
    python_ok = sys.version_info[:2] in ((3, 11), (3, 12), (3, 13))
    local_gates = {
        "python_supported": python_ok,
        "docker_compose_ready": runtime["docker"]["compose_usable"] and runtime["docker"]["daemon_usable"],
        "flower_runtime_ready": runtime["flower"]["installed"] and runtime["flower"]["cli_available"],
        "opendp_runtime_ready": runtime["opendp"]["installed"],
        "opacus_runtime_ready": runtime["opacus"]["installed"],
    }
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "os": platform.system(),
            "runtime": runtime, "local_gates": local_gates,
            "all_local_prerequisites": all(local_gates.values()),
            "disclaimer": "Package installation and Docker readiness are prerequisites, not proof of an executed experiment"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "reports/results/runtime_gates/v011_preflight.json")
    p.add_argument("--strict", action="store_true", help="Exit nonzero if a native prerequisite is missing")
    a = p.parse_args()
    status = evaluate()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if a.strict and not status["all_local_prerequisites"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
