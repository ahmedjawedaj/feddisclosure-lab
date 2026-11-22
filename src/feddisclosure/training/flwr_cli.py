"""Run the `flwr run` CLI and retry ONLY on known local SuperLink start-up races.

`flwr run` starts a local SuperLink and connects to it straight away. On a slow host
the CLI can lose the connection and exits with "Connection to the SuperLink is
unavailable". The failed attempt may already have registered a run and left a
half-initialised SuperLink database behind (hosted CI showed a second attempt
crashing with "table node already exists"), so before every retry the local
SuperLink process and its state directory are reset. Any other failure, and any
timeout, is final. Every attempt is recorded with the tail of its output so a retry
can never hide a real failure.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

UNAVAILABLE = "Connection to the SuperLink is unavailable"
# Signatures of local SuperLink / simulation START-UP races. Each one happens before any
# application code runs. Anything else (a training error, a failed assertion, a timeout)
# is final and never retried.
STARTUP_RACES = {
    "superlink_unavailable": UNAVAILABLE,
    "sqlite_table_exists": "table node already exists",
    "ray_startup_timeout": "The current node timed out during startup",
}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOCAL_SUPERLINK_STATE = Path.home() / ".flwr" / "local-superlink"


CONNECTION = "local-simulation"
SUPERNODES = 3


def ensure_simulation_connection(config_path: Path | None = None, supernodes: int = SUPERNODES) -> Path:
    """Make sure `~/.flwr/config.toml` has a local-simulation connection with the requested number of supernodes.

    Defining the connection up front means Flower never has to migrate a legacy
    `[tool.flwr.federations]` table out of the app's pyproject.toml while a run starts.
    An existing connection with a different supernode count is an error, never overwritten.
    """
    path = config_path or Path.home() / ".flwr" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    existing = tomllib.loads(text).get("superlink", {}).get(CONNECTION) if text else None
    if existing is None:
        header = "" if text else f'[superlink]\ndefault = "{CONNECTION}"\n'
        path.write_text(text + ("\n" if text and not text.endswith("\n") else "") + header
                        + f"\n[superlink.{CONNECTION}]\noptions.num-supernodes = {supernodes}\n")
    elif existing.get("options", {}).get("num-supernodes") != supernodes:
        raise RuntimeError(f"{path} defines {CONNECTION} with a different supernode count; this run needs {supernodes}")
    return path


def reset_local_superlink(state_dir: Path = LOCAL_SUPERLINK_STATE) -> None:
    """Stop leftover local SuperLink processes and drop their state (simulation scratch data)."""
    subprocess.run(["pkill", "-f", "flower-superlink"], check=False, capture_output=True)
    time.sleep(1.0)
    shutil.rmtree(state_dir, ignore_errors=True)


def startup_race(text: str):
    """Name of the known start-up race in a CLI's output, or None.

    The CLI draws errors in a wrapped, coloured box, so colour codes and box characters
    are removed and whitespace is collapsed before matching.
    """
    plain = " ".join(_ANSI.sub("", text).replace("\u2502", " ").split())
    for name, signature in STARTUP_RACES.items():
        if signature in plain:
            return name
    return None


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])[-3000:]


def run_flwr_cli(command, *, cwd, timeout: float, attempts: int = 6, wait: float = 5.0,
                 reset=reset_local_superlink):
    """Return (exit_code, stdout, stderr, timed_out, attempt_log)."""
    log = []
    deadline = time.monotonic() + timeout
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 124, "", "", True, log
        try:
            done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            out, err = exc.stdout or "", exc.stderr or ""
            out = out.decode(errors="replace") if isinstance(out, bytes) else out
            err = err.decode(errors="replace") if isinstance(err, bytes) else err
            log.append({"attempt": attempt, "exit_code": 124, "superlink_unavailable": False,
                        "output_tail": _tail(err + "\n" + out)})
            return 124, out, err, True, log
        race = startup_race(done.stdout + "\n" + done.stderr) if done.returncode != 0 else None
        log.append({"attempt": attempt, "exit_code": done.returncode,
                    "superlink_unavailable": race == "superlink_unavailable", "startup_race": race,
                    "output_tail": _tail(done.stderr + "\n" + done.stdout)})
        if race is None or attempt == attempts:
            return done.returncode, done.stdout, done.stderr, False, log
        reset()
        time.sleep(wait)
    raise AssertionError("unreachable")
