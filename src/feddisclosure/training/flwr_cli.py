"""Run the `flwr run` CLI and retry ONLY when it could not reach the local SuperLink.

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

import shutil
import subprocess
import time
from pathlib import Path

UNAVAILABLE = "Connection to the SuperLink is unavailable"
LOCAL_SUPERLINK_STATE = Path.home() / ".flwr" / "local-superlink"


def reset_local_superlink(state_dir: Path = LOCAL_SUPERLINK_STATE) -> None:
    """Stop leftover local SuperLink processes and drop their state (simulation scratch data)."""
    subprocess.run(["pkill", "-f", "flower-superlink"], check=False, capture_output=True)
    time.sleep(1.0)
    shutil.rmtree(state_dir, ignore_errors=True)


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
        unavailable = done.returncode != 0 and UNAVAILABLE in (done.stdout + done.stderr)
        log.append({"attempt": attempt, "exit_code": done.returncode, "superlink_unavailable": unavailable,
                    "output_tail": _tail(done.stderr + "\n" + done.stdout)})
        if not unavailable or attempt == attempts:
            return done.returncode, done.stdout, done.stderr, False, log
        reset()
        time.sleep(wait)
    raise AssertionError("unreachable")
