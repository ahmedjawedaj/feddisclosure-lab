"""Run the `flwr run` CLI and retry ONLY when it could not reach the local SuperLink.

`flwr run` starts a local SuperLink and connects to it straight away. On a slow host
the CLI can connect before the SuperLink is listening and exits with "Connection to
the SuperLink is unavailable". Nothing has run at that point, so trying again is safe.
Any other failure, and any timeout, is final. Every attempt is recorded so a retry
can never hide a real failure.
"""
from __future__ import annotations

import subprocess
import time

UNAVAILABLE = "Connection to the SuperLink is unavailable"


def run_flwr_cli(command, *, cwd, timeout: float, attempts: int = 6, wait: float = 5.0):
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
            log.append({"attempt": attempt, "exit_code": 124, "superlink_unavailable": False})
            return 124, out, err, True, log
        unavailable = done.returncode != 0 and UNAVAILABLE in (done.stdout + done.stderr)
        log.append({"attempt": attempt, "exit_code": done.returncode, "superlink_unavailable": unavailable})
        if not unavailable or attempt == attempts:
            return done.returncode, done.stdout, done.stderr, False, log
        time.sleep(wait)
    raise AssertionError("unreachable")
