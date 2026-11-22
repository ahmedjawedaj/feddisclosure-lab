"""Run the fast test suite and record its real closing line for `scripts/verify_readme_claims.py`.

Runs `pytest -q -m 'not slow'`, keeps only the date and pytest's own closing summary line (for example
`195 passed, 4 skipped, 4 deselected, 2 warnings in 20.30s`) and writes them to
`reports/results/privacy/v0143/fast_suite.txt`. Nothing is typed by hand. If pytest prints no closing
line, or exits non-zero, the file is not written and the exit code is 1.
"""
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_readme_claims import SUITE, SUITE_EVIDENCE  # noqa: E402


def closing_line(output: str) -> str | None:
    """The last line of pytest output that is a closing summary, without surrounding '=' decoration."""
    found = None
    for ln in output.splitlines():
        s = ln.strip().strip("=").strip()
        if SUITE_EVIDENCE.match(s):
            found = s
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=ROOT / SUITE)
    p.add_argument("--pytest-arg", action="append", default=[], help="extra argument passed to pytest, repeatable")
    a = p.parse_args(argv)
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow", *a.pytest_arg], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    line = closing_line(r.stdout)
    if line is None:
        print("no pytest closing line found, nothing written", file=sys.stderr)
        print(r.stdout[-2000:], file=sys.stderr)
        return 1
    if r.returncode != 0:
        print(f"pytest exited with {r.returncode}, nothing written: {line}", file=sys.stderr)
        return 1
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(f"date: {datetime.date.today().isoformat()}\n{line}\n")
    print(f"wrote {a.output}: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
