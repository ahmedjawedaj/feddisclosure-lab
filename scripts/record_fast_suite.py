"""Run the fast test suite and record its real closing line for `scripts/verify_readme_claims.py`.

Runs `pytest -q -m 'not slow'` with the Python that runs this script and writes a small record: the date,
the git commit the suite ran on, whether the code was dirty, the Python and the versions of the packages that
change the counts (torch, flwr, opacus, hypothesis), the tree that was tested and pytest's own closing summary
line (for example `358 passed, 48 skipped, 4 deselected in 20.30s`). Nothing is typed by hand. If pytest prints
no closing line, or exits non-zero, nothing is written and the exit code is 1.

The counts depend on what is installed, so there is one record per environment:

    python scripts/record_fast_suite.py --kind full                      # flwr and opacus installed, working copy
    /tmp/venv/bin/python scripts/record_fast_suite.py --kind ci --clean-checkout   # torch only, no flwr or opacus

`--clean-checkout` runs pytest in a `git archive` of HEAD, so git-ignored files under `data/` and `*.pt` are
absent, as on the hosted runners. `--kind ci` needs it. Records go to `reports/results/privacy/v0150/` by default.
`verify_readme_claims.py` uses the newest record and flags one older than the code.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata as md
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_readme_claims import CODE_PATHS, RECORD_DIR, SUITE_EVIDENCE, SUITE_FILES  # noqa: E402

PACKAGES = ("torch", "flwr", "opacus", "hypothesis")


def closing_line(output: str) -> str | None:
    """The last line of pytest output that is a closing summary, without surrounding '=' decoration."""
    found = None
    for ln in output.splitlines():
        s = ln.strip().strip("=").strip()
        if SUITE_EVIDENCE.match(s):
            found = s
    return found


def package_versions() -> str:
    parts = []
    for name in PACKAGES:
        try:
            parts.append(f"{name} {md.version(name)}")
        except md.PackageNotFoundError:
            parts.append(f"{name} absent")
    return ", ".join(parts)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def code_dirty() -> bool:
    return bool(git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS))


def record_text(line: str, commit: str, dirty: bool, tree: str, today: str | None = None) -> str:
    return (f"date: {today or datetime.date.today().isoformat()}\ncommit: {commit}\ncode_dirty: {'yes' if dirty else 'no'}\n"
            f"tree: {tree}\npython: {platform.python_version()}\npackages: {package_versions()}\n{line}\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=sorted(SUITE_FILES), default="full",
                   help="which environment this run represents, it picks the default file name")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--clean-checkout", action="store_true", help="run pytest in a git archive of HEAD (no git-ignored files)")
    p.add_argument("--pytest-arg", action="append", default=[], help="extra argument passed to pytest, repeatable")
    a = p.parse_args(argv)
    if a.kind == "ci" and not a.clean_checkout:
        print("--kind ci needs --clean-checkout, the hosted runners have no git-ignored files", file=sys.stderr)
        return 2
    output = a.output or ROOT / RECORD_DIR / SUITE_FILES[a.kind]
    commit = git("rev-parse", "HEAD")
    dirty = False if a.clean_checkout else code_dirty()
    with tempfile.TemporaryDirectory() as tmp:
        work = ROOT
        if a.clean_checkout:
            work = Path(tmp) / "checkout"
            work.mkdir()
            tar = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD"], capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(work)], input=tar, check=True)
        env = {**os.environ, "PYTHONPATH": f"{work / 'src'}:{work}"}
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow", *a.pytest_arg], cwd=work, env=env,
                           capture_output=True, text=True)
    line = closing_line(r.stdout)
    if line is None:
        print("no pytest closing line found, nothing written", file=sys.stderr)
        print(r.stdout[-2000:], file=sys.stderr)
        return 1
    if r.returncode != 0:
        print(f"pytest exited with {r.returncode}, nothing written: {line}", file=sys.stderr)
        return 1
    tree = f"clean checkout of {commit}" if a.clean_checkout else "working copy"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(record_text(line, commit, dirty, tree))
    print(f"wrote {output}: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
