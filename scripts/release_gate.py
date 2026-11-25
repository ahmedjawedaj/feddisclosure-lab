"""Release gate: the mechanical checks that a reviewer would otherwise repeat by hand.

Usage: python3 scripts/release_gate.py [BASE]    (BASE defaults to origin/main)
       python3 scripts/release_gate.py --staged   (prose check on staged markdown only, used by the pre-commit hook)

It checks the commits from BASE to HEAD and the working tree, and prints one OK or FAIL line per check.
Exit code 1 when any check fails. It does not judge science. A fresh reviewer is still needed for
re-deriving a result from source and for reading the wording against the evidence.

Checks
1. Commit identity and trailers (uses scripts/check_commit_attribution.sh).
2. Pre-registration text above its "## Deviation log" heading is unchanged since the commit that added it.
3. Added prose lines in markdown files have no em dash, no en dash, no semicolon and no banned word.
4. The tree holds no strings from material that is held back (see HELD below).
5. Version in pyproject.toml, CITATION.cff and the top CHANGELOG entry agree.
6. scripts/verify_readme_claims.py passes.
7. Gate integrity: the gate script and the two commit hooks match the sha256 pins in scripts/gate_pins.json.
   A mismatch fails the gate unless GATE_REPIN=1 is set, and then the full diff of the gate files against BASE
   is printed so that a reviewer sees it. `GATE_REPIN=1 python3 scripts/release_gate.py --repin [BASE]` writes new pins.
8. Pre-registration power: a changed pre-registration file with a heading that has the word Predictions must also
   have a heading with the word Power. Only files changed since BASE are checked.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BANNED_WORDS = ("bug", "flaw", "vulnerability", "broken", "insecure")
DASHES = ("—", "–")
# Held back material. Word boundaries keep ordinary words from matching.
HELD = (r"\bpfl\b", r"JointMechanism", r"\bT3b\b", r"\bT4b\b", r"fedml\.init\b")
HELD_SKIP = ("scripts/release_gate.py",)
PINS_FILE = "scripts/gate_pins.json"
PINNED = ("scripts/release_gate.py", "scripts/hooks/commit-msg", "scripts/hooks/pre-commit")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def check_attribution(base: str) -> tuple[bool, str]:
    r = subprocess.run(["bash", "scripts/check_commit_attribution.sh", base, "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "no output"


def check_prereg(base: str) -> tuple[bool, str]:
    bad = []
    changed = set(git("diff", "--name-only", f"{base}...HEAD", "--", "docs/preregistration").split())
    files = [f for f in git("ls-files", "docs/preregistration").split() if f.endswith(".md") and f in changed]
    for f in files:
        text = (ROOT / f).read_text()
        marker = "## Deviation log"
        if marker not in text:
            continue
        added = git("log", "--diff-filter=A", "--format=%H", "--", f).split()
        if not added:
            continue
        first = added[-1]
        original = git("show", f"{first}:{f}")
        if marker not in original:
            continue
        if text.split(marker)[0] != original.split(marker)[0]:
            bad.append(f)
    return not bad, "text above the Deviation log changed in " + ", ".join(bad) if bad else f"{len(files)} changed pre-registration files checked"


def file_sha256(rel: str, root: Path = ROOT) -> str:
    p = root / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"


def pin_mismatches(pins: dict, root: Path = ROOT, pinned: tuple[str, ...] = PINNED) -> list[str]:
    """Names of pinned files whose sha256 differs from the pin, or that have no pin."""
    bad = []
    for rel in pinned:
        if rel not in pins:
            bad.append(f"{rel} (no pin)")
        elif file_sha256(rel, root) != pins[rel]:
            bad.append(rel)
    return bad


def gate_diff(base: str) -> str:
    return git("diff", base, "--", *PINNED, PINS_FILE)


def check_integrity(base: str) -> tuple[bool, str]:
    pins_path = ROOT / PINS_FILE
    repin = os.environ.get("GATE_REPIN") == "1"
    try:
        pins = json.loads(pins_path.read_text())
    except (OSError, ValueError):
        pins = {}
    bad = pin_mismatches(pins)
    if not bad:
        return True, f"{len(PINNED)} gate files match their pins"
    if repin:
        print("GATE_REPIN=1 is set. Diff of the gate files and pins against " + base + ":")
        print(gate_diff(base) or "(no difference against the base)")
        return True, "GATE_REPIN=1, " + ", ".join(bad) + " differ from the pins, review the diff printed above"
    return False, ", ".join(bad) + f" differ from {PINS_FILE}. Review the change and rerun with GATE_REPIN=1 to see the diff"


def repin(base: str) -> int:
    if os.environ.get("GATE_REPIN") != "1":
        print("FAIL  repin needs GATE_REPIN=1")
        return 2
    old = {}
    try:
        old = json.loads((ROOT / PINS_FILE).read_text())
    except (OSError, ValueError):
        pass
    new = {rel: file_sha256(rel) for rel in PINNED}
    (ROOT / PINS_FILE).write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    for rel in PINNED:
        print(f"{'same   ' if old.get(rel) == new[rel] else 'changed'}  {rel}  {old.get(rel, 'none')} -> {new[rel]}")
    print(gate_diff(base) or "(no difference against the base)")
    return 0


def prereg_power_missing(text: str) -> bool:
    """True when a heading has the word Predictions and no heading has the word Power."""
    heads = [m.group(0) for m in re.finditer(r"^#{1,6}[ \t].*$", text, re.M)]
    has_pred = any(re.search(r"\bpredictions?\b", h, re.I) for h in heads)
    has_power = any(re.search(r"\bpower\b", h, re.I) for h in heads)
    return has_pred and not has_power


def check_prereg_power(base: str) -> tuple[bool, str]:
    changed = set(git("diff", "--name-only", base, "--", "docs/preregistration").split())
    files = [f for f in sorted(changed) if f.endswith(".md") and (ROOT / f).is_file()]
    bad = [f for f in files if prereg_power_missing((ROOT / f).read_text())]
    return not bad, ("no Power heading in " + ", ".join(bad)) if bad else f"{len(files)} changed pre-registration files checked"


def added_markdown_lines(base: str):
    out = git("diff", "--cached", "--unified=0", "--", "*.md") if base == "--staged" else git("diff", "--unified=0", f"{base}...HEAD", "--", "*.md")
    path, fence = "", False
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            path, fence = line[6:], False
        elif line.startswith("+") and not line.startswith("+++"):
            body = line[1:]
            if body.lstrip().startswith("```"):
                fence = not fence
                continue
            if not fence:
                yield path, body


def prose_violations(path: str, body: str) -> list[str]:
    bad = []
    low = body.lower()
    if any(d in body for d in DASHES):
        bad.append(f"{path}: dash")
    if ";" in body:
        bad.append(f"{path}: semicolon")
    for w in BANNED_WORDS:
        if re.search(rf"\b{w}\b", low):
            bad.append(f"{path}: '{w}'")
    return bad


def check_prose(base: str) -> tuple[bool, str]:
    bad = []
    for path, body in added_markdown_lines(base):
        bad += prose_violations(path, body)
    return not bad, "; ".join(sorted(set(bad))[:8]) if bad else "no dashes, semicolons or banned words in added prose"


def check_held() -> tuple[bool, str]:
    hits = []
    files = [f for f in git("ls-files").splitlines() if f not in HELD_SKIP]
    for f in files:
        p = ROOT / f
        if p.suffix in (".png", ".json", ".pt", ".npz", ".zip") or not p.is_file():
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        for pat in HELD:
            if re.search(pat, text):
                hits.append(f"{f}: {pat}")
    return not hits, "; ".join(hits[:6]) if hits else "no held back strings in the tree"


def check_versions() -> tuple[bool, str]:
    py = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    cff = re.search(r"^version: (\S+)", (ROOT / "CITATION.cff").read_text(), re.M).group(1)
    top = re.search(r"^## \[([^\]]+)\]", (ROOT / "CHANGELOG.md").read_text(), re.M).group(1)
    readme = re.search(r"Current release: (\d+\.\d+\.\d+)", (ROOT / "README.md").read_text())
    values = {"pyproject": py, "CITATION": cff, "CHANGELOG": top, "README": readme.group(1) if readme else "missing"}
    return len(set(values.values())) == 1, ", ".join(f"{k} {v}" for k, v in values.items())


def check_claims() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "scripts/verify_readme_claims.py"], cwd=ROOT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["no output"])[-1]
    return r.returncode == 0, last


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    if base == "--staged":
        ok, detail = check_prose(base)
        print(f"{'OK  ' if ok else 'FAIL'}  prose: {detail}")
        return 0 if ok else 1
    if base == "--repin":
        return repin(sys.argv[2] if len(sys.argv) > 2 else "origin/main")
    checks = [("gate_integrity", lambda: check_integrity(base)), ("attribution", lambda: check_attribution(base)),
              ("prereg_integrity", lambda: check_prereg(base)), ("prereg_power", lambda: check_prereg_power(base)),
              ("prose", lambda: check_prose(base)), ("held_back", check_held), ("versions", check_versions),
              ("claims", check_claims)]
    failed = 0
    for name, fn in checks:
        ok, detail = fn()
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {name}: {detail}")
    print(f"{len(checks) - failed} of {len(checks)} gate checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
