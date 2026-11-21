"""Static consistency check of the GitHub Actions workflows. NOT a hosted run.

Parses every workflow, requires the standard keys, and checks that every repository script,
config and test path that a `run:` step names exists. It cannot show that GitHub would run them
(that needs an authorised repository), so hosted CI stays BLOCKED regardless of this result.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH_RE = re.compile(r"(?<![\w./-])((?:scripts|configs|tests|src|experiments|docs)/[\w./-]+\.(?:py|sh|yaml|yml|json|md))")


def check():
    report = {}
    for wf in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        doc = yaml.safe_load(wf.read_text())
        triggers = doc.get(True, doc.get("on"))       # PyYAML parses a bare `on` key as True
        problems = []
        if not triggers:
            problems.append("no trigger")
        if not doc.get("jobs"):
            problems.append("no jobs")
        missing, referenced = set(), set()
        for job in (doc.get("jobs") or {}).values():
            if "runs-on" not in job and "uses" not in job:
                problems.append("job without runs-on")
            for step in job.get("steps", []):
                for m in PATH_RE.findall(step.get("run", "") or ""):
                    referenced.add(m)
                    if not (ROOT / m).exists():
                        missing.add(m)
        report[wf.name] = {"problems": problems, "paths_referenced": len(referenced), "missing_paths": sorted(missing),
                           "ok": not problems and not missing}
    return report


if __name__ == "__main__":
    r = check()
    print(json.dumps(r, indent=1))
    sys.exit(0 if all(v["ok"] for v in r.values()) else 1)
