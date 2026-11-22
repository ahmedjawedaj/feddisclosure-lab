"""v0.15 WP8 fault matrix runner. Resumable. See docs/v0.15-fault-matrix.md.

    PYTHONPATH=src:. python scripts/run_fault_matrix.py dp --workers 2      # about 30 minutes, background it
    PYTHONPATH=src:. python scripts/run_fault_matrix.py sim                 # about 5 minutes
    PYTHONPATH=src:. python scripts/run_fault_matrix.py gates               # about 1 minute
    PYTHONPATH=src:. python scripts/run_fault_matrix.py summarize           # matrix.json and prereg_check_faults.json

Tampered evidence copies live under /tmp (or --work) and never inside saved evidence. Evaluation audits
are 0..199. Nothing is tuned on them.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from feddisclosure.evaluation import fault_matrix as fm  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # half-written last line of an interrupted run is redone
    return out


def _dp_worker(args):
    cond, audits, out_dir, budget = args
    from feddisclosure.attacks import dp_informed as dpi
    x, y = dpi.load_public_digits()
    path = Path(out_dir) / "dp" / f"{cond}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["audit"] for r in read_jsonl(path)}
    t0, new = time.time(), 0
    for a in audits:
        if a in done:
            continue
        if budget and time.time() - t0 > budget:
            break
        rec = fm.run_dp_audit(a, cond, x, y)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        new += 1
    return cond, new


def cmd_dp(a):
    conds = a.conditions or list(fm.DP_CONDITIONS)
    audits = list(range(a.audits))
    jobs = [(c, audits, str(a.out), a.budget_s) for c in conds]
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for cond, new in pool.imap_unordered(_dp_worker, jobs):
                print(cond, "new audits", new, flush=True)
    else:
        for j in jobs:
            print(*_dp_worker(j), flush=True)


def cmd_sim(a):
    from feddisclosure.attacks import dp_informed as dpi
    x, y = dpi.load_public_digits()
    for cond in fm.SIM_CONDITIONS:
        path = a.out / "secagg_sim" / f"{cond}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {r["audit"] for r in read_jsonl(path)}
        t0 = time.time()
        for au in range(a.audits):
            if au in done:
                continue
            with open(path, "a") as fh:
                fh.write(json.dumps(fm.run_sim_trial(au, cond, x, y)) + "\n")
        print(cond, "done", round(time.time() - t0, 1), "s", flush=True)


def cmd_gates(a):
    t0 = time.time()
    res = fm.run_all_gates(a.work)
    res["work_dir"] = str(a.work)
    res["note"] = "Tampered copies were written under the work directory only. Saved evidence digests before and after are equal."
    (a.out / "gates").mkdir(parents=True, exist_ok=True)
    (a.out / "gates" / "gate_results.json").write_text(json.dumps(res, indent=2) + "\n")
    print("saved evidence unchanged:", res["saved_evidence_unchanged"], round(time.time() - t0, 1), "s")


def cmd_summarize(a):
    dp = fm.summarize_dp(a.out, a.audits)
    sim = fm.summarize_sim(a.out, a.audits)
    gates = json.loads((a.out / "gates" / "gate_results.json").read_text())
    matrix = fm.build_matrix(dp, sim, gates)
    fpr = fm.false_positive_rates(dp, sim, gates)
    check = fm.prereg_check(matrix, fpr, dp)
    (a.out / "summary.json").write_text(json.dumps({"schema_version": 1, "dp": dp, "secagg_sim": sim,
                                                    "matrix": matrix, "false_positive_rates": fpr}, indent=2) + "\n")
    (a.out.parent / "prereg_check_faults.json").write_text(json.dumps(check, indent=2) + "\n")
    print("F1", check["F1"]["pass"], check["F1"]["observed"]["strict_detectors_A_B_C_D1"]["n_detected"],
          "F2", check["F2"]["pass"])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("dp", "sim", "gates", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, default=fm.OUT_DIR)
        s.add_argument("--audits", type=int, default=fm.N_AUDITS)
        if name == "dp":
            s.add_argument("--workers", type=int, default=2)
            s.add_argument("--conditions", nargs="*")
            s.add_argument("--budget-s", type=float, default=0.0)
        if name == "gates":
            s.add_argument("--work", type=Path, default=Path("/tmp/fault_matrix_evidence"))
    a = p.parse_args()
    {"dp": cmd_dp, "sim": cmd_sim, "gates": cmd_gates, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
