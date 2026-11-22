"""WP5: informed-attacker identification against record-level DP-SGD (one full-batch step).

Replication, in the FL update setting, of Hayes, Mahloujifar and Balle (arXiv 2302.07225).
Stages (all resumable, each seed and sigma is written as soon as it is done):
  run         real Opacus trials, attacks and exact-optimal Monte Carlo
  crosscheck  real Opacus output versus clipped sum plus Gaussian noise (a few hundred trials)
  summarize   tables, intervals and the pre-registered D1 to D5 checks
Evaluation seeds are 0..trials-1. Tuning seeds start at 9000 and are never reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feddisclosure.attacks import dp_informed as dpi  # noqa: E402

DEFAULT_OUT = ROOT / "reports/results/privacy/v0150/dp_informed"
PREREG_OUT = ROOT / "reports/results/privacy/v0150/prereg_check_dp.json"


def stage_run(args) -> None:
    t0 = time.time()
    res = dpi.run_trials(args.out, range(args.trials), dpi.SIGMAS, draws=args.mc_draws,
                         time_budget_s=args.time_budget,
                         progress=lambda s: print(f"seed {s} done, {time.time() - t0:.0f}s", flush=True)
                         if s % 50 == 0 else None)
    print(json.dumps(res), f"elapsed {time.time() - t0:.0f}s", flush=True)


def stage_crosscheck(args) -> None:
    out = {"description": "Real Opacus 1.6.0 output minus hand-computed clipped sum, divided by sigma*clip. "
                          "Standard normal expected (mean 0, variance 1). sigma 0 must equal the clipped sum.",
           "results": [dpi.crosscheck_opacus_noise(sigma=s, trials=args.crosscheck_trials)
                       for s in (0.0, 0.2, 1.0, 4.0)]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crosscheck.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def stage_summarize(args) -> None:
    summary = dpi.summarize(args.out, expected_trials=args.trials)
    summary["trials_per_cell"] = args.trials
    summary["mc_draws_per_trial"] = args.mc_draws
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    check = dpi.prereg_check(summary)
    args.prereg.write_text(json.dumps(check, indent=2) + "\n")
    for k in ("D1", "D2", "D3", "D4", "D5"):
        print(k, "PASS" if check[k]["pass"] else "FAIL")
    if not check["D1"]["pass"]:
        print("D1 breach: treat as a bug in the mechanism or accounting and stop", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "crosscheck", "summarize", "all"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prereg", type=Path, default=PREREG_OUT)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--mc-draws", type=int, default=dpi.MC_DRAWS_PER_TRIAL)
    ap.add_argument("--crosscheck-trials", type=int, default=300)
    ap.add_argument("--time-budget", type=float, default=None, help="stop starting new seeds after this many seconds")
    args = ap.parse_args()
    if args.stage in ("run", "all"):
        stage_run(args)
    if args.stage in ("crosscheck", "all"):
        stage_crosscheck(args)
    if args.stage in ("summarize", "all"):
        stage_summarize(args)


if __name__ == "__main__":
    main()
