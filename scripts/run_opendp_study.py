"""Native OpenDP study: privacy map, empirical error vs exact counts, real-ledger
retry/exhaustion, and an empirical distinguishing test against the epsilon-DP bound.

Mechanism: opendp.prelude make_laplace on a float-valued typed count (sensitivity 1,
add/remove-one adjacency, pure epsilon-DP per release). Noise comes ONLY from OpenDP.
Empirical numbers below are measurements; they are not the privacy guarantee.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata as md
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.data.synthetic import initialize_database, InstitutionSpec
from feddisclosure.defenses.dp_release import (BudgetExhausted, ReleaseIdConflict,
                                               release_dp_at_site)
from feddisclosure.defenses.opendp_count import OpenDPCount


def utility(eps_grid, trials, true_counts):
    rows = []
    for eps in eps_grid:
        mech = OpenDPCount(eps)
        for true in true_counts:
            errs = [mech.release(true) - true for _ in range(trials)]
            abs_errs = [abs(e) for e in errs]
            mae = statistics.fmean(abs_errs)
            sd = statistics.pstdev(abs_errs)
            half = 1.96 * sd / math.sqrt(trials)
            rows.append({"epsilon": eps, "true_count": true, "trials": trials,
                         "privacy_map_epsilon": mech.verified_map_epsilon,
                         "mae": mae, "mae_ci95": [mae - half, mae + half],
                         "theoretical_laplace_mae": (1 + 1e-9) / eps,
                         "rmse": math.sqrt(statistics.fmean(e * e for e in errs)),
                         "unprotected_abs_error": 0.0,
                         "relative_mae": mae / true if true else None})
    return rows


def distinguishing(eps_grid, trials):
    """Optimal threshold test between neighbouring true counts c and c+1."""
    out = []
    for eps in eps_grid:
        mech = OpenDPCount(eps)
        c = 20
        a = [mech.release(c) for _ in range(trials)]
        b = [mech.release(c + 1) for _ in range(trials)]
        thr = c + 0.5
        tpr = sum(x > thr for x in b) / trials
        fpr = sum(x > thr for x in a) / trials
        adv = tpr - fpr
        bound = (math.exp(eps) - 1) / (math.exp(eps) + 1)
        out.append({"epsilon": eps, "trials_per_world": trials, "empirical_advantage": adv,
                    "dp_total_variation_bound": bound,
                    "advantage_within_bound_plus_3se": adv <= bound + 3 * math.sqrt(2 * 0.25 / trials),
                    "note": "single threshold test; DP bounds ALL tests, empirical advantage is a lower bound"})
    return out


def ledger_checks(eps=0.25, cap=1.0):
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "site.sqlite")
        initialize_database(db, InstitutionSpec("site", 40, 1))
        mech = OpenDPCount(eps)
        kw = dict(principal_scope="principal", epsilon=eps, total_budget=cap, mechanism=mech)
        released = []
        for i in range(4):
            released.append(release_dp_at_site(db, query_id="all_records", request_id=f"req-aaaaaaaaaa{i:02d}", **kw).noisy_count)
        result["four_releases_ok"] = len(released) == 4
        spent = lambda: sqlite3.connect(db).execute("SELECT spent_micro FROM dp_budget").fetchone()[0]
        result["spent_after_four_micro"] = spent()
        try:
            release_dp_at_site(db, query_id="all_records", request_id="req-new-0000001", **kw)
            result["fifth_release_refused"] = False
        except BudgetExhausted:
            result["fifth_release_refused"] = True
        replay = release_dp_at_site(db, query_id="all_records", request_id="req-aaaaaaaaaa00", **kw)
        result["retry_returns_identical_value"] = replay.noisy_count == released[0] and replay.replayed
        result["retry_spent_nothing_extra"] = spent() == 1_000_000
        try:
            release_dp_at_site(db, query_id="negative_records", request_id="req-aaaaaaaaaa00", **kw)
            result["id_reuse_other_query_rejected"] = False
        except ReleaseIdConflict:
            result["id_reuse_other_query_rejected"] = True
        try:
            release_dp_at_site(db, query_id="all_records", request_id="req-aaaaaaaaaa00",
                               principal_scope="principal", epsilon=eps, total_budget=2.0, mechanism=mech)
            result["cap_increase_refused"] = False
        except ValueError:
            result["cap_increase_refused"] = True
        # fresh connection = restart: persisted state still blocks spending
        try:
            release_dp_at_site(db, query_id="negative_records", request_id="req-after-restart1", **kw)
            result["restart_cannot_reset_budget"] = False
        except BudgetExhausted:
            result["restart_cannot_reset_budget"] = True
    result["all_passed"] = all(v for k, v in result.items() if isinstance(v, bool))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=20000)
    a = p.parse_args()
    grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    report = {"schema_version": 1, "datetime_utc": datetime.now(timezone.utc).isoformat(),
              "engine": "native OpenDP make_laplace", "opendp_version": md.version("opendp"),
              "adjacency": "add/remove one record; count sensitivity 1", "delta": 0,
              "privacy_accounting": "pure epsilon, sequential composition via persistent SQLite ledger",
              "utility": utility(grid, a.trials, [3, 40]),
              "distinguishing_test": distinguishing(grid, a.trials),
              "ledger_with_real_mechanism": ledger_checks(),
              "limitations": ["Guarantee covers one allowlisted count per release only",
                              "Empirical attack numbers are not privacy guarantees",
                              "Principal scope is a single shared principal"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["ledger_with_real_mechanism"], indent=2))
    for r in report["utility"]:
        if r["true_count"] == 40:
            print(f'eps={r["epsilon"]:<5} MAE={r["mae"]:.3f} theory={r["theoretical_laplace_mae"]:.3f} map={r["privacy_map_epsilon"]:.6f}')
    for r in report["distinguishing_test"]:
        print(f'eps={r["epsilon"]:<5} adv={r["empirical_advantage"]:.3f} bound={r["dp_total_variation_bound"]:.3f} ok={r["advantage_within_bound_plus_3se"]}')


if __name__ == "__main__":
    main()
