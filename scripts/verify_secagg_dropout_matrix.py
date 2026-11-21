"""Independent verification of the 4-client SecAgg+ runs and the mid-round dropout matrix.

Re-reads saved evidence only. For every scenario it records what the protocol actually did
(rounds aggregated, results per round, upstream error lines, receipts per client) and, when a
model was written, checks it against an independent plain-PyTorch FedAvg that uses the
participation pattern the logs report, within the fixed-point quantisation bound.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_secagg_native import flat, maxdiff, per_round_bound, scenario_flags, simulate  # noqa: E402

CFG = {"seed": 20261118, "local_epochs": 1, "batch_size": 32, "learning_rate": 0.15}


def receipts(run: Path, n: int):
    out = {}
    for i in range(n):
        f = run / "clients" / f"site-{i}" / "events.jsonl"
        out[i] = f.read_text().count('"event_type": "train"') if f.is_file() else 0
    return out


def inspect(run: Path, data_dir: Path, manifest: dict, victims, n: int, t: int):
    flags = scenario_flags(run)
    log = (run / "flower_cli_stdout.log").read_text() if (run / "flower_cli_stdout.log").is_file() else ""
    results = [int(m) for m in re.findall(r"aggregate_fit: received (\d+) results", log)]
    entry = {**flags, "n_clients": n, "reconstruction_threshold": t, "victims_killed_in_round_2": victims,
             "results_per_round": results, "rounds_aggregated": len(results),
             "not_enough_shares_error": "Not enough shares to recover secret" in log,
             "train_receipts_per_client": receipts(run, n),
             "model_written": (run / "server" / "final_model.pt").is_file()}
    expected_survivors = n - len(victims)
    entry["protocol_outcome"] = ("TOLERATED" if entry["model_written"] and results == [n, expected_survivors]
                                 else "HALTED" if entry["not_enough_shares_error"] and not entry["model_written"]
                                 else "OTHER")
    entry["prediction_surviving_holders_ge_threshold"] = expected_survivors >= t
    entry["prediction_matches_outcome"] = (
        (entry["protocol_outcome"] == "TOLERATED") == entry["prediction_surviving_holders_ge_threshold"]
        and entry["protocol_outcome"] != "OTHER")
    if entry["model_written"]:
        sizes = [p["samples"] for p in manifest["client_partitions"]]
        survivors = tuple(i for i in range(n) if i not in victims)
        agg = torch.load(run / "server" / "final_model.pt", weights_only=True)
        ref, _ = simulate(data_dir, manifest, CFG, [tuple(range(n)), survivors])
        tol = (per_round_bound(n, sum(sizes), 1000.0)
               + per_round_bound(len(survivors), sum(sizes[i] for i in survivors), 1000.0))
        entry["vs_independent_pytorch"] = {"participation": [list(range(n)), list(survivors)],
                                           "max_abs_diff": maxdiff(agg, ref), "tolerance": tol,
                                           "within_tolerance": maxdiff(agg, ref) <= tol}
    return entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, default=ROOT / "reports/results/secagg/v014_dropout_matrix")
    p.add_argument("--baseline", type=Path, default=ROOT / "reports/results/secagg/v014_n4/n4-t2-baseline")
    p.add_argument("--output", type=Path, default=ROOT / "reports/results/secagg/v014_dropout_matrix/dropout_matrix_summary.json")
    a = p.parse_args()
    d4 = ROOT / "data/training/digits-docker-iid-4c-v014"
    d3 = ROOT / "data/training/digits-docker-iid-v012"
    m4 = json.loads((d4 / "manifest.json").read_text())
    m3 = json.loads((d3 / "manifest.json").read_text())
    plan = {"n4-t2-kill1-r2": (d4, m4, [3], 4, 2), "n4-t3-kill1-r2": (d4, m4, [3], 4, 3),
            "n4-t2-kill2-r2": (d4, m4, [2, 3], 4, 2), "n4-t3-kill2-r2": (d4, m4, [2, 3], 4, 3),
            "n3-t2-kill1-r2": (d3, m3, [2], 3, 2)}
    report = {"schema_version": 1,
              "rule_under_test": "a mid-round loss is tolerated iff surviving share holders >= reconstruction threshold",
              "upstream_constraint": "reconstruction_threshold must be less than num_shares (n4-t4 run refused by upstream)",
              "earlier_claim_corrected": "The v0.14 'n=3, t=2 halts on dropout' runs stopped the client after round 1, so it was "
                                         "absent when round 2 began and had distributed no key shares. That halts SecAgg+ at any n "
                                         "or threshold. It was not a mid-protocol dropout.",
              "scenarios": {}}
    for name, (dd, mm, victims, n, t) in plan.items():
        run = a.matrix / name
        if (run / "server" / "flower_run.json").is_file():
            report["scenarios"][name] = inspect(run, dd, mm, victims, n, t)
    # n=4 full participation: SecAgg+ vs plaintext control vs independent PyTorch
    ctl = a.matrix / "n4-plaintext-control" / "server" / "final_model.pt"
    on = a.baseline / "server" / "final_model.pt"
    if on.is_file() and ctl.is_file():
        sizes = [p_["samples"] for p_ in m4["client_partitions"]]
        ref, _ = simulate(d4, m4, CFG, [(0, 1, 2, 3)] * 2)
        s_on, s_ctl = torch.load(on, weights_only=True), torch.load(ctl, weights_only=True)
        tol = 2 * per_round_bound(4, sum(sizes), 1000.0)
        report["n4_full_participation_equivalence"] = {
            "secagg_vs_plaintext_control_max_abs_diff": maxdiff(s_on, s_ctl),
            "secagg_vs_independent_pytorch_max_abs_diff": maxdiff(s_on, ref),
            "plaintext_control_vs_independent_pytorch_max_abs_diff": maxdiff(s_ctl, ref),
            "tolerance": tol, "within_tolerance": maxdiff(s_on, ref) <= tol and maxdiff(s_on, s_ctl) <= tol,
            "secagg_run_certified_native": json.loads((a.baseline / "server/flower_run.json").read_text())["native_flower_verified"]}
    # Replicates, 1-round bound check and the error-propagation study (all saved evidence)
    one = a.matrix / "n4-1round-secagg" / "server" / "final_model.pt"
    if one.is_file():
        sizes = [p_["samples"] for p_ in m4["client_partitions"]]
        ref1, _ = simulate(d4, m4, CFG, [(0, 1, 2, 3)])
        b1 = per_round_bound(4, sum(sizes), 1000.0)
        d1 = maxdiff(torch.load(one, weights_only=True), ref1)
        report["n4_one_round_bound_check"] = {"max_abs_diff": d1, "bound": b1, "within_bound": d1 <= b1,
                                              "note": "Single round, so the derived worst-case quantisation bound applies directly."}
    reps = a.matrix.parent / "v014_n4_replicates"
    if reps.is_dir():
        ref2, _ = simulate(d4, m4, CFG, [(0, 1, 2, 3)] * 2)
        sizes = [p_["samples"] for p_ in m4["client_partitions"]]
        b1 = per_round_bound(4, sum(sizes), 1000.0)
        rows = {}
        for r in sorted(reps.glob("rep*/server/final_model.pt")):
            rows[r.parent.parent.name] = maxdiff(torch.load(r, weights_only=True), ref2)
        allruns = {"baseline": report.get("n4_full_participation_equivalence", {}).get("secagg_vs_independent_pytorch_max_abs_diff"), **rows}
        report["n4_two_round_replicates"] = {
            "max_abs_diff_vs_independent_pytorch": allruns, "naive_two_round_tolerance_2x_bound": 2 * b1,
            "runs_above_naive_tolerance": [k for k, v in allruns.items() if v is not None and v > 2 * b1],
            "reading": "The naive 2 x one-round tolerance assumed round-1 error does not grow. It can (see error_propagation.json: a noise study inside the bound exceeds 2 x bound in about 40% of trials): "
                       "training is non-smooth, so a bounded round-1 error can move the round-2 result by more. The strict naive "
                       "check is therefore recorded as FAILED for the baseline run, not relaxed."}
    prop = a.matrix / "error_propagation.json"
    if prop.is_file():
        report["error_propagation_study"] = json.loads(prop.read_text())
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    for k, v in report["scenarios"].items():
        print(k, v["protocol_outcome"], "predicted_tolerated=%s" % v["prediction_surviving_holders_ge_threshold"],
              "results_per_round=%s" % v["results_per_round"], v.get("vs_independent_pytorch", {}).get("within_tolerance"))


if __name__ == "__main__":
    main()
