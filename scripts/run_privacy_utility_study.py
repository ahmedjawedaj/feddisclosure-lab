"""Multi-seed matched study: non-private FedAvg vs native-Opacus DP-FedAvg.

Writes one JSON per (partition, mode, epsilon, seed) plus an aggregate with 95% CIs
(t-interval over seeds) and PAIRED differences DP minus non-private per seed.
Fresh output directory required; nothing is overwritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.training.datasets import prepare_partitions
from feddisclosure.training.privacy_utility import run_federation

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145, 19: 2.093, 29: 2.045}


def ci(values):
    n = len(values)
    m = statistics.fmean(values)
    if n < 2:
        return {"mean": m, "ci95": None, "n": n}
    t = T975.get(n - 1) or T975[min(T975, key=lambda k: abs(k - (n - 1)))]
    half = t * statistics.stdev(values) / math.sqrt(n)
    return {"mean": m, "ci95": [m - half, m + half], "n": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sklearn_digits")
    p.add_argument("--partitions", nargs="+", default=["iid", "label_skew"])
    p.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 4.0, 16.0])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--data-root", type=Path, required=True, help="Where partitions are staged (new dirs)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit("Output directory must be fresh")
    a.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for partition in a.partitions:
        data_dir = a.data_root / f"{a.dataset}-{partition}"
        if not (data_dir / "manifest.json").exists():
            prepare_partitions(data_dir, dataset=a.dataset, clients=3, partition=partition, seed=20261118, limit=None)
        for seed in a.seeds:
            base = dict(rounds=a.rounds, local_epochs=a.local_epochs, batch_size=a.batch_size, lr=a.lr, seed=seed)
            runs.append(run_federation(data_dir, mode="nonprivate", **base))
            print(partition, seed, "nonprivate", round(runs[-1]["final_test_accuracy"], 4), flush=True)
            for eps in a.epsilons:
                runs.append(run_federation(data_dir, mode="opacus_dp", target_epsilon=eps, delta=a.delta,
                                           max_grad_norm=a.max_grad_norm, **base))
                r = runs[-1]
                print(partition, seed, f"eps={eps}", round(r["final_test_accuracy"], 4),
                      "achieved", round(max(r["achieved_epsilon_per_client"]), 3), flush=True)
            (a.output / "runs_partial.json").write_text(json.dumps(runs))
    summary = {"schema_version": 1, "datetime_utc": datetime.now(timezone.utc).isoformat(),
               "args": {k: str(v) for k, v in vars(a).items()}, "groups": []}
    for partition in a.partitions:
        group = [r for r in runs if r["partition"] == partition]
        np_runs = {r["seed"]: r for r in group if r["mode"] == "nonprivate"}
        conditions = [("nonprivate", None)] + [("opacus_dp", e) for e in a.epsilons]
        for mode, eps in conditions:
            rs = [r for r in group if r["mode"] == mode and r["target_epsilon"] == eps]
            row = {"partition": partition, "mode": mode, "target_epsilon": eps,
                   "test_accuracy": ci([r["final_test_accuracy"] for r in rs]),
                   "worst_client_val_accuracy": ci([r["final_worst_client_val_accuracy"] for r in rs]),
                   "membership_auc": ci([r["membership_inference_auc"] for r in rs]),
                   "wall_seconds": ci([r["wall_seconds"] for r in rs]),
                   "bytes_communicated": rs[0]["bytes_communicated"]}
            if mode == "opacus_dp":
                row["achieved_epsilon_max_client"] = ci([max(r["achieved_epsilon_per_client"]) for r in rs])
                row["paired_diff_test_accuracy_vs_nonprivate"] = ci(
                    [r["final_test_accuracy"] - np_runs[r["seed"]]["final_test_accuracy"] for r in rs])
                row["paired_diff_membership_auc_vs_nonprivate"] = ci(
                    [r["membership_inference_auc"] - np_runs[r["seed"]]["membership_inference_auc"] for r in rs])
            summary["groups"].append(row)
    (a.output / "runs.json").write_text(json.dumps(runs, indent=1))
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.output / "runs_partial.json").unlink(missing_ok=True)
    for g in summary["groups"]:
        print(g["partition"], g["mode"], g["target_epsilon"], "acc", round(g["test_accuracy"]["mean"], 4),
              "worst", round(g["worst_client_val_accuracy"]["mean"], 4), "mia", round(g["membership_auc"]["mean"], 4))


if __name__ == "__main__":
    main()
