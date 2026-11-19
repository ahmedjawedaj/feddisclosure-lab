"""Matched equal-client-size FedAvg partition ablation. CPU/PyTorch reference only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import yaml
from feddisclosure.training.datasets import prepare_partitions

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("iid", "dirichlet_balanced_0.1", "dirichlet_balanced_0.5", "dirichlet_balanced_10")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_for_run(run: dict, manifest: dict, data_dir: Path) -> dict:
    counts = [site["samples"] for site in manifest["client_partitions"]]
    val_counts = [site["val_samples"] for site in manifest["client_partitions"]]
    if len(set(counts)) != 1 or len(set(val_counts)) != 1:
        raise ValueError("Client sizes differ; this is not a matched-optimizer-budget study")
    stats = json.loads((data_dir / "evaluator_only/distribution_stats.json").read_text())
    histograms = np.asarray([r["label_histogram"] for r in stats["client_label_histograms"]], dtype=float)
    center = histograms.sum(axis=0) / histograms.sum()
    distributions = histograms / histograms.sum(axis=1, keepdims=True)
    tv = (0.5 * np.abs(distributions - center).sum(axis=1)).tolist()
    validators = [x["evaluation"]["accuracy"] for x in run["final_global_per_client_validation"]]
    return {"test_accuracy": run["rounds"][-1]["public_test"]["accuracy"],
            "worst_client_accuracy": min(validators), "client_accuracy_gap": max(validators) - min(validators),
            "round_accuracies": [r["public_test"]["accuracy"] for r in run["rounds"]],
            "client_validation_accuracies": validators,
            "train_samples_per_client": counts, "val_samples_per_client": val_counts,
            "local_optimizer_batches_per_round_per_client": [run["config"]["local_epochs"] *
                        ((n + run["config"]["batch_size"] - 1) // run["config"]["batch_size"]) for n in counts],
            "mean_client_label_tv": float(np.mean(tv)), "per_client_label_tv": tv,
            "total_communication_bytes": sum(r["communication_model_bytes"] for r in run["rounds"]),
            "manifest_sha256": run["manifest_sha256"],
            "checkpoint_sha256": run["final_checkpoint"]["sha256"]}


def aggregate(rows: list[dict], seeds: list[int], partitions: list[str]) -> dict:
    complete = [seed for seed in seeds if all(any(r["seed"] == seed and r["partition"] == p for r in rows)
                                                for p in partitions)]
    result = {"completed_seeds": complete, "by_partition": {}}
    for p in partitions:
        relevant = [r for r in rows if r["partition"] == p and r["seed"] in complete]
        def stats(key):
            vals = [r["metrics"][key] for r in relevant]
            return {"mean": statistics.mean(vals) if vals else None,
                    "sample_sd": statistics.stdev(vals) if len(vals) > 1 else None,
                    "min": min(vals) if vals else None, "max": max(vals) if vals else None}
        result["by_partition"][p] = {"n": len(relevant), "accuracy": stats("test_accuracy"),
                                       "worst_client_accuracy": stats("worst_client_accuracy"),
                                       "mean_client_label_tv": stats("mean_client_label_tv")}
    if "iid" in partitions:
        index = {(r["seed"], r["partition"]): r for r in rows}
        result["paired_minus_iid_accuracy_pp"] = {
            p: {"by_seed": {str(s): 100 * (index[(s, p)]["metrics"]["test_accuracy"] -
                                            index[(s, "iid")]["metrics"]["test_accuracy"]) for s in complete},
                "mean": statistics.mean([100 * (index[(s, p)]["metrics"]["test_accuracy"] -
                                               index[(s, "iid")]["metrics"]["test_accuracy"]) for s in complete]) if complete else None}
            for p in partitions if p != "iid"
        }
    return result


def run(seeds: list[int], partitions: list[str], out: Path, *, rounds: int, limit: int) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or not partitions or any(p not in PARTITIONS for p in partitions):
        raise ValueError("Invalid seed or partition selection")
    if rounds < 1 or limit < 60 or limit > 1797 or (limit - int(0.2 * limit)) % 3:
        raise ValueError("Require matched divisible training cohort and valid rounds")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cohort_hash: dict[int, str] = {}
    for seed in seeds:
        for p in partitions:
            data_dir = out / "data" / f"seed-{seed}" / p
            if not (data_dir / "manifest.json").exists():
                prepare_partitions(data_dir, dataset="sklearn_digits", clients=3,
                                   partition=p, seed=seed, limit=limit)
            m = json.loads((data_dir / "manifest.json").read_text())
            if any((m[k] != v) for k, v in {"seed": seed, "partition": p, "clients": 3,
                     "source": "sklearn_digits", "source_limit": limit}.items()):
                raise RuntimeError("Data manifest differs from predeclared study")
            with np.load(data_dir / "evaluator_only/test.npz", allow_pickle=False) as d:
                holdout_hash = hashlib.sha256(d["source_idx"].tobytes()).hexdigest()
            if seed in cohort_hash and holdout_hash != cohort_hash[seed]:
                raise RuntimeError("Held-out cohort differs within matched seed")
            cohort_hash[seed] = holdout_hash
            ident = f"balanced-{p}-seed{seed}-r{rounds}"
            config = {"id": ident, "seed": seed, "dataset": "sklearn_digits", "partition": p,
                      "clients": 3, "rounds": rounds, "local_epochs": 2, "batch_size": 32,
                      "learning_rate": 0.15, "source_limit": limit, "algorithm": "fedavg",
                      "proximal_mu": 0.0, "observer": "honest_but_curious_coordinator"}
            config_path = out / "configs" / f"{ident}.yaml"
            config_path.parent.mkdir(exist_ok=True)
            if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
                raise RuntimeError("Refusing changed experiment config on resume")
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            artifact = out / "runs" / f"{ident}.json"
            artifact.parent.mkdir(exist_ok=True)
            if not artifact.exists():
                env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                subprocess.run([sys.executable, str(ROOT / "scripts/run_torch_reference.py"),
                                "--config", str(config_path), "--data-dir", str(data_dir),
                                "--output", str(artifact)], cwd=ROOT, env=env, timeout=300, check=True)
            run_data = json.loads(artifact.read_text())
            if run_data["config"] != config or run_data["manifest_sha256"] != _digest(data_dir / "manifest.json"):
                raise RuntimeError("Stale or mismatched artifact")
            if len(run_data["rounds"]) != rounds:
                raise RuntimeError("Incomplete run")
            checkpoint = artifact.parent / "checkpoints" / f"{ident}_final.pt"
            if not checkpoint.exists() or _digest(checkpoint) != run_data["final_checkpoint"]["sha256"]:
                raise RuntimeError("Missing or modified checkpoint")
            metric = summary_for_run(run_data, m, data_dir)
            rows.append({"seed": seed, "partition": p, "test_cohort_sha256": holdout_hash,
                         "config_sha256": _digest(config_path), "metrics": metric})
            report = {"engine": "torch_independent_process_reference_NOT_Flower",
                      "protocol": "docs/v0.7-pre-registered-protocol.md#b1-equal-capacity-non-iid-study",
                      "seeds_requested": seeds, "partitions_requested": partitions, "rounds": rounds,
                      "source_limit": limit, "results": rows,
                      "aggregate": aggregate(rows, seeds, partitions),
                      "limitations": ["Public 8x8 digits, no native Flower, no DP or secure aggregation",
                                      "Small, short-duration exploratory study; do not claim monotonicity or significance",
                                      "Same record/optimizer budgets, but label assignment changes validation difficulty"]}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"completed seed={seed} partition={p} test={metric['test_accuracy']:.4f} "
                  f"worst={metric['worst_client_accuracy']:.4f} TV={metric['mean_client_label_tv']:.4f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[241, 242, 243])
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS, default=list(PARTITIONS))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--limit", type=int, default=900)
    parser.add_argument("--out", default="reports/results/sprint5/balanced_heterogeneity")
    args = parser.parse_args()
    report = run(args.seeds, args.partitions, ROOT / args.out, rounds=args.rounds, limit=args.limit)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
