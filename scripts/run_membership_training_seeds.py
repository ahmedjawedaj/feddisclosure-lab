"""Output-only membership inference on multiple independently trained public-digit models.

Unlike the earlier query-seed experiment, each entry uses a different *training*
seed, corresponding data partitions and final checkpoint. Attack code receives
only probability vectors, never membership labels or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from feddisclosure.evaluation.model_output_observer import create_output_observations

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("max_confidence", "negative_entropy", "top2_margin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[241, 242, 243])
    parser.add_argument("--output-dir", default="reports/results/sprint3/membership_training_seeds")
    parser.add_argument("--per-class", type=int, default=6)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate independently trained model seed")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        # Exact trained FedAvg checkpoint from v0.5 severe non-IID paired study.
        baseline_dir = ROOT / "reports/results/sprint2/ablations/paired"
        baseline = json.loads((baseline_dir / f"fedavg-{seed}.json").read_text())
        assert baseline["config"]["seed"] == seed and baseline["config"]["algorithm"] == "fedavg"
        checkpoint = baseline_dir / "checkpoints" / f"{baseline['config']['id']}_final.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == baseline["final_checkpoint"]["sha256"]
        data_dir = ROOT / "data/training/ablations" / f"non-iid-{seed}"
        manifest_path = data_dir / "manifest.json"
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == baseline["manifest_sha256"]
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == 2
        folder = output / f"model_seed_{seed}"
        # Strict replay protection: no stale score data will be mistaken for this run.
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Refusing overwrite of existing model experiment {folder}")
        contract = create_output_observations(
            checkpoint, [data_dir / x["file"] for x in manifest["client_partitions"]],
            data_dir / manifest["evaluator_test"]["file"], folder,
            per_class=args.per_class, seed=seed + 1000,
        )
        method_results = {}
        for method in METHODS:
            result_dir = folder / "methods" / method
            subprocess.run([sys.executable, "-m", "feddisclosure.attacks.membership_cli",
                            "--observation", str(folder / "attacker" / "model_predictions.npz"),
                            "--method", method, "--output-dir", str(result_dir / "attacker")],
                           check=True, capture_output=True, text=True, timeout=60)
            subprocess.run([sys.executable, "-m", "feddisclosure.evaluation.membership_cli",
                            "--scores", str(result_dir / "attacker" / "scores.npz"),
                            "--truth", str(folder / "evaluator_only" / "membership_labels.npz"),
                            "--output", str(result_dir / "evaluation.json")],
                           check=True, capture_output=True, text=True, timeout=60)
            method_results[method] = json.loads((result_dir / "evaluation.json").read_text())
        run = {"training_seed": seed, "query_seed": seed + 1000,
               "training_algorithm": "fedavg", "rounds": baseline["config"]["rounds"],
               "checkpoint_sha256": baseline["final_checkpoint"]["sha256"],
               "query_count": contract["query_count"], "per_class_requested": args.per_class,
               "methods": method_results}
        runs.append(run)
        summary = {"engine": "output_only_membership_multiple_independent_training_seeds",
                   "training_seed_count": len(runs), "query_seed_policy": "training_seed+1000",
                   "model_observer": "predictions_softmax_only",
                   "attacker_knows_true_labels": False, "attacker_knows_membership": False,
                   "methods_pre_registered": list(METHODS), "runs": runs,
                   "limitations": ["Small public 8x8 digit networks, four training rounds",
                                   "Three different model training seeds, one query subset per model; no model confidence interval claim",
                                   "Class-matched, balanced sampled queries and unrestricted confidence access",
                                   "These are uncalibrated heuristics and not a proof of protection or vulnerability",
                                   "No deployed HTTP inference API or rate limiting verified"]}
        summary["aggregate"] = {}
        for attack_name in METHODS:
            aucs = [row["methods"][attack_name]["roc_auc"] for row in runs]
            summary["aggregate"][attack_name] = {
                "auc_mean": statistics.mean(aucs),
                "auc_sample_sd": statistics.stdev(aucs) if len(aucs) > 1 else None,
                "observed_aucs": aucs,
            }
        (output / "membership_training_seeds_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"seed={seed}: " + ", ".join(f"{name}={method_results[name]['roc_auc']:.4f}" for name in METHODS), flush=True)


if __name__ == "__main__":
    main()
