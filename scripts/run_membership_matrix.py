"""Output-only membership robustness across query samples for one frozen model.

Unlike the training sweep, this does NOT repeat model training. Each run changes
only the balanced public member/nonmember query subset and order.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--per-class", type=int, default=6)
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        folder = root / f"query_seed_{seed}"
        if not (folder / "evaluation.json").exists():
            subprocess.run([sys.executable, "scripts/run_membership_inference.py",
                            "--checkpoint", args.checkpoint, "--data-dir", args.data_dir,
                            "--output-dir", str(folder), "--seed", str(seed),
                            "--per-class", str(args.per_class)], check=True, timeout=75,
                           capture_output=True, text=True)
        score = json.loads((folder / "evaluation.json").read_text())
        runs.append({"query_seed": seed, "roc_auc": score["roc_auc"],
                     "advantage": score["attack_advantage"], "tpr_at_fpr": score["tpr_at_fpr"],
                     "num_members": score["num_members"], "num_nonmembers": score["num_nonmembers"]})
        print(f"query seed={seed}: ROC-AUC={score['roc_auc']:.4f}", flush=True)
    vals = [r["roc_auc"] for r in runs]
    report = {"model": "one frozen FedAvg checkpoint from prior 2-round public digit experiment",
              "variation": "query subsampling and ordering ONLY; not training seed",
              "n": len(runs), "mean_roc_auc": statistics.mean(vals),
              "std_roc_auc": statistics.stdev(vals) if len(vals) > 1 else None,
              "runs": runs, "observer": "external_model_user_confidence_scores_only",
              "privacy_claim": "none", "limitations": ["Small sample, short-training baseline", "Confidence heuristic is uncalibrated", "Results do not generalize to deployed APIs"]}
    (root / "membership_matrix_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
