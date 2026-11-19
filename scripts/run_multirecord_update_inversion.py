"""Compare 1/8/32 distinct-record client updates with optional multiple SGD steps.

All image truth stays in evaluator_only; attacker receives just the observation
contract, initial model, delta, and privileged labels for batch experiments.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from feddisclosure.evaluation.update_observer import create_update_observation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cases", nargs="+", default=["1:1", "8:1", "8:4", "32:1", "32:4"],
                   help="Each case is batch_size:local_steps")
    p.add_argument("--attack-iterations", type=int, default=160)
    p.add_argument("--seed", type=int, default=241)
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.cases:
        batch, steps = (int(v) for v in spec.split(":"))
        case = root / f"batch_{batch}_steps_{steps}"
        contract = create_update_observation(args.site_file, case, local_steps=steps,
                                               batch_size=batch, disclose_labels=batch > 1,
                                               seed=args.seed)
        begin = time.perf_counter()
        subprocess.run([sys.executable, "-m", "feddisclosure.attacks.gradient_cli",
                        "--observation", str(case / "attacker" / "observation.pt"),
                        "--output-dir", str(case / "attacker"),
                        "--iterations", str(args.attack_iterations), "--seed", str(args.seed + 1)],
                       check=True, capture_output=True, text=True, timeout=360)
        subprocess.run([sys.executable, "-m", "feddisclosure.evaluation.gradient_cli",
                        "--ground-truth", str(case / "evaluator_only" / "original.npz"),
                        "--recovered", str(case / "attacker" / "recovered.npz"),
                        "--output", str(case / "evaluation.json")],
                       check=True, capture_output=True, text=True, timeout=60)
        metrics = json.loads((case / "evaluation.json").read_text())
        result = {"batch_size": batch, "local_steps": steps, "known_labels": batch > 1,
                  "image_mean_mse": metrics["mean_mse"], "mean_psnr_db": metrics["mean_psnr_db"],
                  "mean_ssim": metrics["mean_ssim"],
                  "reconstruction_rate_mse_below_0_05": metrics["reconstruction_rate_mse_below_0_05"],
                  "gradient_proxy_relative_l2_error": contract["gradient_proxy_relative_l2_error_vs_exact_initial_gradient"],
                  "elapsed_seconds": round(time.perf_counter() - begin, 3),
                  "evaluator_result": str((case / "evaluation.json").relative_to(root))}
        results.append(result)
        summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "threat_model": "server_individual_model_delta_unprotected",
                   "dataset": "approved_public_8x8_digits", "seed": args.seed,
                   "attack_iterations": args.attack_iterations, "matched_initialization": True,
                   "known_labels_for_batch_gt_1": True, "fixed_local_batch_repeated": True,
                   "secure_aggregation": False, "differential_privacy": False,
                   "runs": results,
                   "limitations": ["One tiny client dataset per case; no changing minibatch schedule",
                                   "All batch labels disclosed to attack for batch sizes 8/32",
                                   "Not a Flower runtime or a DP mechanism; no general privacy implication"]}
        (root / "multirecord_update_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"batch={batch}, steps={steps}: mse={result['image_mean_mse']:.5g}", flush=True)


if __name__ == "__main__":
    main()
