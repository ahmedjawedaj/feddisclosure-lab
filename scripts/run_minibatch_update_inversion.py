"""Execute an update-inversion study on actual disjoint changing SGD minibatches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from feddisclosure.evaluation.minibatch_observer import create_minibatch_update_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", nargs="+", default=["8:4:0", "8:4:1"],
                        help="minibatch_size:local_steps:schedule_seed; same data seed for all")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=241)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    completed = []
    for item in args.cases:
        minibatch_size, local_steps, schedule_seed = map(int, item.split(":"))
        root = out / f"minibatch_{minibatch_size}_steps_{local_steps}_schedule_{schedule_seed}"
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Refusing to silently overwrite evidence: {root}")
        contract = create_minibatch_update_observation(
            args.site_file, root, batch_size=minibatch_size, local_steps=local_steps,
            seed=args.seed, schedule_seed=schedule_seed, disclose_labels=True,
        )
        start = time.perf_counter()
        attack_cmd = [sys.executable, "-m", "feddisclosure.attacks.gradient_cli",
                      "--observation", str(root / "attacker" / "observation.pt"),
                      "--output-dir", str(root / "attacker"),
                      "--iterations", str(args.iterations), "--seed", str(args.seed + 1)]
        subprocess.run(attack_cmd, check=True, capture_output=True, text=True, timeout=360)
        evaluator_cmd = [sys.executable, "-m", "feddisclosure.evaluation.gradient_cli",
                         "--ground-truth", str(root / "evaluator_only" / "original.npz"),
                         "--recovered", str(root / "attacker" / "recovered.npz"),
                         "--output", str(root / "evaluation.json")]
        subprocess.run(evaluator_cmd, check=True, capture_output=True, text=True, timeout=90)
        metrics = json.loads((root / "evaluation.json").read_text())
        case = {
            "minibatch_size": minibatch_size, "local_steps": local_steps,
            "total_distinct_records": contract["distinct_training_records"],
            "schedule_seed": schedule_seed, "known_all_labels": True,
            "reconstruction_mse": metrics["mean_mse"], "psnr_db": metrics["mean_psnr_db"],
            "ssim": metrics["mean_ssim"],
            "gradient_proxy_relative_l2_error": contract[
                "gradient_proxy_relative_l2_error_vs_exact_initial_full_dataset_gradient"],
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "contract_path": str((root / "observation_contract.json").relative_to(out)),
            "evaluation_path": str((root / "evaluation.json").relative_to(out)),
        }
        completed.append(case)
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "engine": "local_SGD_changing_minibatches_NOT_Flower",
            "observer": "unprotected_honest_but_curious_coordinator_individual_update",
            "data_seed": args.seed, "attack_iterations": args.iterations,
            "attacker_has_all_contributing_labels": True,
            "attacker_has_true_images": False,
            "minibatch_schedule_disclosed": False,
            "cases": completed,
            "limitations": ["One client and public 8x8 digit examples; no native Flower runtime",
                            "Known all labels is a strong additional attacker privilege",
                            "Data seed matched across schedule variants; one SGD epoch across disjoint records",
                            "No DP or secure aggregation and no cryptographic privacy claim",
                            "Attacker/evaluator are separate processes but not OS-level sandboxed"],
        }
        (out / "minibatch_update_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{item}: MSE={case['reconstruction_mse']:.6g}, proxy error={case['gradient_proxy_relative_l2_error']:.6g}", flush=True)


if __name__ == "__main__":
    main()
