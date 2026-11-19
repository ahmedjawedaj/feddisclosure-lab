"""Compare inversion from real local SGD model updates across update-step budgets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from feddisclosure.evaluation.update_observer import create_update_observation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--local-steps", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--attack-iterations", type=int, default=350)
    p.add_argument("--seed", type=int, default=231)
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for steps in args.local_steps:
        case = root / f"steps_{steps}"
        contract = create_update_observation(args.site_file, case, local_steps=steps, seed=args.seed)
        subprocess.run([sys.executable, "-m", "feddisclosure.attacks.gradient_cli",
                        "--observation", str(case / "attacker" / "observation.pt"),
                        "--output-dir", str(case / "attacker"), "--iterations", str(args.attack_iterations),
                        "--seed", str(args.seed + 1)], check=True, capture_output=True, text=True, timeout=200)
        subprocess.run([sys.executable, "-m", "feddisclosure.evaluation.gradient_cli",
                        "--ground-truth", str(case / "evaluator_only" / "original.npz"),
                        "--recovered", str(case / "attacker" / "recovered.npz"),
                        "--output", str(case / "evaluation.json")], check=True, capture_output=True, text=True, timeout=40)
        metrics = json.loads((case / "evaluation.json").read_text())
        item = {"local_steps": steps, "image_mean_mse": metrics["mean_mse"], "mean_psnr_db": metrics["mean_psnr_db"],
                "gradient_proxy_relative_l2_error": contract["gradient_proxy_relative_l2_error_vs_exact_initial_gradient"],
                "attack_iterations": args.attack_iterations, "result": str((case / "evaluation.json").relative_to(root))}
        results.append(item)
        (root / "update_attack_summary.json").write_text(json.dumps({"observer": "server_individual_model_delta",
            "dataset": "approved_public_8x8_digits", "seed": args.seed,
            "sample_budget": "one image repeated for local SGD steps", "runs": results,
            "note": "1-step SGD update corresponds to gradient, multi-step averaged delta is only an approximation",
            "secure_aggregation": False, "differential_privacy": False,
            "limitations": ["One client update visible, not a secure aggregate",
                            "A single image repeated across local steps is not realistic multi-minibatch client training"]}, indent=2) + "\n")
        print(f"local_steps={steps}: image_mse={item['image_mean_mse']:.6g}; proxy_relative_error={item['gradient_proxy_relative_l2_error']:.4g}", flush=True)


if __name__ == "__main__":
    main()
