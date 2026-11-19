"""Run independent attacker and evaluator subprocesses over approved public digit files."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from feddisclosure.evaluation.gradient_observer import create_gradient_observation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1, choices=[1, 8, 32])
    p.add_argument("--known-labels", action="store_true")
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--gradient-clip", type=float)
    p.add_argument("--noise-std", type=float, default=0)
    args = p.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = create_gradient_observation(args.site_file, root, batch_size=args.batch_size,
                                           seed=args.seed, known_labels=args.known_labels,
                                           gradient_clip=args.gradient_clip,
                                           gradient_noise_std=args.noise_std)
    subprocess.run([sys.executable, "-m", "feddisclosure.attacks.gradient_cli", "--observation",
                    str(root / "attacker" / "observation.pt"), "--output-dir", str(root / "attacker"),
                    "--iterations", str(args.steps), "--seed", str(args.seed + 1)],
                   check=True, timeout=240)
    subprocess.run([sys.executable, "-m", "feddisclosure.evaluation.gradient_cli", "--ground-truth",
                    str(root / "evaluator_only" / "original.npz"), "--recovered",
                    str(root / "attacker" / "recovered.npz"), "--output", str(root / "evaluation.json")],
                   check=True, timeout=40)
    eval_result = json.loads((root / "evaluation.json").read_text())
    print(json.dumps({"contract": contract, "evaluation": eval_result}, indent=2))


if __name__ == "__main__":
    main()
