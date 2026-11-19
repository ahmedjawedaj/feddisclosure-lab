"""Benchmark individual vs aggregate-only vs all-but-one collusion observations."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from feddisclosure.evaluation.aggregate_observer import CASES, create_aggregate_observations


def run(files, out_dir, *, seed=241, examples=8, iterations=160):
    out = Path(out_dir)
    contract = create_aggregate_observations(files, out, seed=seed, examples_per_site=examples)
    cases = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for case in CASES:
        attack = out / case / "attacker"
        subprocess.run([sys.executable, "-m", "feddisclosure.attacks.gradient_cli",
                        "--observation", str(attack / "observation.pt"),
                        "--output-dir", str(attack), "--iterations", str(iterations),
                        "--seed", str(seed + 1)], check=True, timeout=180, capture_output=True, text=True, env=env)
        evaluation = out / case / "evaluation.json"
        subprocess.run([sys.executable, "-m", "feddisclosure.evaluation.gradient_cli",
                        "--ground-truth", str(out / "evaluator_only" / "target.npz"),
                        "--recovered", str(attack / "recovered.npz"),
                        "--output", str(evaluation)], check=True, timeout=90, capture_output=True, text=True, env=env)
        result = json.loads(evaluation.read_text())
        cases[case] = {"mse": result["mean_mse"], "psnr": result["mean_psnr_db"],
                       "ssim": result["mean_ssim"],
                       "reconstruction_sha256": hashlib.sha256((attack / "recovered.npz").read_bytes()).hexdigest(),
                       "evaluator_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "contract": contract, "attack_iterations": iterations, "results": cases}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Directory holding institution-00..02/train.npz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=241)
    p.add_argument("--examples", type=int, default=8)
    p.add_argument("--iterations", type=int, default=160)
    args = p.parse_args()
    if args.iterations < 1:
        p.error("iterations must be positive")
    files = [Path(args.data_root) / f"institution-{i:02d}" / "train.npz" for i in range(3)]
    result = run(files, args.output_dir, seed=args.seed, examples=args.examples,
                 iterations=args.iterations)
    print(json.dumps({"results": result["results"], "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
