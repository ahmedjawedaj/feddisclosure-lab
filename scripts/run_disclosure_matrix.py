"""Matched public-digit exact-versus-noisy gradient inversion runs.

Noise here is *not* claimed to satisfy differential privacy.  Each pair shares
identical original public image and model initialization for a seed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(pairs):
    conditions = {}
    for noise in sorted({p["noise_std"] for p in pairs}):
        found = [p for p in pairs if p["noise_std"] == noise]
        conditions[str(noise)] = {
            "runs": len(found),
            "mean_mse": statistics.mean(p["mean_mse"] for p in found),
            "std_mse": statistics.stdev(p["mean_mse"] for p in found) if len(found) > 1 else None,
            "mean_psnr_db": statistics.mean(p["mean_psnr_db"] for p in found),
            "mean_ssim": statistics.mean(p["mean_ssim"] for p in found),
        }
    diffs = []
    for exact in (p for p in pairs if p["noise_std"] == 0):
        noisy = next((p for p in pairs if p["seed"] == exact["seed"] and p["noise_std"] > 0), None)
        if noisy:
            diffs.append({"seed": exact["seed"], "noise_std": noisy["noise_std"],
                          "noisy_minus_exact_mse": noisy["mean_mse"] - exact["mean_mse"],
                          "exact_minus_noisy_psnr_db": exact["mean_psnr_db"] - noisy["mean_psnr_db"]})
    return {"observer": "server_individual_exact_gradient", "dataset": "approved_public_sklearn_digits",
            "conditions": conditions, "pairs": pairs, "paired_differences": diffs,
            "privacy_claim": "NONE: manually injected Gaussian gradient noise has no formal DP accountant, sampling or adjacency definition",
            "interpretation": "An empirical attack-performance ablation only; not production FL or secure aggregation"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[231, 232, 233, 234, 235])
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--noise-std", type=float, default=0.01)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for seed in args.seeds:
        for noise in (0.0, args.noise_std):
            dirname = f"seed{seed}_noise{str(noise).replace('.', 'p')}"
            directory = out / dirname
            metrics = directory / "evaluation.json"
            if not metrics.exists():
                subprocess.run([sys.executable, "scripts/run_gradient_inversion.py",
                                "--site-file", args.site_file, "--output-dir", str(directory),
                                "--batch-size", "1", "--seed", str(seed), "--steps", str(args.steps),
                                "--noise-std", str(noise)], check=True, timeout=160, capture_output=True, text=True)
            evaluation = json.loads(metrics.read_text())
            item = {"seed": seed, "noise_std": noise, "mean_mse": evaluation["mean_mse"],
                    "mean_psnr_db": evaluation["mean_psnr_db"], "mean_ssim": evaluation["mean_ssim"],
                    "result": str(metrics.relative_to(out))}
            pairs.append(item)
            (out / "matrix_summary.json").write_text(json.dumps(summarize(pairs), indent=2) + "\n")
            print(f"completed {dirname}: mse={item['mean_mse']:.7f}", flush=True)


if __name__ == "__main__":
    main()
