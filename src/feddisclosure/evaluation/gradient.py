"""Privileged evaluator: only this module accesses original public images."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim


def score_reconstruction(original: np.ndarray, recovered: np.ndarray) -> dict:
    if original.shape != recovered.shape or original.ndim != 2 or original.shape[1] != 64:
        raise ValueError("Expected equal sets of flattened 8x8 public images")
    if not np.isfinite(recovered).all() or not np.isfinite(original).all():
        raise ValueError("Nonfinite reconstruction")
    # Best-match assignment is EVALUATOR PRIVILEGE ONLY. The attacker never sees
    # original pixels or the permutation, and neither is written to its report.
    distances = ((original[:, None, :] - recovered[None, :, :]) ** 2).mean(axis=2)
    oi, ri = linear_sum_assignment(distances)
    mse = distances[oi, ri]
    image_ssim = [float(np.clip(ssim(original[a].reshape(8, 8), recovered[b].reshape(8, 8), data_range=1.0), -1.0, 1.0))
                  for a, b in zip(oi, ri)]
    psnr = [-10 * np.log10(max(float(value), 1e-12)) for value in mse]
    return {"batch_size": int(len(oi)), "mean_mse": float(np.mean(mse)),
            "mean_psnr_db": float(np.mean(psnr)), "mean_ssim": float(np.mean(image_ssim)),
            "per_image_mse": mse.tolist(), "per_image_psnr_db": [float(v) for v in psnr],
            "reconstruction_rate_mse_below_0_05": float(np.mean(mse < 0.05)),
            "assignment": "privileged evaluator optimal MSE matching; not available to attacker",
            "limitations": ["Optimal matching is privileged scoring, not an attacker capability",
                            "PSNR computed on normalized 0..1 public images, not a perceptual identification test"]}


def evaluate_attack(ground_truth: str | Path, recovered: str | Path, output: str | Path) -> dict:
    with np.load(ground_truth, allow_pickle=False) as truth, np.load(recovered, allow_pickle=False) as rec:
        report = score_reconstruction(truth["x"], rec["x"])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
