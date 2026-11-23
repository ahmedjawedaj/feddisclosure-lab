"""Seed-level bootstrap intervals for the Fashion-MNIST study, recomputed from the per-seed files.

    python scripts/recompute_fashion_seed_bootstrap.py

Reads `reports/results/privacy/v0150/fashion_mnist/runs/seed*.json` and writes `seed_bootstrap.json`
next to them. Percentile bootstrap over seeds, 10,000 resamples, fixed resampling seed. Nothing here
changes a verdict. It adds intervals that respect the fact that images of one seed are not independent.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feddisclosure.evaluation.intervals import wilson_interval  # noqa: E402
from feddisclosure.evaluation.scaling_study import BOOT_SEED, RECOGNISABLE, bootstrap_ci  # noqa: E402

DIR = ROOT / "reports/results/privacy/v0150/fashion_mnist"
RESAMPLES = 10000
THRESHOLD_POINTS = 0.10


def seed_rates(runs: list[dict], key) -> np.ndarray:
    return np.array([np.mean(np.asarray(key(r)) < RECOGNISABLE) for r in runs])


def main(directory: Path = DIR):
    files = sorted((directory / "runs").glob("seed*.json"), key=lambda f: int(f.stem[4:]))
    runs = [json.loads(f.read_text()) for f in files]
    floor = seed_rates(runs, lambda r: r["label_floor_per_image_mse"])
    rows = {"label_floor": ("label_floor_per_image_mse", None),
            "mean_image": ("mean_image_per_image_mse", None)}
    out = {"seeds": len(runs), "images_per_seed": len(runs[0]["label_floor_per_image_mse"]),
           "resamples": RESAMPLES, "resampling_seed": BOOT_SEED, "unit_resampled": "evaluation seed",
           "rows": {}}
    names = {"label_floor": lambda r: r["label_floor_per_image_mse"], "mean_image": lambda r: r["mean_image_per_image_mse"]}
    for v in runs[0]["views"]:
        names[v] = (lambda v: lambda r: r["views"][v])(v)
    for name, key in names.items():
        rates = seed_rates(runs, key)
        k, n = int(round(rates.sum() * out["images_per_seed"])), len(runs) * out["images_per_seed"]
        row = {"rate": float(rates.mean()), "wilson": list(wilson_interval(k, n)),
               "seed_bootstrap": bootstrap_ci(rates, RESAMPLES)[1:]}
        if name not in ("label_floor", "mean_image"):
            row["gap_to_floor"] = float(rates.mean() - floor.mean())
            row["gap_seed_bootstrap"] = bootstrap_ci(rates - floor, RESAMPLES)[1:]
        out["rows"][name] = row
    col = "secagg_collusion_maxw1000"
    gap = seed_rates(runs, names[col]) - floor
    rng = np.random.default_rng(BOOT_SEED)
    means = gap[rng.integers(0, len(gap), size=(RESAMPLES, len(gap)))].mean(axis=1)
    out["x1_collusion_gap_vs_10_points"] = {
        "gap": float(gap.mean()), "share_of_resamples_with_gap_at_least_10_points": float((means >= THRESHOLD_POINTS - 1e-9).mean()),
        "lower_limit": float(np.quantile(means, 0.025)), "upper_limit": float(np.quantile(means, 0.975))}
    diff = seed_rates(runs, names["secagg_collusion_maxw16"]) - seed_rates(runs, names[col])
    out["paired_collusion_maxw16_minus_maxw1000"] = {"mean": float(diff.mean()),
                                                      "seed_bootstrap": bootstrap_ci(diff, RESAMPLES)[1:]}
    (directory / "seed_bootstrap.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=DIR, help="folder that holds runs/seed*.json")
    main(ap.parse_args().dir)
