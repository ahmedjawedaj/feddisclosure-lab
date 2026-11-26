#!/usr/bin/env python
"""Power table for the v0.16.10 P3b pre-registration (docs/preregistration/v0.16-p3b.md).

Simulates the exact decision rules of the study on seed level values drawn from a normal distribution with an
assumed true mean and an assumed seed level standard deviation, with the same interval method as the study
(95 percent percentile bootstrap of the mean over 20 seeds). No data of the study is read. The assumed standard
deviations are the ones written in the pre-registration. Usage: python3 scripts/p3b_power_table.py [--sims N]
"""
from __future__ import annotations

import argparse
import math

import numpy as np

N_SEEDS = 20
SIMS = 4000
RESAMPLES = 2000
RNG_SEED = 20261126

# (name, kind, margin, assumed seed level sd). kind: lower (lower limit above margin), upper (upper limit at or below
# margin), equiv (both limits inside minus margin and plus margin).
RULES = {
    "K0 control B=1 PSNR gate": ("lower", 2.0, 3.3),
    "G4P plaintext B=4 PSNR gate": ("lower", 1.0, 1.8),
    "G2P plaintext B=2 PSNR gate": ("lower", 1.0, 2.4),
    "G4S plaintext B=4 SSIM gate": ("lower", 0.20, 0.12),
    "G2S plaintext B=2 SSIM gate": ("lower", 0.20, 0.17),
    "A4P aggregate B=4 PSNR at floor": ("upper", 1.0, 1.3),
    "A2P aggregate B=2 PSNR at floor": ("upper", 1.0, 1.6),
    "A4S aggregate B=4 SSIM at floor": ("upper", 0.10, 0.08),
    "A2S aggregate B=2 SSIM at floor": ("upper", 0.10, 0.11),
    "D4P plaintext minus aggregate B=4 PSNR": ("lower", 1.0, 2.2),
    "D2P plaintext minus aggregate B=2 PSNR": ("lower", 1.0, 2.9),
    "D4S plaintext minus aggregate B=4 SSIM": ("lower", 0.10, 0.13),
    "D2S plaintext minus aggregate B=2 SSIM": ("lower", 0.10, 0.18),
    "Q4P plaintext minus collusion B=4 PSNR": ("equiv", 1.0, 0.9),
    "Q2P plaintext minus collusion B=2 PSNR": ("equiv", 1.0, 1.2),
    "Q4S plaintext minus collusion B=4 SSIM": ("equiv", 0.10, 0.06),
    "Q2S plaintext minus collusion B=2 SSIM": ("equiv", 0.10, 0.08),
    "T1 trend B=2 minus B=4 PSNR": ("lower", 0.0, 2.4),
}


def rule_passes(kind: str, margin: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if kind == "lower":
        return lo > margin
    if kind == "upper":
        return hi <= margin
    return (lo >= -margin) & (hi <= margin)


def power(kind: str, margin: float, sd: float, mu: float, sims: int = SIMS, seed: int = RNG_SEED) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N_SEEDS, size=(RESAMPLES, N_SEEDS))
    passed = 0
    for start in range(0, sims, 200):
        k = min(200, sims - start)
        x = rng.normal(mu, sd, size=(k, N_SEEDS))
        means = x[:, idx].mean(axis=2)            # (k, RESAMPLES)
        lo, hi = np.quantile(means, 0.025, axis=1), np.quantile(means, 0.975, axis=1)
        passed += int(rule_passes(kind, margin, lo, hi).sum())
    return passed / sims


def mde(kind: str, margin: float, sd: float, target: float = 0.8, sims: int = SIMS) -> float:
    """True mean at which the rule passes with probability `target` (lower and equiv: search upward from the margin,
    upper: search downward from the margin). For equiv the true mean is the true difference."""
    if kind == "lower":
        a, b = margin, margin + 6 * sd
    elif kind == "upper":
        a, b = margin - 6 * sd, margin
    else:
        a, b = 0.0, margin
    for _ in range(14):
        m = (a + b) / 2
        p = power(kind, margin, sd, m, sims)
        if kind == "lower":
            a, b = (m, b) if p < target else (a, m)
        elif kind == "upper":
            a, b = (a, m) if p < target else (m, b)
        else:       # equiv: power falls as |mu| grows, find mu where it drops to target
            a, b = (a, m) if p < target else (m, b)
    return (a + b) / 2


def analytic_lower(margin: float, sd: float, mu: float) -> float:
    """t approximation, for a cross check of the simulation."""
    se = sd / math.sqrt(N_SEEDS)
    z = (mu - margin) / se - 2.093
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=SIMS)
    args = ap.parse_args()
    print(f"{N_SEEDS} seeds, {RESAMPLES} bootstrap resamples, {args.sims} simulations per cell, rng seed {RNG_SEED}")
    print("rule | kind | margin | assumed sd | power at the true mean given as mean:power | value with 80% power")
    for name, (kind, margin, sd) in RULES.items():
        if kind == "equiv":
            pts = [0.0, 0.25 * margin, 0.5 * margin, 0.75 * margin]
        elif kind == "upper":
            pts = [0.0, margin, margin + 0.5 * sd, margin + sd]
        else:
            pts = [0.0, margin, margin + 0.5 * sd, margin + sd, margin + 1.5 * sd]
        ps = [power(kind, margin, sd, m, args.sims) for m in pts]
        print(f"{name} | {kind} | {margin} | {sd} | " + ", ".join(f"{m:.3g}:{p:.2f}" for m, p in zip(pts, ps)) + f" | {mde(kind, margin, sd, 0.8, args.sims):.3f}")


if __name__ == "__main__":
    main()
