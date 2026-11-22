"""Interval estimates for rates and a seed convention shared by all v0.15 studies.

Rates live in [0, 1], so a t-interval on a mean of 0/1 values can leave that range. Use the
Wilson interval for reporting and the Clopper-Pearson interval when a hard guarantee on the
coverage matters (for example when comparing an attack success rate with a formal bound).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == trials else min(1.0, centre + half)
    return float(lo), float(hi)


def clopper_pearson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    alpha = 1 - confidence
    lo = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lo, hi


def rate_summary(flags, confidence: float = 0.95) -> dict:
    """Rate with Wilson and Clopper-Pearson intervals from per-trial booleans."""
    arr = np.asarray(flags, dtype=bool).ravel()
    k, n = int(arr.sum()), int(arr.size)
    return {"successes": k, "trials": n, "rate": k / n,
            "wilson": list(wilson_interval(k, n, confidence)),
            "clopper_pearson": list(clopper_pearson_interval(k, n, confidence))}


# Seed convention. Evaluation seeds are 0..N-1 and are never used for tuning. Tuning seeds start
# at TUNING_SEED_BASE so that choices such as attack iterations or learning rates cannot be fit
# on the data that is later reported.
TUNING_SEED_BASE = 9000


def tuning_seed(i: int) -> int:
    return TUNING_SEED_BASE + int(i)


def client_seed(seed: int, client: int, *, max_clients: int = 1024) -> int:
    """Distinct per (seed, client) pair for client < max_clients. The old seed*10+client scheme
    collided from 11 clients upward."""
    if not 0 <= client < max_clients:
        raise ValueError("client index out of range")
    return int(seed) * max_clients + int(client)
