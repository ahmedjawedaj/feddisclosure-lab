"""Record selection and per-client seeds shared by the view study scripts.

`select_view_records` is the data selection that `scripts/run_view_comparison.py` used in v0.14.2,
factored out so the label-floor script draws exactly the same victim records and public pool.
`client_step_seed` keeps the v0.14.2 per-client seed (`seed * 10 + client`) when `legacy` is true
and uses `feddisclosure.evaluation.intervals.client_seed` otherwise.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from feddisclosure.evaluation.intervals import client_seed

SELECTION_SEED_BASE = 1000


class ViewSelection(NamedTuple):
    idx: np.ndarray        # the n_clients * records_per_client records held by the clients
    pool_idx: np.ndarray   # everything else, the public pool
    rng: np.random.Generator   # generator state after the permutation, for later draws


def select_view_records(n_total: int, n_clients: int, records_per_client: int, seed: int) -> ViewSelection:
    """Client records are the first n_clients * records_per_client entries of a seeded permutation.
    The public pool is the rest. Client 0 (the victim) owns the first records_per_client of idx."""
    rng = np.random.default_rng(SELECTION_SEED_BASE + seed)
    perm = rng.permutation(n_total)
    k = n_clients * records_per_client
    if k > n_total:
        raise ValueError("more client records requested than records available")
    return ViewSelection(idx=perm[:k], pool_idx=perm[k:], rng=rng)


def client_step_seed(seed: int, client: int, *, legacy: bool = True) -> int:
    """Seed for one client's local step. legacy=True reproduces the saved v0.14.2 runs."""
    if legacy:
        return int(seed) * 10 + int(client)
    return client_seed(seed, client)
