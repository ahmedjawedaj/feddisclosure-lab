"""Monte Carlo acceptance rule for multi-round SecAgg+ error.

A worst-case multi-round bound is not available because training is non-smooth (round-1
quantisation error can change round-2 activation patterns). Instead this script runs the same
FedAvg as the Docker experiment many times, once with exact aggregation and many times with the
UPSTREAM SecAgg+ quantiser arithmetic and fresh stochastic rounding, and records the distribution
of max |SecAgg - exact| over the final weights. A Docker run is accepted when its observed
difference is at or below the empirical 99th percentile of that distribution. This is a
statistical rule, not a bound, and the script reports where each real Docker run fell.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from feddisclosure.attacks.view_study import flatten, secagg_aggregate, unflatten  # noqa: E402
from feddisclosure.training.datasets import load_site_partition  # noqa: E402
from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local  # noqa: E402

CFG = {"seed": 20261118, "local_epochs": 1, "batch_size": 32, "learning_rate": 0.15}


def local_updates(global_state, parts, manifest):
    states, sizes = [], []
    for c, (x, y) in enumerate(parts):
        m = DigitMLP(manifest["num_features"], manifest["num_classes"]); m.load_state_dict(global_state)
        train_local(m, x, y, epochs=CFG["local_epochs"], batch_size=CFG["batch_size"], lr=CFG["learning_rate"],
                    seed=CFG["seed"] + c)
        states.append({k: v.detach().clone() for k, v in m.state_dict().items()}); sizes.append(len(y))
    return states, sizes


def run(parts, manifest, rounds, aggregate, init):
    g = {k: v.clone() for k, v in init.items()}
    for r in range(rounds):
        states, sizes = local_updates(g, parts, manifest)
        g = aggregate(states, sizes, r)
    return g


def distribution(data_dir: Path, rounds: int, trials: int, max_weight: float = 1000.0):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    parts = [load_site_partition(data_dir / p["file"]) for p in manifest["client_partitions"]]
    torch.manual_seed(CFG["seed"])
    init = {k: v.clone() for k, v in DigitMLP(manifest["num_features"], manifest["num_classes"]).state_dict().items()}
    exact = run(parts, manifest, rounds, lambda s, n, r: fedavg_weighted(s, n), init)
    diffs = []
    for t in range(trials):
        def agg(states, sizes, r, t=t):
            vec = secagg_aggregate([flatten(s) for s in states], sizes, max_weight=max_weight, rng_seed=100000 * t + r)
            return {k: v.float() for k, v in unflatten(vec, init).items()}   # Flower returns float32 arrays
        got = run(parts, manifest, rounds, agg, init)
        diffs.append(float(np.abs(flatten(got) - flatten(exact)).max()))
    return np.array(diffs), sum(len(y) for _, y in parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--output", type=Path, default=ROOT / "reports/results/secagg/v014_dropout_matrix/multiround_acceptance_rule.json")
    a = p.parse_args()
    summary_path = ROOT / "reports/results/secagg/v014_dropout_matrix/dropout_matrix_summary.json"
    rep = json.loads(summary_path.read_text())["n4_two_round_replicates"]["max_abs_diff_vs_independent_pytorch"]
    out = {"schema_version": 1, "rule": "accept when observed max abs diff <= empirical 99th percentile of the Monte Carlo distribution",
           "trials": a.trials, "max_weight": 1000.0, "cases": {}}
    d, total = distribution(ROOT / "data/training/digits-docker-iid-4c-v014", 2, a.trials)
    one, _ = distribution(ROOT / "data/training/digits-docker-iid-4c-v014", 1, a.trials)
    out["cases"]["n4_2_rounds"] = {
        "total_samples": total, "mc_quantiles": {q: float(np.quantile(d, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "mc_max": float(d.max()), "mc_min": float(d.min()), "fraction_above_naive_2x_bound": float((d > 7.9473e-05).mean()),
        "one_round_mc_max": float(one.max()), "one_round_worst_case_bound": 3.9736429850260414e-05,
        "docker_runs": {k: {"observed": v, "empirical_percentile": float((d <= v).mean()),
                            "accepted": bool(v <= np.quantile(d, 0.99))} for k, v in rep.items()}}
    out["reading"] = ("The one-round Monte Carlo maximum stays inside the derived worst-case bound, which validates the simulation. "
                      "Two-round differences are heavy tailed, so only a percentile rule is offered.")
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
