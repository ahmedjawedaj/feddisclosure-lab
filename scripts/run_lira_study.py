"""Shadow-model likelihood-ratio membership audit of nonprivate vs Opacus DP federated models.

Per regime and mode: train K shadow and T target federations, each on a random half of a fixed
public pool, split IID over three sites. The attacker uses the shadows (trained with the SAME
mode, hyperparameters and epsilon) to build per-example IN/OUT score distributions, then scores
each target model. Reports the weak loss-threshold attack and the stronger LiRA attack side by
side, with AUC and TPR at fixed FPR, over the T targets. Empirical only, not a guarantee.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from feddisclosure.training.datasets import _source
from feddisclosure.training.lira_study import (lira_scores, lira_scores_global_var, logit_confidence,
                                               roc_summary, train_federation)
from feddisclosure.training.task import evaluate_local

ROOT = Path(__file__).resolve().parents[1]
REGIMES = {
    "overfit": dict(pool=360, rounds=40, local_epochs=10, batch_size=32, lr=0.15),
    "moderate": dict(pool=1200, rounds=20, local_epochs=5, batch_size=32, lr=0.15),
}


def _pool(n, seed):
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    idx = np.random.default_rng(seed).permutation(len(y))[:n]
    return x[idx], y[idx]


def _one(args):
    regime, mode, eps, model_id, seed = args
    cfg = REGIMES[regime]
    x, y = _pool(cfg["pool"], seed)
    rng = np.random.default_rng(10_000 * seed + 7919 * model_id + (0 if mode == "nonprivate" else 1))
    member = np.zeros(len(y), dtype=bool)
    member[rng.permutation(len(y))[: len(y) // 2]] = True
    idx = np.flatnonzero(member)
    sites = [(x[p], y[p]) for p in np.array_split(rng.permutation(idx), 3)]
    start = time.perf_counter()
    model, eps_list = train_federation(
        sites, features=64, classes=10, mode=mode, rounds=cfg["rounds"], local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"], lr=cfg["lr"], seed=seed * 1000 + model_id, target_epsilon=eps)
    phi = logit_confidence(model, x, y)
    return {"regime": regime, "mode": mode, "target_epsilon": eps, "model_id": model_id,
            "member": member.tolist(), "phi": phi.tolist(),
            "train_accuracy": evaluate_local(model, x[member], y[member])["accuracy"],
            "heldout_accuracy": evaluate_local(model, x[~member], y[~member])["accuracy"],
            "achieved_epsilon_max": max(eps_list) if eps_list else None,
            "seconds": time.perf_counter() - start}


def _ci(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return [float(v.mean()), None, None]
    from scipy import stats
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(v.mean() - half), float(v.mean() + half)]


def attack(models, shadows: int):
    """models: list of per-model dicts. First `shadows` are shadows, rest are targets."""
    phi = np.array([m["phi"] for m in models])
    mem = np.array([m["member"] for m in models])
    sh_phi, sh_mem = phi[:shadows], mem[:shadows]
    results = []
    for t in range(shadows, len(models)):
        ins = [sh_phi[sh_mem[:, i], i] for i in range(phi.shape[1])]
        outs = [sh_phi[~sh_mem[:, i], i] for i in range(phi.shape[1])]
        lira = roc_summary(lira_scores(phi[t], ins, outs), mem[t].astype(int))
        lira_gv = roc_summary(lira_scores_global_var(phi[t], ins, outs), mem[t].astype(int))
        weak = roc_summary(phi[t], mem[t].astype(int))
        results.append({"target": models[t]["model_id"], "lira": lira, "lira_global_var": lira_gv,
                        "loss_threshold": weak})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=list(REGIMES), required=True)
    p.add_argument("--epsilons", nargs="+", type=float, default=[8.0, 2.0])
    p.add_argument("--shadows", type=int, default=24)
    p.add_argument("--targets", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    modes = [("nonprivate", None)] + [("opacus_dp", e) for e in a.epsilons]
    jobs = [(a.regime, m, e, i, a.seed) for (m, e) in modes for i in range(a.shadows + a.targets)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        done = list(pool.map(_one, jobs))
    summary = {"schema_version": 1, "regime": a.regime, "config": REGIMES[a.regime], "delta": 1e-5,
               "shadows": a.shadows, "targets": a.targets, "seed": a.seed,
               "attacker": "knows pipeline, hyperparameters, mechanism and epsilon, trains matched shadows, "
                           "sees only the final global model's output on candidate records",
               "modes": []}
    for (m, e) in modes:
        group = [d for d in done if d["mode"] == m and d["target_epsilon"] == e]
        res = attack(group, a.shadows)
        tgt = group[a.shadows:]
        entry = {"mode": m, "target_epsilon": e,
                 "achieved_epsilon_max": max((d["achieved_epsilon_max"] or 0) for d in group) or None,
                 "train_accuracy": _ci([d["train_accuracy"] for d in tgt]),
                 "heldout_accuracy": _ci([d["heldout_accuracy"] for d in tgt]),
                 "generalisation_gap": _ci([d["train_accuracy"] - d["heldout_accuracy"] for d in tgt]),
                 "lira_auc": _ci([r["lira"]["auc"] for r in res]),
                 "lira_tpr_at_fpr_0.05": _ci([r["lira"]["tpr_at_fpr_0.05"] for r in res]),
                 "lira_tpr_at_fpr_0.01": _ci([r["lira"]["tpr_at_fpr_0.01"] for r in res]),
                 "lira_global_var_auc": _ci([r["lira_global_var"]["auc"] for r in res]),
                 "lira_global_var_tpr_at_fpr_0.05": _ci([r["lira_global_var"]["tpr_at_fpr_0.05"] for r in res]),
                 "lira_global_var_tpr_at_fpr_0.01": _ci([r["lira_global_var"]["tpr_at_fpr_0.01"] for r in res]),
                 "loss_threshold_auc": _ci([r["loss_threshold"]["auc"] for r in res]),
                 "per_target": res}
        summary["modes"].append(entry)
        print(m, e, "LiRA AUC %.3f  LiRA-globalvar %.3f  loss-threshold AUC %.3f  gap %.3f" % (
            entry["lira_auc"][0], entry["lira_global_var_auc"][0], entry["loss_threshold_auc"][0],
            entry["generalisation_gap"][0]), flush=True)
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Per-model scores and membership masks (public pool only) are kept for independent re-analysis.
    (a.output / "models.json").write_text(json.dumps(
        [{k: d[k] for k in ("mode", "target_epsilon", "model_id", "train_accuracy", "heldout_accuracy",
                            "achieved_epsilon_max", "seconds")} for d in done]) + "\n")
    np.savez_compressed(a.output / "scores.npz",
                        phi=np.array([d["phi"] for d in done], dtype=np.float32),
                        member=np.array([d["member"] for d in done], dtype=bool),
                        mode=np.array([d["mode"] for d in done]),
                        target_epsilon=np.array([d["target_epsilon"] if d["target_epsilon"] is not None else -1.0
                                                 for d in done]),
                        model_id=np.array([d["model_id"] for d in done]))


if __name__ == "__main__":
    main()
