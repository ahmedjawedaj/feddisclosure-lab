"""WP11 external baselines (B1 inversion against `breaching`, B2 LiRA against a reference LiRA).

Run with the isolated environment that has `breaching` and `mialibrary`, for example
    PYTHONPATH=src:. /tmp/venv_ext/bin/python scripts/run_external_baselines.py tune
    PYTHONPATH=src:. /tmp/venv_ext/bin/python scripts/run_external_baselines.py inversion --max-seconds 420
    PYTHONPATH=src:. /tmp/venv_ext/bin/python scripts/run_external_baselines.py lira
    PYTHONPATH=src:. /tmp/venv_ext/bin/python scripts/run_external_baselines.py check

Stages
  tune       choose the breaching configuration on tuning seeds (9000 and up), never on evaluation seeds
  inversion  repo attack and every breaching configuration on evaluation seeds 0..N-1, one JSON per seed
             (resumable, `--max-seconds` stops cleanly between seeds)
  lira       repo LiRA against the reference LiRA on the saved shadow and target models
  check      writes reports/results/privacy/v0150/prereg_check_external.json from the saved stage outputs
Thresholds come from docs/preregistration/v0.15.md and are constants in
feddisclosure.evaluation.external_baselines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/results/privacy/v0150/external"
PREREG_OUT = ROOT / "reports/results/privacy/v0150/prereg_check_external.json"
SAVED_LIRA = ROOT / "reports/results/privacy/v014/lira/large_overfit/scores.npz"
LR, B, N = 0.15, 8, 4
REPO_ITERATIONS = 300     # the saved v0.14.2 view study setting
LIRA_SHADOWS = 64

# Configurations run on the evaluation seeds. Library names refer to breaching.config.attack.
# `ig_library_default` is the shipped invertinggradients yaml (24000 iterations). `dlg_known_labels` is
# the shipped deepleakage yaml (Euclidean objective, L-BFGS, 1200 iterations) with the joint label
# optimisation switched off because the labels are known here. The tuned entry is filled in by `tune`.
FIXED_CONFIGS = {
    "ig_library_default": ("invertinggradients", ["optim.max_iterations=24000"]),
    "dlg_known_labels": ("deepleakage", ["attack_type=optimization"]),
}
TUNE_CANDIDATES = {
    "ig_tv0.2": ("invertinggradients", ["optim.max_iterations=4000", "regularization.total_variation.scale=0.2"]),
    "ig_tv0.05": ("invertinggradients", ["optim.max_iterations=4000", "regularization.total_variation.scale=0.05"]),
    "ig_tv0.01": ("invertinggradients", ["optim.max_iterations=4000", "regularization.total_variation.scale=0.01"]),
    "ig_tv0.001": ("invertinggradients", ["optim.max_iterations=4000", "regularization.total_variation.scale=0.001"]),
    "ig_tv0": ("invertinggradients", ["optim.max_iterations=4000", "regularization.total_variation.scale=0.0"]),
    "dlg_known_labels": FIXED_CONFIGS["dlg_known_labels"],
}


def _setup(seed: int):
    import torch
    from feddisclosure.attacks.view_study import client_step
    from feddisclosure.evaluation.external_baselines import matched_records, victim_gradient
    from feddisclosure.training.datasets import _source
    from feddisclosure.training.task import DigitMLP
    torch.set_num_threads(1)
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    rec = matched_records(x, y, seed)
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in DigitMLP(64, 10).state_dict().items()}
    state = client_step(init, rec["victim_x"], rec["victim_y"], lr=LR, sigma=None, seed=seed * 10)
    delta = {k: init[k] - state[k] for k in init}
    return rec, init, delta, victim_gradient(init, delta, lr=LR)


def _score(truth, rec):
    from feddisclosure.attacks.view_study import score_victim
    return score_victim(truth, rec, len(truth))


def run_breaching_config(seed: int, name: str, attack: str, overrides: list[str]) -> dict:
    from feddisclosure.evaluation.external_baselines import breaching_reconstruct
    rec, init, _, grads = _setup(seed)
    t = time.perf_counter()
    r = breaching_reconstruct(init, grads, rec["victim_y"], attack=attack, overrides=overrides, seed=seed + 1)
    return {**_score(rec["victim_x"], r), "seconds": round(time.perf_counter() - t, 1)}


def run_repo(seed: int) -> dict:
    from feddisclosure.attacks.view_study import attack
    rec, init, delta, _ = _setup(seed)
    t = time.perf_counter()
    r = attack(init, delta, rec["victim_y"], "server_individual_model_delta", lr=LR,
               iterations=REPO_ITERATIONS, seed=seed + 1)
    return {**_score(rec["victim_x"], r), "seconds": round(time.perf_counter() - t, 1)}


def run_floor(seed: int) -> dict:
    from feddisclosure.attacks.view_study import label_floor
    rec, *_ = _setup(seed)
    lf = label_floor(rec["victim_x"], rec["victim_y"], rec["pool_x"], rec["pool_y"], u=10,
                     rng=np.random.default_rng(5000 + seed))
    return {"class_average_mse": lf["class_average_mse"],
            "class_average_rate_mse_below_0_05": lf["class_average_rate_mse_below_0_05"],
            "matched_mse": lf["matched_mse"], "matched_rate_mse_below_0_05": lf["matched_rate_mse_below_0_05"]}


def stage_tune(a) -> None:
    from feddisclosure.evaluation.intervals import tuning_seed
    d = OUT / "tuning"
    d.mkdir(parents=True, exist_ok=True)
    seeds = [tuning_seed(i) for i in range(a.tune_seeds)]
    results = {}
    for name, (attack, ov) in TUNE_CANDIDATES.items():
        rows = []
        for s in seeds:
            f = d / f"{name}_seed{s}.json"
            if not f.is_file():
                f.write_text(json.dumps(run_breaching_config(s, name, attack, ov)))
            rows.append(json.loads(f.read_text()))
            print(name, s, "mse %.4f" % rows[-1]["victim_mse"], flush=True)
        results[name] = {"mean_mse": float(np.mean([r["victim_mse"] for r in rows])),
                         "attack": attack, "overrides": ov}
    best = min(results, key=lambda k: results[k]["mean_mse"])
    (d / "selection.json").write_text(json.dumps({
        "tuning_seeds": seeds, "candidates": results, "selected": best,
        "rule": "lowest mean victim MSE over the tuning seeds. Evaluation seeds were not used."}, indent=2) + "\n")
    print("selected", best)


def configs() -> dict:
    sel = json.loads((OUT / "tuning/selection.json").read_text())
    best = sel["selected"]
    cfg = dict(FIXED_CONFIGS)
    cfg[f"tuned_{best}"] = (sel["candidates"][best]["attack"], sel["candidates"][best]["overrides"])
    return cfg


def stage_inversion(a) -> None:
    d = OUT / "inversion"
    d.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cfg = configs()
    for s in range(a.seeds):
        f = d / f"seed{s}.json"
        row = json.loads(f.read_text()) if f.is_file() else {"seed": s, "attacks": {}}
        todo = [k for k in ["repo", *cfg] if k not in row["attacks"]]
        if "floor" not in row:
            row["floor"] = run_floor(s)
        for k in todo:
            if time.perf_counter() - start > a.max_seconds:
                f.write_text(json.dumps(row, indent=1))
                print("time budget reached, resume later", flush=True)
                return
            row["attacks"][k] = run_repo(s) if k == "repo" else run_breaching_config(s, k, *cfg[k])
            print("seed", s, k, "mse %.4f rate %.3f %.0fs" % (row["attacks"][k]["victim_mse"],
                  row["attacks"][k]["victim_rate_mse_below_0_05"], row["attacks"][k]["seconds"]), flush=True)
            f.write_text(json.dumps(row, indent=1))
        f.write_text(json.dumps(row, indent=1))
    print("inversion complete", flush=True)


def stage_summarise_inversion(a) -> dict:
    from feddisclosure.evaluation.external_baselines import b1_verdict, bootstrap_mean_ci
    rows = [json.loads((OUT / f"inversion/seed{s}.json").read_text()) for s in range(a.seeds)]
    cfg = configs()
    names = ["repo", *cfg]
    if any(n not in r["attacks"] for r in rows for n in names):
        raise SystemExit("inversion stage incomplete, run it first")
    per = {n: {"mse": [r["attacks"][n]["victim_mse"] for r in rows],
               "rate": [r["attacks"][n]["victim_rate_mse_below_0_05"] for r in rows]} for n in names}
    floor = {"class_average_mse": [r["floor"]["class_average_mse"] for r in rows],
             "class_average_rate": [r["floor"]["class_average_rate_mse_below_0_05"] for r in rows],
             "matched_mse": [r["floor"]["matched_mse"] for r in rows],
             "matched_rate": [r["floor"]["matched_rate_mse_below_0_05"] for r in rows]}
    out = {"seeds": a.seeds, "attacks": {}, "label_floor": {}, "b1": {}}
    for n in names:
        out["attacks"][n] = {"mse_mean_bootstrap_ci": bootstrap_mean_ci(per[n]["mse"]),
                             "recognisable_share": float(np.mean(per[n]["rate"])),
                             "mean_seconds": float(np.mean([r["attacks"][n]["seconds"] for r in rows]))}
    for k, v in floor.items():
        out["label_floor"][k] = bootstrap_mean_ci(v)
    for n in cfg:
        out["b1"][n] = b1_verdict(per["repo"]["mse"], per[n]["mse"])
        diff = np.array(per["repo"]["mse"]) - np.array(per[n]["mse"])
        out["b1"][n]["paired_mse_repo_minus_external_ci"] = bootstrap_mean_ci(diff)
        out["b1"][n]["seeds_repo_lower_mse"] = int((diff < 0).sum())
    return out


def stage_lira(a) -> dict:
    from feddisclosure.evaluation import external_baselines as eb
    models = eb.load_saved_lira(SAVED_LIRA)
    n_models = len(models["phi"])
    shadows = LIRA_SHADOWS
    targets = n_models - shadows
    sh, tg = eb.split_models(models, shadows)
    variants = {"mean_perexample_var": dict(fix_variance=False, use_median=False),
                "mean_global_var": dict(fix_variance=True, use_median=False),
                "median_perexample_var": dict(fix_variance=False, use_median=True),
                "median_global_var": dict(fix_variance=True, use_median=True)}
    members = list(tg["member"])
    scores = {"repo": eb.evaluate_lira(models, shadows, eb.repo_lira_scores)}
    for k, kw in variants.items():
        scores[f"mialibrary_{k}"] = eb.evaluate_lira(
            models, shadows, lambda p, sp, sm, kw=kw: eb.mialibrary_scores(p, sp, sm, **kw))
        scores[f"independent_{k}"] = eb.evaluate_lira(
            models, shadows, lambda p, sp, sm, kw=kw: eb.independent_lira_scores(p, sp, sm, **kw))
    out = {"setting": {"source": str(SAVED_LIRA.relative_to(ROOT)), "dataset": "sklearn_digits", "regime": "overfit",
                       "mode": "nonprivate (no DP)", "shadows": shadows, "targets": targets,
                       "pool": int(models["phi"].shape[1])}, "attacks": {}, "paired_vs_repo": {}}
    for k, (sc, aucs) in scores.items():
        ci = eb.bootstrap_auc(sc, members, n_boot=a.n_boot, seed=1)
        out["attacks"][k] = {"auc_mean": float(np.mean(aucs)), "auc_per_target": [float(v) for v in aucs],
                             "auc_bootstrap_ci": [ci["mean"], ci["low"], ci["high"]]}
    for k in scores:
        if k == "repo":
            continue
        d = eb.bootstrap_auc(scores["repo"][0], members, n_boot=a.n_boot, seed=2, paired_with=scores[k][0])
        out["paired_vs_repo"][k] = {"repo_minus_reference_auc": [d["mean"], d["low"], d["high"]]}
    out["agreement_independent_vs_mialibrary"] = {
        v: float(np.max(np.abs(np.concatenate(scores[f"independent_{v}"][0]) -
                               np.concatenate(scores[f"mialibrary_{v}"][0])))) for v in variants}
    return out


def pick_b2_reference(lira: dict) -> tuple[str, str]:
    """Headline reference: the strongest mialibrary variant by mean AUC. Strongest is the most demanding
    choice for the repo and is fixed here before any comparison is read."""
    cand = {k: v["auc_mean"] for k, v in lira["attacks"].items() if k.startswith("mialibrary_")}
    best = max(cand, key=cand.get)
    return best, "mialibrary_mean_perexample_var"


def stage_check(a) -> None:
    from feddisclosure.evaluation.external_baselines import b2_verdict, external_versions
    res = {"schema_version": 1, "work_package": "WP11", "versions": external_versions(), "checks": {}}
    inv_dir = OUT / "inversion_summary.json"
    lira_file = OUT / "lira_summary.json"
    sel = OUT / "tuning/selection.json"
    if inv_dir.is_file() and sel.is_file():
        inv = json.loads(inv_dir.read_text())
        selected = json.loads(sel.read_text())["selected"]
        tuned = f"tuned_{selected}"
        res["checks"]["B1"] = {
            "claim": "Our inversion attack is not much weaker than breaching on a matched setup",
            "rule": "Our mean MSE at most 1.5 times the external baseline MSE",
            "headline_external_configuration": tuned,
            "headline_rule": "the breaching configuration selected on tuning seeds (strongest available)",
            "verdict": inv["b1"][tuned]["verdict"], "headline": inv["b1"][tuned],
            "all_configurations": {k: {"ratio": v["ratio"], "verdict": v["verdict"]} for k, v in inv["b1"].items()},
            "seeds": inv["seeds"]}
    else:
        res["checks"]["B1"] = {"verdict": "NOT RUN", "reason": "inversion summary missing"}
    if lira_file.is_file():
        lira = json.loads(lira_file.read_text())
        strongest, like = pick_b2_reference(lira)
        repo_auc = lira["attacks"]["repo"]["auc_mean"]
        res["checks"]["B2"] = {
            "claim": "Our LiRA AUC is not much weaker than a reference LiRA",
            "rule": "Our AUC at least reference AUC minus 0.03",
            "reference_implementation": "mialibrary 0.1.5 simple_lira (PyPI, independent of this repo)",
            "headline_reference": strongest,
            "headline_rule": "strongest mialibrary variant by mean AUC (most demanding for the repo)",
            "verdict": b2_verdict(repo_auc, lira["attacks"][strongest]["auc_mean"])["verdict"],
            "headline": b2_verdict(repo_auc, lira["attacks"][strongest]["auc_mean"]),
            "like_for_like": {"reference": like, **b2_verdict(repo_auc, lira["attacks"][like]["auc_mean"])},
            "all_references": {k: b2_verdict(repo_auc, v["auc_mean"])["verdict"]
                               for k, v in lira["attacks"].items() if k != "repo"}}
    else:
        res["checks"]["B2"] = {"verdict": "NOT RUN", "reason": "lira summary missing"}
    PREREG_OUT.write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res["checks"].items():
        print(k, v["verdict"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["tune", "inversion", "summarise-inversion", "lira", "check"])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--tune-seeds", type=int, default=3)
    p.add_argument("--max-seconds", type=float, default=420.0)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "tune":
        stage_tune(a)
    elif a.stage == "inversion":
        stage_inversion(a)
    elif a.stage == "summarise-inversion":
        (OUT / "inversion_summary.json").write_text(json.dumps(stage_summarise_inversion(a), indent=2) + "\n")
    elif a.stage == "lira":
        (OUT / "lira_summary.json").write_text(json.dumps(stage_lira(a), indent=2) + "\n")
    else:
        stage_check(a)


if __name__ == "__main__":
    sys.exit(main())
