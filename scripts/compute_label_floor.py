"""Label-conditional floor for the v0.14.2 view study (evaluation seeds, no attack is run).

The attacker in the view study knows the victim's labels. The right floor for such an attacker
rebuilds every victim record as the class-average image of the public pool (all images not held by
the 4 clients), not as the overall mean image. Data selection is the exact one used by
`scripts/run_view_comparison.py` through `feddisclosure.attacks.view_seed.select_view_records`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feddisclosure.attacks.view_seed import select_view_records  # noqa: E402
from feddisclosure.attacks.view_study import chance_floor, label_floor  # noqa: E402
from feddisclosure.training.datasets import _source  # noqa: E402

N, B = 4, 8
POOL_SIZES = (20, 50, 100, 300)
SUBSETS = 20
SENS_SEED_BASE = 7000
DEFAULT_OUT = ROOT / "reports/results/privacy/v0143/label_floor.json"
SAVED = ROOT / "reports/results/privacy/v014/view_comparison"


def floor_for_seed(x: np.ndarray, y: np.ndarray, seed: int, *, u: int = 10, repeats: int = 200) -> dict:
    sel = select_view_records(len(y), N, B, seed)
    victim = sel.idx[:B]
    pool_x, pool_y = x[sel.pool_idx], y[sel.pool_idx]
    assert not set(victim.tolist()) & set(sel.pool_idx.tolist()), "victim leaked into the public pool"
    fl = label_floor(x[victim], y[victim], pool_x, pool_y, u=u, repeats=repeats,
                     rng=np.random.default_rng(5000 + seed))
    mean_img = chance_floor(x[victim], pool_x, sel.rng)["mean_image_mse"]
    per_image_mean = ((x[victim] - pool_x.mean(axis=0)) ** 2).mean(axis=1)
    fl.update({"seed": seed, "mean_image_mse": float(mean_img),
               "mean_image_rate_mse_below_0_05": float((per_image_mean < 0.05).mean()),
               "public_pool_size": int(len(sel.pool_idx))})
    return fl


def subset_floor(victim_x: np.ndarray, victim_y: np.ndarray, sub_x: np.ndarray, sub_y: np.ndarray) -> dict:
    """Label floor from a small public subset. A class with no record in the subset falls back to the
    subset mean image, which is what an attacker holding only that subset could do."""
    mean_img = sub_x.mean(axis=0)
    cm = np.tile(mean_img, (10, 1))
    missing = 0
    for c in range(10):
        members = sub_x[sub_y == c]
        if len(members):
            cm[c] = members.mean(axis=0)
        else:
            missing += 1
    per_image = ((victim_x - cm[victim_y.astype(int)]) ** 2).mean(axis=1)
    return {"mse": float(per_image.mean()), "rate": float((per_image < 0.05).mean()), "fallback": missing > 0}


def pool_size_sensitivity(x: np.ndarray, y: np.ndarray, seeds: int, *, sizes=POOL_SIZES, subsets: int = SUBSETS) -> dict:
    """Label floor as a function of the number of labelled public images the attacker holds.

    For each evaluation seed and each pool size, `subsets` random subsets of that size are drawn without
    replacement from the public pool of that seed (rng seeded with [7000, seed, size], so the numbers are
    reproducible). MSE and recognisable rate are averaged over subsets and seeds. The last row is the
    full pool (one subset, no sampling)."""
    rows = []
    for size in (*sizes, None):
        mses, rates, fallbacks = [], [], []
        for seed in range(seeds):
            sel = select_view_records(len(y), N, B, seed)
            v = sel.idx[:B]
            rng = np.random.default_rng([SENS_SEED_BASE, seed, size or 0])
            for _ in range(1 if size is None else subsets):
                pidx = sel.pool_idx if size is None else rng.choice(sel.pool_idx, size, replace=False)
                r = subset_floor(x[v], y[v], x[pidx], y[pidx])
                mses.append(r["mse"]); rates.append(r["rate"]); fallbacks.append(r["fallback"])
        rows.append({"pool_size": int(size) if size else "full", "subsets_per_seed": 1 if size is None else subsets,
                     "mse": float(np.mean(mses)), "rate_mse_below_0_05": float(np.mean(rates)),
                     "share_of_subsets_missing_a_class": float(np.mean(fallbacks))})
    return {"definition": "class-average floor built from a random labelled subset of the public pool",
            "rng": f"default_rng([{SENS_SEED_BASE}, seed, pool_size])", "subsets_per_seed": subsets, "seeds": seeds,
            "missing_class_rule": "a class absent from the subset is rebuilt as the subset mean image",
            "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--u", type=int, default=10)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    per_seed = [floor_for_seed(x, y, s, u=a.u, repeats=a.repeats) for s in range(a.seeds)]
    # The saved v0.14.2 mean-image floor is an independent check that the selection is exact.
    saved_diff = []
    for r in per_seed:
        f = SAVED / f"seed{r['seed']}.json"
        if f.is_file():
            saved_diff.append(abs(json.loads(f.read_text())["floor"]["mean_image_mse"] - r["mean_image_mse"]))
    pooled = {k: float(np.mean([r[k] for r in per_seed])) for k in
              ("class_average_mse", "class_average_rate_mse_below_0_05", "matched_mse",
               "matched_rate_mse_below_0_05", "mean_image_mse", "mean_image_rate_mse_below_0_05",
               "identification_rate")}
    pooled["victim_images"] = a.seeds * B
    sensitivity = pool_size_sensitivity(x, y, a.seeds)
    saved_summary = SAVED / "summary.json"
    if saved_summary.is_file():
        sv = json.loads(saved_summary.read_text())["views"]
        sensitivity["aggregate_attack_reference"] = {
            "view": "secagg_aggregate_maxw1000", "note": "this attack uses no public images",
            "mse": sv["secagg_aggregate_maxw1000"]["victim_mse"][0],
            "rate_mse_below_0_05": sv["secagg_aggregate_maxw1000"]["rate_mse_below_0_05"][0]}
    out = {"schema_version": 1, "dataset": "sklearn_digits", "clients": N, "records_per_client": B,
           "seeds": a.seeds, "selection": "default_rng(1000+seed), permutation, first 32 records held by clients, rest public pool",
           "floor_definition": "every victim record is rebuilt as the class-average image of the public pool, labels known",
           "recognisable_definition": "per-image MSE below 0.05",
           "headline_scoring": "class_average_* fields, direct per-record MSE with no matching. matched_* fields use optimal matching as the attack views do.",
           "identification": {"u": a.u, "repeats": a.repeats, "chance": 1.0 / a.u,
                              "note": "Not comparable to MSE. Rate at which the true record is nearest to the class-average image among u same-label public candidates."},
           "pool_description": "labelled public pool of 1765 images per seed, the images no client holds. The 4 clients hold 32 records. The attack itself uses no public images.",
           "pool_size_sensitivity": sensitivity,
           "pooled": pooled, "per_seed": per_seed,
           "check_against_saved_mean_image_floor": {"seeds_compared": len(saved_diff),
                                                    "max_abs_diff": max(saved_diff) if saved_diff else None},
           "limitations": ["Evaluation seeds 0 to 9 only, the same ones as v0.14.2",
                           "Floor uses the public pool of the same seed, 1765 images, victim images excluded",
                           "Digits only"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(pooled, indent=1))
    for r in sensitivity["rows"]:
        print("pool", r["pool_size"], "mse %.4f rate %.3f" % (r["mse"], r["rate_mse_below_0_05"]))
    print("max abs diff to saved mean-image floor:", out["check_against_saved_mean_image_floor"]["max_abs_diff"])


if __name__ == "__main__":
    main()
