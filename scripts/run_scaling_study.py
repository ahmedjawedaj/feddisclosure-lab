"""Aggregate and collusion views at 4 and 8 clients (pre-registered rows S1 and S2).

    python scripts/run_scaling_study.py --seeds 30 --clients 4 8

Evaluation seeds are 0 to N-1. Each seed is saved as it finishes, so an interrupted run resumes.
Use --legacy-seeds with --clients 4 to check that this code reproduces the saved v0.14.2 victim
numbers exactly (see `--check-v0142`). The default per-client seed is `intervals.client_seed`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feddisclosure.evaluation import scaling_study as ss  # noqa: E402
from feddisclosure.training.datasets import _source  # noqa: E402

OUT = ROOT / "reports/results/privacy/v0150/scaling"
SAVED_V0142 = ROOT / "reports/results/privacy/v014/view_comparison"
PRIMARY_AGG = f"secagg_aggregate_maxw{ss.DEFAULT_MAX_WEIGHT:g}"
PRIMARY_COL = f"secagg_collusion_n-1_maxw{ss.DEFAULT_MAX_WEIGHT:g}"


def check_against_v0142(runs4_legacy: list[dict]) -> dict:
    """Compare legacy-seed 4-client runs with the saved v0.14.2 per-seed results (same code path)."""
    diffs, compared = [], 0
    for r in runs4_legacy:
        f = SAVED_V0142 / f"seed{r['seed']}.json"
        if not f.is_file():
            continue
        saved = json.loads(f.read_text())["views"]
        for name, mses in r["views"].items():
            if name in saved:
                compared += 1
                diffs.append(abs(float(np.mean(mses)) - saved[name]["victim_mse"]))
    return {"compared_view_seed_pairs": compared, "max_abs_diff_victim_mse": max(diffs) if diffs else None,
            "reproduces_saved_v0142": bool(diffs) and max(diffs) < 1e-9}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--clients", type=int, nargs="+", default=[4, 8])
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--max-weights", type=float, nargs="+", default=[ss.DEFAULT_MAX_WEIGHT, 16.0])
    p.add_argument("--legacy-seeds", action="store_true", help="per-client seed seed*10+client (v0.14.2 check only)")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    if a.seeds < 10:
        p.error("at least 10 evaluation seeds are required")
    out = a.output or (OUT / ("legacy_seed_check" if a.legacy_seeds else "runs"))
    out.mkdir(parents=True, exist_ok=True)
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    by_n: dict[int, list[dict]] = {}
    for n in a.clients:
        runs = []
        for seed in range(a.seeds):
            f = out / f"n{n}_seed{seed}.json"
            if f.is_file():
                runs.append(json.loads(f.read_text()))
                continue
            r = ss.run_seed(x, y, seed, n, iterations=a.iterations, max_weights=a.max_weights,
                            legacy_seeds=a.legacy_seeds)
            f.write_text(json.dumps(r))
            runs.append(r)
            print(f"n={n} seed {seed} done", flush=True)
        by_n[n] = runs
    summaries = {n: ss.summarise(runs) for n, runs in by_n.items()}
    result = {"schema_version": 1, "dataset": "sklearn_digits", "records_per_client": ss.RECORDS_PER_CLIENT,
              "seeds": a.seeds, "evaluation_seeds": f"0 to {a.seeds - 1}", "attack_iterations": a.iterations,
              "lr": ss.LR, "labels_known_to_attacker": True, "legacy_client_seeds": a.legacy_seeds,
              "secagg": "upstream Flower SecAgg+ quantiser arithmetic, clipping_range 8, quantization_range 2**22",
              "primary_views": {"aggregate": PRIMARY_AGG, "collusion": PRIMARY_COL},
              "summaries": {str(n): s for n, s in summaries.items()}}
    if a.legacy_seeds and 4 in by_n:
        result["v0142_reproduction"] = check_against_v0142(by_n[4])
    if not a.legacy_seeds and 4 in summaries and 8 in summaries:
        result["S1"] = ss.check_s1(summaries[8], PRIMARY_AGG)
        result["S2"] = ss.check_s2({4: summaries[4], 8: summaries[8]}, PRIMARY_COL)
        result["S1_at_4_clients_for_reference"] = ss.check_s1(summaries[4], PRIMARY_AGG)
    result["limitations"] = [
        "Observation-model study with the real quantiser arithmetic, not an attack on captured Docker traffic",
        "One local SGD step on 8 distinct records per client, known labels, known learning rate and initial model",
        "Digits are 8x8 public images from one dataset, so the numbers say nothing about larger images",
        "Rate intervals are Wilson intervals pooled over victim images. Images of one seed are not independent, "
        "so the seed-level bootstrap interval is reported as well",
        "A failed reconstruction is evidence about this attack only, not a privacy guarantee"]
    target = out / ("summary_legacy.json" if a.legacy_seeds else "summary.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    for n, s in summaries.items():
        print(f"n={n} floor {s['label_floor']['rate']:.3f}", {k: round(v["rate"], 3) for k, v in s["views"].items()})
    for k in ("S1", "S2", "v0142_reproduction"):
        if k in result:
            print(k, "pass" if result[k].get("pass", result[k].get("reproduces_saved_v0142")) else "FAIL")


if __name__ == "__main__":
    main()
