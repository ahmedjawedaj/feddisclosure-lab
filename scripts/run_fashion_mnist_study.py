"""Fashion-MNIST ordering study with a small CNN (pre-registered row X1).

    python scripts/run_fashion_mnist_study.py --tune        # attack settings on tuning seeds 9000 and up
    python scripts/run_fashion_mnist_study.py --seeds 30    # evaluation seeds 0 to 29, writes summary.json
    add --download on a machine with network access when data/source/fashion_mnist is missing

The study refuses to run without the verified Fashion-MNIST files. There is no fallback to other data.
Tuning results are written to tuning.json and read by the evaluation, so the evaluation seeds never
influence the attack settings.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feddisclosure.evaluation import fashion_mnist_study as fs  # noqa: E402
from feddisclosure.evaluation import scaling_study as ss  # noqa: E402
from feddisclosure.evaluation.intervals import tuning_seed  # noqa: E402
from feddisclosure.training.fashion_mnist import PROVENANCE, SHA256, load_fashion_mnist  # noqa: E402

OUT = ROOT / "reports/results/privacy/v0150/fashion_mnist"
TUNING_SEEDS = [tuning_seed(i) for i in range(3)]
STAGE1 = {"iterations": 2000}
LR_GRID, TV_GRID = (0.1, 0.2, 0.3), (0.005, 0.01, 0.02)
AGG_LR_GRID, AGG_TV_GRID = (0.1, 0.3), (0.005, 0.02)
ITER_GRID = (2000, 3000)


def _pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with Pool(workers) as p:
        return p.map(fn, jobs)


TV_EXTENSIONS = ((0.05, 0.1), (0.2, 0.5))   # stages 3 and 4, tried at the best iterations and step size so far.
# A stage runs only if the best setting so far has the largest total-variation weight tried, so the search never stops at an edge by choice.


def _key(view, setting):
    return json.dumps([view, setting], sort_keys=True)


def tune(x, y, workers: int, cache: dict | None = None) -> dict:
    """Three stages per view, all on tuning seeds. A row that is already in `cache` is not recomputed,
    because a row is a deterministic function of (view, setting, tuning seeds)."""
    cache = {} if cache is None else cache

    def run(view, settings):
        todo = [s for s in settings if _key(view, s) not in cache]
        for row in _pool_map(fs.tune_one, [(x, y, view, s, TUNING_SEEDS) for s in todo], workers):
            cache[_key(view, row["setting"])] = row
        return [cache[_key(view, s)] for s in settings]

    rows = {"individual": [], "collusion": [], "aggregate": []}
    for view, lrs, tvs in (("individual", LR_GRID, TV_GRID), ("collusion", LR_GRID, TV_GRID),
                           ("aggregate", AGG_LR_GRID, AGG_TV_GRID)):
        rows[view] += run(view, [dict(STAGE1, attack_lr=lr, tv_weight=tv) for lr in lrs for tv in tvs])
        best = fs.best_setting(rows[view])["setting"]
        rows[view] += run(view, [dict(best, iterations=i) for i in ITER_GRID if i != best["iterations"]])
        for extension in TV_EXTENSIONS:
            best = fs.best_setting(rows[view])["setting"]
            if best["tv_weight"] < max(r["setting"]["tv_weight"] for r in rows[view]):
                break
            rows[view] += run(view, [dict(best, tv_weight=tv) for tv in extension])
        print(view, "best", fs.best_setting(rows[view]), flush=True)
    return {"tuning_seeds": TUNING_SEEDS, "criterion": "lowest mean victim MSE on the tuning seeds, per view type",
            "grid": {"individual": {"lr": LR_GRID, "tv": TV_GRID, "iterations": ITER_GRID, "tv_extensions": TV_EXTENSIONS},
                     "collusion": {"lr": LR_GRID, "tv": TV_GRID, "iterations": ITER_GRID, "tv_extensions": TV_EXTENSIONS},
                     "aggregate": {"lr": AGG_LR_GRID, "tv": AGG_TV_GRID, "iterations": ITER_GRID,
                                   "tv_extensions": TV_EXTENSIONS}},
            "rows": rows, "chosen": {v: fs.best_setting(r)["setting"] for v, r in rows.items()}}


def _seed_job(args):
    x, y, seed, settings, path = args
    if path.is_file():
        return json.loads(path.read_text())
    r = fs.run_seed(x, y, seed, settings=settings, max_weights=(1000.0, 16.0))
    path.write_text(json.dumps(r))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--download", action="store_true", help="fetch the pinned files if they are missing")
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    x, y, provenance = load_fashion_mnist(download=a.download)
    tuning_file = a.output / "tuning.json"
    if a.tune:
        t = time.time()
        cache = {}
        if tuning_file.is_file():
            for view, rows in json.loads(tuning_file.read_text())["rows"].items():
                for row in rows:
                    cache[_key(view, row["setting"])] = row
        tuning_file.write_text(json.dumps(tune(x, y, a.workers, cache), indent=1) + "\n")
        print("tuning done in", round(time.time() - t), "s")
        return
    if a.seeds < 10:
        p.error("at least 10 evaluation seeds are required")
    if not tuning_file.is_file():
        p.error("run --tune first, the evaluation reads the chosen settings from tuning.json")
    settings = json.loads(tuning_file.read_text())["chosen"]
    runs_dir = a.output / "runs"
    runs_dir.mkdir(exist_ok=True)
    jobs = [(x, y, s, settings, runs_dir / f"seed{s}.json") for s in range(a.seeds)]
    runs = _pool_map(_seed_job, jobs, a.workers)
    summary = ss.summarise(runs)
    result = {"schema_version": 1, "dataset": "fashion_mnist", "provenance": provenance, "sha256": SHA256,
              "model": "SmallCNN, two 3x3 conv with max pooling and one linear layer", "clients": fs.N_CLIENTS,
              "records_per_client": fs.RECORDS_PER_CLIENT, "lr": fs.LR, "seeds": a.seeds,
              "evaluation_seeds": f"0 to {a.seeds - 1}", "attack_settings_from_tuning_seeds": settings,
              "labels_known_to_attacker": True,
              "secagg": "upstream Flower SecAgg+ quantiser arithmetic, clipping_range 8, quantization_range 2**22",
              "summary": summary, "X1": fs.check_x1(summary, 1000.0),
              "X1_secondary_max_weight_16": fs.check_x1(summary, 16.0),
              "primary_max_weight": 1000.0,
              "limitations": [
                  "Observation-model study with the real quantiser arithmetic, not an attack on captured Docker traffic",
                  "One local SGD step on 8 distinct records per client, known labels, known learning rate and initial model",
                  "One small CNN with ReLU and max pooling. Other architectures can be easier or harder to invert",
                  "Attack settings were tuned on 3 tuning seeds, a stronger attack could do better",
                  "Rate intervals are Wilson intervals pooled over victim images, plus a seed-level bootstrap",
                  "A failed reconstruction is evidence about this attack only, not a privacy guarantee"]}
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["X1"], indent=1))
    print("secondary maxw16 pass:", result["X1_secondary_max_weight_16"]["pass"])


if __name__ == "__main__":
    main()
