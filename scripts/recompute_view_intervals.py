"""Recompute the v0.14.2 view-study statistics with proper intervals. No attack is rerun.

Reads the saved per-seed files under reports/results/privacy/v014/view_comparison/ and the label
floor from reports/results/privacy/v0143/label_floor.json (run scripts/compute_label_floor.py
first). For every view it writes the pooled recognisable rate over the 80 victim images with Wilson
and Clopper-Pearson intervals, a percentile bootstrap interval over seeds for the mean MSE, and the
paired difference to the label floor. It also checks that point estimates equal the saved
summary.json (pre-registered check C5) and evaluates checks C1 to C5 into
prereg_check_corrections.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feddisclosure.evaluation.intervals import (clopper_pearson_interval, wilson_interval)  # noqa: E402

SAVED = ROOT / "reports/results/privacy/v014/view_comparison"
OUT = ROOT / "reports/results/privacy/v0143"
RECORDS = 8
BOOT_RESAMPLES = 2000
BOOT_SEED = 20261123


def bootstrap_mean_ci(values: np.ndarray, resamples: int = BOOT_RESAMPLES, seed: int = BOOT_SEED,
                      confidence: float = 0.95) -> list[float]:
    """Percentile bootstrap of the mean over seeds. Returns [mean, low, high]."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    a = (1 - confidence) / 2
    return [float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def pooled_rate(rates: np.ndarray, records: int = RECORDS) -> dict:
    counts = np.asarray(rates, dtype=float) * records
    if not np.allclose(counts, np.round(counts), atol=1e-9):
        raise ValueError("per-seed rate times records is not an integer count")
    k, n = int(round(counts.sum())), int(len(counts) * records)
    return {"successes": k, "trials": n, "rate": k / n, "wilson": list(wilson_interval(k, n)),
            "clopper_pearson": list(clopper_pearson_interval(k, n))}


def load_runs(seeds: int) -> list[dict]:
    runs = []
    for s in range(seeds):
        f = SAVED / f"seed{s}.json"
        if not f.is_file():
            raise FileNotFoundError(f)
        runs.append(json.loads(f.read_text()))
    return runs


def compute(seeds: int = 10, floor_path: Path = OUT / "label_floor.json") -> dict:
    runs = load_runs(seeds)
    saved = json.loads((SAVED / "summary.json").read_text())
    floor = json.loads(floor_path.read_text())
    fl_mse = np.array([r["class_average_mse"] for r in floor["per_seed"]])
    fl_rate = np.array([r["class_average_rate_mse_below_0_05"] for r in floor["per_seed"]])
    if len(fl_mse) != seeds:
        raise ValueError("label floor and saved runs cover different seed counts")
    views, max_diff = {}, 0.0
    for name in runs[0]["views"]:
        mse = np.array([r["views"][name]["victim_mse"] for r in runs])
        rate = np.array([r["views"][name]["victim_rate_mse_below_0_05"] for r in runs])
        for key, val in (("victim_mse", mse.mean()), ("rate_mse_below_0_05", rate.mean())):
            max_diff = max(max_diff, abs(val - saved["views"][name][key][0]))
        pr = pooled_rate(rate)
        views[name] = {
            "mse_mean": float(mse.mean()), "mse_bootstrap95": bootstrap_mean_ci(mse),
            "recognisable": pr,
            "paired_mse_minus_label_floor_bootstrap95": bootstrap_mean_ci(mse - fl_mse),
            "paired_rate_minus_label_floor_bootstrap95": bootstrap_mean_ci(rate - fl_rate),
            "paired_rate_minus_label_floor_pooled": pr["rate"] - float(fl_rate.mean())}
    fl_pr = pooled_rate(fl_rate)
    return {"views": views, "label_floor": {"mse_mean": float(fl_mse.mean()), "mse_bootstrap95": bootstrap_mean_ci(fl_mse),
                                             "recognisable": fl_pr},
            "mean_image_mse_saved": saved["floor"]["mean_image_mse"][0],
            "max_abs_diff_to_saved_summary": max_diff}


POST_PROBE = ("consistency check on data already seen (post-probe), not a confirmation. The label-only probe "
              "was run on these same seeds before this check was written")


def unit_tests_ok(ut: dict) -> bool:
    """Skipped tests are not passes: needs a clean exit, no failures, no skips, at least one pass,
    and the label-floor tests must have run and passed."""
    return bool(ut.get("returncode") == 0 and ut.get("failed") == 0 and ut.get("skipped") == 0
                and ut.get("passed", 0) > 0 and ut.get("label_floor_passed", 0) > 0
                and ut.get("label_floor_skipped") == 0)


def parse_pytest_verbose(out: str) -> dict:
    """Count outcomes from `pytest -v` lines such as 'tests/x.py::test_a PASSED [ 10%]'."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "label_floor_passed": 0, "label_floor_skipped": 0}
    for line in out.splitlines():
        m = re.match(r"^(\S+\.py)(?:::\S+)?\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b", line)
        if not m:
            continue
        word = m.group(2)
        key = {"PASSED": "passed", "FAILED": "failed", "ERROR": "failed", "SKIPPED": "skipped"}.get(word)
        if key is None:
            continue
        counts[key] += 1
        if key in ("passed", "skipped") and m.group(1).endswith("test_label_floor.py"):
            counts["label_floor_" + key] += 1
    return counts


def checks(res: dict, unit_tests: dict) -> list[dict]:
    fl = res["label_floor"]
    f_mse, f_rate = fl["mse_mean"], fl["recognisable"]["rate"]
    v = res["views"]
    agg = v["secagg_aggregate_maxw1000"]
    c3 = {k: v[k]["recognisable"]["rate"] - f_rate for k in ("plaintext_individual", "secagg_collusion_n-1_maxw1000")}
    dp = v["dp0.2_individual"]["mse_mean"]
    mean_img = res["mean_image_mse_saved"]
    ints = [i for x in v.values() for i in (x["recognisable"]["wilson"], x["recognisable"]["clopper_pearson"])]
    inside = all(0.0 <= lo <= hi <= 1.0 for lo, hi in ints)
    mse_ints = [x["mse_bootstrap95"] for x in (*v.values(), fl)]
    mse_inside = all(0.0 <= lo <= m <= hi for m, lo, hi in mse_ints)
    return [
        {"id": "C1", "kind": POST_PROBE, "prediction": "Label-only floor reproduces the probe: MSE within 0.045 +- 0.005 and recognisable within 67.5% +- 5 points",
         "observed": {"floor_mse": f_mse, "floor_rate": f_rate}, "pass": bool(abs(f_mse - 0.045) <= 0.005 and abs(f_rate - 0.675) <= 0.05)},
        {"id": "C2", "kind": POST_PROBE, "prediction": "Aggregate-only inversion (n=4) is not better than the floor: rate <= floor + 5 points and MSE >= floor MSE - 0.005",
         "observed": {"aggregate_rate": agg["recognisable"]["rate"], "floor_rate": f_rate, "aggregate_mse": agg["mse_mean"], "floor_mse": f_mse},
         "pass": bool(agg["recognisable"]["rate"] <= f_rate + 0.05 and agg["mse_mean"] >= f_mse - 0.005)},
        {"id": "C3", "prediction": "Plaintext and n-1 collusion rows are at least floor + 15 points",
         "observed": {"plaintext_minus_floor": c3["plaintext_individual"], "collusion_minus_floor": c3["secagg_collusion_n-1_maxw1000"]},
         "pass": bool(all(d >= 0.15 for d in c3.values()))},
        {"id": "C4", "kind": POST_PROBE, "prediction": "Opacus DP sigma 0.2 individual view has victim MSE above the mean-image MSE",
         "observed": {"dp0.2_individual_mse": dp, "mean_image_mse": mean_img}, "pass": bool(dp > mean_img)},
        {"id": "C5", "kind": "integrity check",
         "prediction": "Rate intervals stay inside [0, 1], MSE intervals are ordered low <= mean <= high and not negative, unit tests pass (skipped tests do not count as passed), point estimates match the saved summary within 1e-9",
         "observed": {"max_abs_diff_to_saved_summary": res["max_abs_diff_to_saved_summary"], "all_rate_intervals_inside_0_1": inside,
                      "all_mse_intervals_ordered_and_nonnegative": mse_inside, "unit_tests": unit_tests},
         "pass": bool(res["max_abs_diff_to_saved_summary"] < 1e-9 and inside and mse_inside and unit_tests_ok(unit_tests))},
    ]


def run_unit_tests() -> dict:
    names = ("test_intervals.py", "test_label_floor.py", "test_view_seed_helper.py",
             "test_recompute_view_intervals.py", "test_view_comparison_helpers.py")
    files = [f"tests/training/{n}" for n in names if (ROOT / f"tests/training/{n}").is_file()]
    r = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", *files], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return {"files": files, "returncode": r.returncode, "summary": tail, **parse_pytest_verbose(r.stdout)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--skip-unit-tests", action="store_true")
    a = p.parse_args()
    res = compute(a.seeds)
    ut = {"returncode": None, "summary": "not run"} if a.skip_unit_tests else run_unit_tests()
    res_out = {"schema_version": 1, "source": "saved seed0..seed9.json, no attack rerun", "seeds": a.seeds,
               "victim_images_per_view": a.seeds * RECORDS,
               "intervals": {"rate": "pooled over 80 victim images, Wilson and Clopper-Pearson 95%",
                             "mse": f"percentile bootstrap over seeds, {BOOT_RESAMPLES} resamples, seed {BOOT_SEED}"},
               **res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "view_intervals.json").write_text(json.dumps(res_out, indent=1) + "\n")
    cs = checks(res, ut)
    (OUT / "prereg_check_corrections.json").write_text(json.dumps(cs, indent=1) + "\n")
    for c in cs:
        print(c["id"], "PASS" if c["pass"] else "FAIL", json.dumps(c["observed"]))


if __name__ == "__main__":
    main()
