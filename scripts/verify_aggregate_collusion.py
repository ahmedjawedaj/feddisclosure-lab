"""Read-only integrity/evaluation check for exact five-seed aggregate-collusion study."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.evaluation.gradient import score_reconstruction
from feddisclosure.evaluation.aggregate_observer import CASES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder: Path, seeds=range(241, 246)) -> dict:
    rows = []
    for seed in seeds:
        one = folder / f"seed{seed}"
        summary = json.loads((one / "summary.json").read_text())
        contract = summary["contract"]
        if summary["attack_iterations"] != 160 or contract["seed"] != seed or contract["sites"] != 3:
            raise AssertionError("Mismatch with predeclared experiment")
        truth = one / "evaluator_only" / "target.npz"
        assert sha(truth) == contract["target_evaluator_sha256"]
        with np.load(truth, allow_pickle=False) as d:
            x = d["x"].copy()
        assert x.shape == (8, 64)
        for case in CASES:
            attack = one / case / "attacker"
            obs = attack / "observation.pt"
            assert sha(obs) == contract["cases"][case]["observation_sha256"]
            o = torch.load(obs, map_location="cpu", weights_only=True)
            assert len(o["known_labels"]) == 8
            assert not any(k in o for k in ("x", "source_idx", "images", "ground_truth"))
            assert not (attack / "target.npz").exists()
            rec = attack / "recovered.npz"
            assert sha(rec) == summary["results"][case]["reconstruction_sha256"]
            with np.load(rec, allow_pickle=False) as r:
                metrics = score_reconstruction(x, r["x"])
            eval_path = one / case / "evaluation.json"
            assert sha(eval_path) == summary["results"][case]["evaluator_sha256"]
            stored = json.loads(eval_path.read_text())
            for source, expected in (("mean_mse", "mse"), ("mean_psnr_db", "psnr"), ("mean_ssim", "ssim")):
                assert math.isclose(metrics[source], stored[source], rel_tol=1e-6, abs_tol=1e-7)
                assert math.isclose(metrics[source], summary["results"][case][expected], rel_tol=1e-6, abs_tol=1e-7)
            rows.append({"seed": seed, "case": case, "mse": metrics["mean_mse"]})
        assert math.isclose(summary["results"]["individual"]["mse"],
                            summary["results"]["all_but_one_collusion"]["mse"], abs_tol=1e-6)
        assert contract["collusion_isolation_max_tensor_error"] < 2e-6
    means = {case: float(np.mean([r["mse"] for r in rows if r["case"] == case])) for case in CASES}
    return {"seeds": list(seeds), "cases": list(CASES), "n_attack_executions": len(rows),
            "mean_mse": means, "integrity": "passed",
            "warning": "No Flower SecAgg+ or cryptographic security claimed"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, default=ROOT / "reports/results/sprint4/aggregate_collusion")
    args = p.parse_args()
    print(json.dumps(verify(args.folder), indent=2))

if __name__ == "__main__":
    main()
