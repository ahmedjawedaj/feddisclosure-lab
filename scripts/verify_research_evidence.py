"""Read-only integrity checks for v0.4 public-data experiment handoff."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(root: Path) -> dict:
    def read(rel):
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing research evidence: {rel}")
        return json.loads(path.read_text())
    sweep = read("reports/results/sprint2/sweep/sweep_summary.json")
    gradients = read("reports/results/sprint3/matrix/matrix_summary.json")
    membership = read("reports/results/sprint3/membership_matrix/membership_matrix_summary.json")
    update = read("reports/results/sprint3/model_update_inversion/update_attack_summary.json")
    runtime = read("reports/results/runtime_gates/status.json")
    assert sweep["runs_completed"] == 10
    assert sweep["conditions"]["iid"]["n"] == sweep["conditions"]["label_skew"]["n"] == 5
    assert len(sweep["paired_differences"]) == 5
    assert len(gradients["pairs"]) == 10
    assert all(math.isfinite(p["mean_mse"]) for p in gradients["pairs"])
    assert membership["n"] == 5 and len(membership["runs"]) == 5
    assert [x["local_steps"] for x in update["runs"]] == [1, 4, 16]
    assert all(math.isfinite(x["image_mean_mse"]) for x in update["runs"])
    assert update["runs"][0]["gradient_proxy_relative_l2_error"] < 1e-5
    assert not runtime["native_docker"]["verified"] and not runtime["native_flower"]["verified"]
    baseline = read("reports/results/sprint2/torch_iid_reference.json")
    chk = root / "reports/results/sprint2/checkpoints/digits-iid-smoke-v1_final.pt"
    # Archived public-digit checkpoint is optional for Git source checkout but
    # mandatory for *this* full-evidence release bundle.
    if not chk.exists():
        raise FileNotFoundError("Missing recorded public-digits model checkpoint")
    actual_hash = hashlib.sha256(chk.read_bytes()).hexdigest()
    assert actual_hash == baseline["final_checkpoint"]["sha256"]
    for rel in ("reports/figures/sprint2_five_seed_accuracy.png",
                "reports/figures/sprint3_gradient_comparison.png",
                "reports/figures/sprint3_membership_roc.png",
                "reports/figures/sprint3_model_update_inversion.png"):
        assert (root / rel).stat().st_size > 2000
    return {"status": "verified", "training_runs": 10,
            "gradient_trials": 10, "membership_query_trials": 5, "model_update_trials": 3,
            "baseline_checkpoint_sha256": actual_hash,
            "docker_native": "blocked", "flower_native": "blocked"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    args = p.parse_args()
    print(json.dumps(verify(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
