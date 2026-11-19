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
    multi = read("reports/results/sprint3/multirecord_update/multirecord_update_summary.json")
    pairs = read("reports/results/sprint2/ablations/paired/paired_summary.json")
    assert [(a["batch_size"], a["local_steps"]) for a in multi["runs"]] == [
        (1, 1), (8, 1), (8, 4), (32, 1), (32, 4)]
    assert all(math.isfinite(x["image_mean_mse"]) and 0 <= x["mean_ssim"] <= 1
               for x in multi["runs"])
    assert pairs["completed_seeds"] == 3 and len(pairs["results"]) == 3
    assert {x["seed"] for x in pairs["results"]} == {241, 242, 243}
    assert pairs["matched_fields"] and pairs["different_fields"] == ["algorithm", "proximal_mu"]
    assert all(x["fedavg"]["model_bytes_total"] == x["fedprox"]["model_bytes_total"]
               for x in pairs["results"])
    import hashlib
    for seed in (241, 242, 243):
        for algorithm in ("fedavg", "fedprox"):
            artifact = read(f"reports/results/sprint2/ablations/paired/{algorithm}-{seed}.json")
            checkpoint = root / artifact["final_checkpoint"]["path"]
            # Absolute paths in saved artifacts are allowed only if they point
            # inside this extracted release; handle relocations by suffix.
            if not checkpoint.is_file():
                checkpoint = root / "reports/results/sprint2/ablations/paired/checkpoints" / (artifact["config"]["id"] + "_final.pt")
            assert checkpoint.is_file(), f"Missing ablation checkpoint {checkpoint}"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == artifact["final_checkpoint"]["sha256"]
    for fig in ("reports/figures/sprint3_multirecord_inversion.png",
                "reports/figures/sprint2_fedprox_paired.png"):
        assert (root / fig).stat().st_size > 2000
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
            "gradient_trials": 10, "membership_query_trials": 5, "model_update_trials": 3, "multirecord_update_trials": 5,
            "fedprox_matched_seeds": 3,
            "baseline_checkpoint_sha256": actual_hash,
            "docker_native": "blocked", "flower_native": "blocked"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    args = p.parse_args()
    print(json.dumps(verify(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
