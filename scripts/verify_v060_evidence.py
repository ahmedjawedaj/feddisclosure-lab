"""Strict read-only checks for new public-digit v0.6 research artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import torch


def verify(root: Path) -> dict:
    summary_root = root / "reports/results/sprint3/changing_minibatches/seed241"
    summary = json.loads((summary_root / "minibatch_update_summary.json").read_text())
    expected = [(8, 1, 0, 8), (8, 4, 0, 32), (8, 4, 1, 32)]
    got = [(r["minibatch_size"], r["local_steps"], r["schedule_seed"], r["total_distinct_records"])
           for r in summary["cases"]]
    assert got == expected, f"Unexpected changing-minibatch experiment matrix: {got}"
    assert summary["attacker_has_all_contributing_labels"] and not summary["attacker_has_true_images"]
    for row in summary["cases"]:
        assert 0 <= row["reconstruction_mse"] <= 1
        assert math.isfinite(row["gradient_proxy_relative_l2_error"])
        contract = json.loads((summary_root / row["contract_path"]).read_text())
        observer_path = (summary_root / row["contract_path"]).parent / "attacker" / "observation.pt"
        assert hashlib.sha256(observer_path.read_bytes()).hexdigest() == contract["observation_sha256"]
        observer = torch.load(observer_path, map_location="cpu", weights_only=True)
        assert not any(key in observer for key in ("x", "images", "source_idx", "ground_truth", "schedule"))
        assert len(observer["known_labels"]) == row["total_distinct_records"]
        evaluator = json.loads((summary_root / row["evaluation_path"]).read_text())
        assert abs(evaluator["mean_mse"] - row["reconstruction_mse"]) < 1e-12
    assert (root / "reports/figures/sprint3_changing_minibatches.png").stat().st_size > 2000
    extended = root / "reports/results/sprint2/extended/fedavg_labelskew_8round_seed241.json"
    evidence = {"changing_minibatch_cases": len(summary["cases"]), "figure": "verified"}
    if extended.is_file():
        training = json.loads(extended.read_text())
        assert training["engine"] == "torch_independent_process_reference_NOT_Flower"
        assert len(training["rounds"]) == 8 and training["config"]["seed"] == 241
        checkpoint = extended.parent / "checkpoints" / "fedavg-labelskew-8round-seed241_final.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == training["final_checkpoint"]["sha256"]
        evidence["extended_fedavg_accuracy"] = training["rounds"][-1]["public_test"]["accuracy"]
        evidence["extended_worst_client_accuracy"] = training["final_global_worst_client_validation_accuracy"]
    else:
        raise FileNotFoundError("Missing completed 8-round FedAvg reference benchmark")
    prox_path = root / "reports/results/sprint2/extended/fedprox_labelskew_8round_seed241.json"
    prox = json.loads(prox_path.read_text())
    assert prox["config"]["proximal_mu"] == 0.3 and prox["config"]["algorithm"] == "fedprox"
    assert len(prox["rounds"]) == 8 and prox["config"]["seed"] == 241
    assert hashlib.sha256((prox_path.parent / "checkpoints" / "fedprox-labelskew-8round-seed241_final.pt").read_bytes()).hexdigest() == prox["final_checkpoint"]["sha256"]
    for r0, r1 in zip(training["rounds"], prox["rounds"]):
        assert r0["round"] == r1["round"]
    for method, run in (("fedavg", training), ("fedprox", prox)):
        shorter = json.loads((root / f"reports/results/sprint2/ablations/paired/{method}-241.json").read_text())
        assert [r["public_test"]["accuracy"] for r in run["rounds"][:4]] == [r["public_test"]["accuracy"] for r in shorter["rounds"]]
    evidence["extended_fedprox_accuracy"] = prox["rounds"][-1]["public_test"]["accuracy"]
    evidence["extended_fedprox_worst_client_accuracy"] = prox["final_global_worst_client_validation_accuracy"]
    assert (root / "reports/figures/sprint2_extended_8round_paired.png").stat().st_size > 2000
    mia_root = root / "reports/results/sprint3/membership_training_seeds"
    mia = json.loads((mia_root / "membership_training_seeds_summary.json").read_text())
    assert mia["training_seed_count"] == 3 and {x["training_seed"] for x in mia["runs"]} == {241, 242, 243}
    assert mia["methods_pre_registered"] == ["max_confidence", "negative_entropy", "top2_margin"]
    for row in mia["runs"]:
        folder = mia_root / f"model_seed_{row['training_seed']}"
        with __import__('numpy').load(folder / "attacker" / "model_predictions.npz", allow_pickle=False) as data:
            assert data.files == ["probabilities"]
            assert len(data["probabilities"]) == row["query_count"]
        for method in mia["methods_pre_registered"]:
            score = json.loads((folder / "methods" / method / "evaluation.json").read_text())
            assert score["roc_auc"] == row["methods"][method]["roc_auc"]
            assert 0 <= score["roc_auc"] <= 1
    moderate_path = root / "reports/results/sprint2/extended/fedavg_dirichlet_8round_seed241.json"
    moderate = json.loads(moderate_path.read_text())
    assert moderate["config"]["partition"] == "dirichlet_0.5" and len(moderate["rounds"]) == 8
    assert moderate["config"]["seed"] == 241
    moderate_checkpoint = moderate_path.parent / "checkpoints" / "fedavg-dirichlet05-8round-seed241_final.pt"
    assert hashlib.sha256(moderate_checkpoint.read_bytes()).hexdigest() == moderate["final_checkpoint"]["sha256"]
    moderate_data = root / "data/training/extended/dirichlet-seed241"
    severe_data = root / "data/training/ablations/non-iid-241"
    assert hashlib.sha256((moderate_data / "manifest.json").read_bytes()).hexdigest() == moderate["manifest_sha256"]
    assert moderate["rounds"][-1]["public_test"]["num-examples"] == training["rounds"][-1]["public_test"]["num-examples"]
    with __import__('numpy').load(moderate_data / "evaluator_only/test.npz") as a, __import__('numpy').load(severe_data / "evaluator_only/test.npz") as b:
        assert __import__('numpy').array_equal(a["source_idx"], b["source_idx"]), "Mismatched held-out cohort"
    assert (root / "reports/figures/sprint2_partition_sensitivity.png").stat().st_size > 2000
    evidence["moderate_dirichlet_global_accuracy"] = moderate["rounds"][-1]["public_test"]["accuracy"]
    evidence["moderate_dirichlet_worst_client_accuracy"] = moderate["final_global_worst_client_validation_accuracy"]
    evidence["independent_training_seed_membership_runs"] = 3
    evidence["membership_method_evaluations"] = 9
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(verify(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
