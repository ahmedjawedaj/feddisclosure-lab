"""Paired FedAvg/FedProx reference study; observations are NOT Flower runs.

Each seed uses one immutable partition, shared global initialization, fixed
optimizer budget and evaluation. Scalar evidence is saved per finished run
so a timeout can be safely resumed; checksum validation refuses stale data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import yaml
from feddisclosure.training.datasets import prepare_partitions

ROOT = Path(__file__).resolve().parents[1]


def write_summary(out, mu, rounds, rows):
    def mean_sd(values):
        return {"mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None}
    summary = {"engine": "torch_independent_process_reference_NOT_Flower",
               "dataset": "sklearn_digits", "partition": "label_skew",
               "matched_fields": ["seed", "data_partitions", "initial_model", "rounds",
                                  "local_epochs", "batch_size", "learning_rate", "communication_accounting"],
               "different_fields": ["algorithm", "proximal_mu"],
               "mu": mu, "rounds": rounds, "completed_seeds": len(rows),
               "results": rows,
               "fedavg_accuracy": mean_sd([r["fedavg"]["last_test_acc"] for r in rows]),
               "fedprox_accuracy": mean_sd([r["fedprox"]["last_test_acc"] for r in rows]),
               "paired_difference_pp": mean_sd([r["fedprox_minus_fedavg_pp"] for r in rows]),
               "limitations": ["Short four-round runs on approved public 8x8 digits; no MNIST",
                               "The distribution is severely class-sorted; worst-client failure is possible",
                               "PyTorch separate-process reference, NOT native Flower or OS data isolation",
                               "Individual server-observed updates not protected by DP or secure aggregation",
                               "Exploratory three-seed comparison, no algorithm superiority claim"]}
    (out / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run(seeds, rounds, mu, out):
    if len(set(seeds)) != len(seeds) or mu <= 0 or rounds < 1:
        raise ValueError("Require unique seeds, positive mu, positive round count")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        data_dir = ROOT / "data" / "training" / "ablations" / f"non-iid-{seed}"
        if not (data_dir / "manifest.json").exists():
            prepare_partitions(data_dir, dataset="sklearn_digits", clients=3,
                               partition="label_skew", seed=seed, limit=900)
        manifest_path = data_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[k] != value for k, value in [
                ("seed", seed), ("source", "sklearn_digits"),
                ("partition", "label_skew"), ("clients", 3), ("source_limit", 900)]):
            raise RuntimeError("Existing immutable data manifest is incompatible")
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        row = {"seed": seed, "manifest_source": manifest["provenance"]}
        for method, proximal_mu in (("fedavg", 0.0), ("fedprox", mu)):
            config = {"id": f"{method}-non-iid-{seed}-r{rounds}", "seed": seed,
                      "dataset": "sklearn_digits", "partition": "label_skew", "clients": 3,
                      "rounds": rounds, "local_epochs": 2, "batch_size": 32,
                      "learning_rate": 0.15, "source_limit": 900,
                      "algorithm": method, "proximal_mu": proximal_mu,
                      "observer": "honest_but_curious_coordinator"}
            config_path = out / "configs" / f"{method}-{seed}.yaml"
            config_path.parent.mkdir(exist_ok=True)
            if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
                raise RuntimeError("Refusing config mismatch on resumed run")
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            artifact = out / f"{method}-{seed}.json"
            if not artifact.exists():
                env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                subprocess.run([sys.executable, str(ROOT / "scripts/run_torch_reference.py"),
                                "--config", str(config_path), "--data-dir", str(data_dir),
                                "--output", str(artifact)], check=True, cwd=ROOT,
                               env=env, timeout=180)
            result = json.loads(artifact.read_text())
            if result.get("config") != config or len(result.get("rounds", [])) != rounds:
                raise RuntimeError("Refusing cached run with mismatched config")
            if result.get("engine") != "torch_independent_process_reference_NOT_Flower":
                raise RuntimeError("Unexpected training engine")
            if result.get("manifest_sha256") != manifest_hash:
                raise RuntimeError("Stored result does not belong to this partition")
            checkpoint = artifact.parent / "checkpoints" / (config["id"] + "_final.pt")
            if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result["final_checkpoint"]["sha256"]:
                raise RuntimeError("Missing or modified checkpoint")
            row[method] = {"last_test_acc": result["rounds"][-1]["public_test"]["accuracy"],
                           "worst_client_acc": result["final_global_worst_client_validation_accuracy"],
                           "round_acc": [v["public_test"]["accuracy"] for v in result["rounds"]],
                           "model_bytes_total": sum(v["communication_model_bytes"] for v in result["rounds"]),
                           "checkpoint_sha256": result["final_checkpoint"]["sha256"]}
        row["fedprox_minus_fedavg_pp"] = 100 * (row["fedprox"]["last_test_acc"] - row["fedavg"]["last_test_acc"])
        rows.append(row)
        write_summary(out, mu, rounds, rows)
        print(f"seed={seed}: FedAvg={row['fedavg']['last_test_acc']:.4f}, "
              f"FedProx={row['fedprox']['last_test_acc']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[241, 242, 243])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--out-dir", default="reports/results/sprint2/ablations/paired")
    args = parser.parse_args()
    run(args.seeds, args.rounds, args.mu, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
