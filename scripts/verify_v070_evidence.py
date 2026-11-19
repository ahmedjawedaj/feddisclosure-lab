"""Read-only integrity and scientific-contract validation of saved v0.7 evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from run_balanced_heterogeneity import PARTITIONS, aggregate, summary_for_run


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, *, expect_seeds: tuple[int, ...] = (241, 242, 243, 244, 245)) -> dict:
    folder = root / "reports/results/sprint5/balanced_heterogeneity"
    summary = json.loads((folder / "summary.json").read_text())
    seeds = list(expect_seeds)
    if summary["seeds_requested"] != seeds or summary["aggregate"]["completed_seeds"] != seeds:
        raise AssertionError("Required study seeds missing")
    if summary["partitions_requested"] != list(PARTITIONS) or summary["rounds"] != 4:
        raise AssertionError("Study protocol mismatch")
    assert len(summary["results"]) == 4 * len(seeds)
    for seed in seeds:
        cohort = set()
        client_budgets = set()
        communication = set()
        for partition in PARTITIONS:
            row = next(r for r in summary["results"] if r["seed"] == seed and r["partition"] == partition)
            m = row["metrics"]
            data_dir = folder / "data" / f"seed-{seed}" / partition
            manifest = json.loads((data_dir / "manifest.json").read_text())
            if digest(data_dir / "manifest.json") != m["manifest_sha256"]:
                raise AssertionError("Tampered manifest")
            if manifest["schema_version"] != 2 or manifest["seed"] != seed or manifest["partition"] != partition:
                raise AssertionError("Manifest mismatch")
            assert all("label_histogram" not in r for r in manifest["client_partitions"])
            with np.load(data_dir / "evaluator_only/test.npz", allow_pickle=False) as d:
                cohort.add(hashlib.sha256(d["source_idx"].tobytes()).hexdigest())
            sizes = tuple(x["samples"] for x in manifest["client_partitions"])
            val_sizes = tuple(x["val_samples"] for x in manifest["client_partitions"])
            assert sizes == (192, 192, 192) and val_sizes == (48, 48, 48)
            ids = set()
            for site in manifest["client_partitions"]:
                for path_key, hash_key in (("file", "sha256"), ("val_file", "val_sha256")):
                    data_path = data_dir / site[path_key]
                    assert digest(data_path) == site[hash_key]
                    with np.load(data_path, allow_pickle=False) as d:
                        these = set(d["source_idx"].tolist())
                    assert ids.isdisjoint(these)
                    ids.update(these)
            assert len(ids) == 720
            json_file = folder / "runs" / f"balanced-{partition}-seed{seed}-r4.json"
            observed = json.loads(json_file.read_text())
            assert observed["engine"].endswith("NOT_Flower")
            assert observed["manifest_sha256"] == m["manifest_sha256"]
            assert len(observed["rounds"]) == 4
            ckpt = folder / "runs/checkpoints" / f"balanced-{partition}-seed{seed}-r4_final.pt"
            assert digest(ckpt) == m["checkpoint_sha256"] == observed["final_checkpoint"]["sha256"]
            recalculated = summary_for_run(observed, manifest, data_dir)
            for key in ("test_accuracy", "worst_client_accuracy", "mean_client_label_tv", "total_communication_bytes"):
                assert math.isclose(recalculated[key], m[key]) if key in m else math.isclose(recalculated[key], m.get(key, -1))
            client_budgets.add(tuple(m["local_optimizer_batches_per_round_per_client"]))
            communication.add(m["total_communication_bytes"])
        assert len(cohort) == 1, "Different test cohorts within matched seed"
        assert client_budgets == {(12, 12, 12)}
        assert len(communication) == 1
    recalculated_aggregate = aggregate(summary["results"], seeds, list(PARTITIONS))
    assert recalculated_aggregate == summary["aggregate"]
    figure = root / "reports/figures/sprint5_balanced_heterogeneity.png"
    assert figure.stat().st_size > 15000
    return {"seeds": seeds, "trained_runs": 4 * len(seeds),
            "uniform_training_examples_per_client": 192,
            "uniform_validation_examples_per_client": 48,
            "uniform_local_batches_per_round": 12,
            "global_accuracy_by_partition": {p: round(summary["aggregate"]["by_partition"][p]["accuracy"]["mean"], 6) for p in PARTITIONS},
            "worst_client_by_partition": {p: round(summary["aggregate"]["by_partition"][p]["worst_client_accuracy"]["mean"], 6) for p in PARTITIONS},
            "data_hashes_and_checkpoints": "verified", "plot": "verified"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
