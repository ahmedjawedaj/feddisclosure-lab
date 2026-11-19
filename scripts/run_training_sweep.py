"""Paired-seed federated-training reference sweep.

The two partition conditions share the same public source sample/test partition
for each seed. Every client still trains in a separate OS worker process.
This script does not exercise Flower, Docker, or formal privacy mechanisms.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time
import yaml
from feddisclosure.training.offline_runner import run_reference


def summary(runs):
    output = {"engine": "torch_independent_process_reference_NOT_Flower",
              "comparison": "paired source/test seeds; IID vs severe label-skew; smoke not a significance claim",
              "runs_completed": len(runs), "conditions": {}, "paired_differences": []}
    for partition in ("iid", "label_skew"):
        found = [r for r in runs if r["partition"] == partition]
        if found:
            vals = [r["public_test_accuracy"] for r in found]
            worst = [r["worst_client_accuracy"] for r in found]
            output["conditions"][partition] = {
                "n": len(found), "public_accuracy_mean": statistics.mean(vals),
                "public_accuracy_std": statistics.stdev(vals) if len(vals) > 1 else None,
                "worst_client_accuracy_mean": statistics.mean(worst),
                "worst_client_accuracy_std": statistics.stdev(worst) if len(worst) > 1 else None,
                "individual_results": found}
    for iid in (r for r in runs if r["partition"] == "iid"):
        skew = next((r for r in runs if r["partition"] == "label_skew" and r["seed"] == iid["seed"] and r["clients"] == iid["clients"]), None)
        if skew:
            output["paired_differences"].append({"seed": iid["seed"], "clients": iid["clients"],
                                                 "iid_minus_skew_accuracy": iid["public_test_accuracy"] - skew["public_test_accuracy"],
                                                 "iid_minus_skew_worst_client_accuracy": iid["worst_client_accuracy"] - skew["worst_client_accuracy"]})
    return output


def saved_sweep_runs(output: Path, *, clients: int, rounds: int, source_limit: int) -> list[dict]:
    """Reconstruct the summary from all matching persisted runs, including resumes."""
    records = []
    for path in sorted((output / "runs").glob("*.json")):
        result = json.loads(path.read_text())
        cfg = result["config"]
        if cfg["clients"] != clients or cfg["rounds"] != rounds or cfg.get("source_limit") != source_limit:
            continue
        records.append({"seed": cfg["seed"], "partition": cfg["partition"], "clients": clients,
                        "rounds": rounds, "public_test_accuracy": result["rounds"][-1]["public_test"]["accuracy"],
                        "worst_client_accuracy": result["final_global_worst_client_validation_accuracy"],
                        "model_payload_bytes_total": sum(r["communication_model_bytes"] for r in result["rounds"]),
                        "elapsed_seconds": result["total_elapsed_seconds"],
                        "result": str(path.relative_to(output)),
                        "checkpoint_sha256": result["final_checkpoint"]["sha256"]})
    return records


def run_sweep(output: Path, *, seeds: list[int], clients: int = 3, rounds: int = 2,
              source_limit: int = 600, partitions: tuple[str, ...] = ("iid", "label_skew")) -> dict:
    if not seeds or clients < 1 or rounds < 1 or source_limit < clients * 20:
        raise ValueError("Invalid sweep conditions")
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        for partition in partitions:
            key = f"c{clients}_{partition}_seed{seed}_r{rounds}"
            config = {"id": f"sweep-{key}", "seed": seed, "dataset": "sklearn_digits",
                      "partition": partition, "clients": clients, "rounds": rounds,
                      "local_epochs": 1, "batch_size": 32, "learning_rate": 0.15,
                      "source_limit": source_limit, "observer": "honest_but_curious_coordinator"}
            cfg = output / "configs" / f"{key}.yaml"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(yaml.safe_dump(config, sort_keys=False))
            result_file = output / "runs" / f"{key}.json"
            if result_file.exists():
                result = json.loads(result_file.read_text())
                if result["config"] != config:
                    raise RuntimeError("Refusing to reuse result for a different configuration")
            else:
                result = run_reference(cfg, output / "prepared_public_digits" / key, result_file)
            runs = saved_sweep_runs(output, clients=clients, rounds=rounds, source_limit=source_limit)
            report = summary(runs)
            (output / "sweep_summary.json").write_text(json.dumps(report, indent=2) + "\n")
            current = next(r for r in runs if r["result"] == str(result_file.relative_to(output)))
            print(f"Completed {key}: accuracy={current['public_test_accuracy']:.4f}, worst={current['worst_client_accuracy']:.4f}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/results/sprint2/sweep")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--source-limit", type=int, default=600)
    p.add_argument("--partition", choices=["iid", "label_skew", "both"], default="both")
    args = p.parse_args()
    partition = ("iid", "label_skew") if args.partition == "both" else (args.partition,)
    run_sweep(Path(args.output), seeds=args.seeds, clients=args.clients,
              rounds=args.rounds, source_limit=args.source_limit, partitions=partition)


if __name__ == "__main__":
    main()
