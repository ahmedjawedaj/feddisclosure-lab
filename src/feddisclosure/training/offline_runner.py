"""Reference, independent-process FedAvg experiment for environments without Flower.

The parent/coordinator has only site file *paths* and model updates. All local
training data is loaded by separate worker processes. A distinct, opt-in
public-data evaluator computes reference pooled/local-only performance.
"""
from __future__ import annotations
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import torch
import yaml
from feddisclosure.training.datasets import prepare_partitions
from feddisclosure.training.task import DigitMLP, fedavg_weighted, model_bytes


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    for key in ("seed", "dataset", "partition", "clients", "rounds", "local_epochs", "batch_size", "learning_rate"):
        if key not in config:
            raise ValueError(f"Missing training config key: {key}")
    if config["clients"] < 1 or config["rounds"] < 1 or config["local_epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Invalid training dimensions")
    return config


def _call_evaluator(mode, *, data_dir, weights, output, features, classes, config_path=None):
    cmd = [sys.executable, "-m", "feddisclosure.training.evaluator_process", mode,
           "--data-dir", str(data_dir), "--weights", str(weights), "--output", str(output),
           "--features", str(features), "--classes", str(classes)]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    subprocess.run(cmd, check=True, timeout=90, capture_output=True)
    return json.loads(Path(output).read_text())


def run_reference(config_path: str | Path, data_dir: str | Path, output_path: str | Path,
                  *, include_references: bool = False) -> dict:
    config = load_config(config_path)
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = prepare_partitions(data_dir, dataset=config["dataset"], clients=config["clients"],
                                      partition=config["partition"], seed=config["seed"],
                                      limit=config.get("source_limit"))
    else:
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[k] != config[k] for k in ("partition", "seed", "clients", "source_limit") if k in manifest):
            # dataset stored as 'source' in manifest
            raise ValueError("Dataset manifest configuration mismatch")
        if manifest["source"] != config["dataset"]:
            raise ValueError("Dataset source mismatch")
    # Coordinator reads manifest metadata only, never accesses the private NPZ bytes.
    # Workers verify the hashes inside their own processes before training.
    if len(manifest["client_partitions"]) != config["clients"]:
        raise ValueError("Mismatch in number of institution partitions")
    if manifest.get("schema_version") != 2 or any("label_histogram" in p for p in manifest["client_partitions"]):
        raise ValueError("Refusing legacy manifest with coordinator-visible evaluator-only label histograms; reprepare")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    global_model = DigitMLP(manifest["num_features"], manifest["num_classes"])
    round_metrics = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fdl-fedavg-") as td:
        scratch = Path(td)
        for round_number in range(1, config["rounds"] + 1):
            incoming = scratch / f"round-{round_number}-initial.pt"
            torch.save(global_model.state_dict(), incoming)
            if round_number == 1:
                initial_weights = incoming
            replies, sizes, metrics = [], [], []
            # All three genuine processes train concurrently on independent partitions.
            pending = []
            for site in manifest["client_partitions"]:
                i = site["client_id"]
                output = scratch / f"r{round_number}-site{i}.pt"
                metric_file = scratch / f"r{round_number}-site{i}.json"
                cmd = [sys.executable, "-m", "feddisclosure.training.worker",
                       "--site-file", str(data_dir / site["file"]),
                       "--validation-file", str(data_dir / site["val_file"]),
                       "--train-sha256", site["sha256"], "--validation-sha256", site["val_sha256"],
                       "--weights-in", str(incoming),
                       "--weights-out", str(output), "--metrics-out", str(metric_file),
                       "--features", str(manifest["num_features"]), "--classes", str(manifest["num_classes"]),
                       "--epochs", str(config["local_epochs"]), "--batch-size", str(config["batch_size"]),
                       "--lr", str(config["learning_rate"]), "--seed", str(config["seed"] + i)]
                pending.append((cmd, site, output, metric_file))
            with ThreadPoolExecutor(max_workers=min(config["clients"], 3)) as pool:
                for finished in pool.map(lambda item: subprocess.run(item[0], check=True, timeout=90, capture_output=True), pending):
                    pass
            for _, site, output, metric_file in pending:
                reply = json.loads(metric_file.read_text())
                if reply["num-examples"] != site["samples"]:
                    raise RuntimeError("Client sample count mismatch")
                replies.append(torch.load(output, map_location="cpu", weights_only=True))
                sizes.append(reply["num-examples"])
                metrics.append({"client_id": site["client_id"], **reply})
            global_model.load_state_dict(fedavg_weighted(replies, sizes))
            aggregate_weights = scratch / f"r{round_number}-aggregated.pt"
            torch.save(global_model.state_dict(), aggregate_weights)
            public_eval = _call_evaluator(
                "test", data_dir=data_dir, weights=aggregate_weights,
                output=scratch / f"r{round_number}-evaluator.json",
                features=manifest["num_features"], classes=manifest["num_classes"],
            )
            round_metrics.append({"round": round_number, "public_test": public_eval,
                                  "worst_client_local_update_validation_accuracy": min(m["local_update_validation"]["accuracy"] for m in metrics),
                                  "client_metrics": metrics,
                                  "communication_model_bytes": 2 * len(replies) * model_bytes(global_model.state_dict())})
        # Validate the final aggregated model on each client's held-out set.
        # Client-only worker processes see the global weights and *own* private val.
        final_global_metrics = []
        pending = []
        for site in manifest["client_partitions"]:
            site_id = site["client_id"]
            final_weights = scratch / f"r{config['rounds']}-aggregated.pt"
            metric_path = scratch / f"global-val-site{site_id}.json"
            dummy_out = scratch / f"unused-site{site_id}.pt"
            cmd = [sys.executable, "-m", "feddisclosure.training.worker", "--eval-only",
                   "--site-file", str(data_dir / site["file"]),
                   "--validation-file", str(data_dir / site["val_file"]),
                   "--train-sha256", site["sha256"], "--validation-sha256", site["val_sha256"],
                   "--weights-in", str(final_weights), "--weights-out", str(dummy_out),
                   "--metrics-out", str(metric_path),
                   "--features", str(manifest["num_features"]), "--classes", str(manifest["num_classes"]),
                   "--epochs", str(config["local_epochs"]), "--batch-size", str(config["batch_size"]),
                   "--lr", str(config["learning_rate"]), "--seed", str(config["seed"] + site_id)]
            pending.append((cmd, site_id, metric_path))
        with ThreadPoolExecutor(max_workers=min(config["clients"], 3)) as pool:
            list(pool.map(lambda item: subprocess.run(item[0], check=True, timeout=90, capture_output=True), pending))
        for _, site_id, metric_path in pending:
            final_global_metrics.append({"client_id": site_id, **json.loads(metric_path.read_text())})
        reference_metrics = None
        if include_references:
            reference_metrics = _call_evaluator(
                "references", data_dir=data_dir, weights=initial_weights,
                output=scratch / "references.json", config_path=config_path,
                features=manifest["num_features"], classes=manifest["num_classes"],
            )
    # Persist final learned model (public digits only) for later reproducibility.
    # Checkpoint files remain gitignored by default, because future private models
    # must not be silently published with research results.
    checkpoint = Path(output_path).parent / "checkpoints" / f"{config['id']}_final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {"schema_version": 1, "engine": "torch_independent_process_reference_NOT_Flower",
              "experiment_id": config["id"], "config": config,
              "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "final_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
              "model_bytes": model_bytes(global_model.state_dict()), "rounds": round_metrics,
              "final_global_per_client_validation": final_global_metrics,
              "final_global_worst_client_validation_accuracy": min(x["evaluation"]["accuracy"] for x in final_global_metrics),
              "total_elapsed_seconds": round(time.perf_counter() - start, 4),
              "observer": config.get("observer", "honest_but_curious_coordinator"),
              "actual_observations": ["per_client_model_parameters", "sample_counts", "scalar_metrics"],
              "limitations": [
                  "Not a Flower execution: uses independent local PyTorch worker processes, no Docker isolation.",
                  "Client datasets are public digit partitions, with no real institution records.",
                  "Public evaluator test is global held-out accuracy. Final worst-client metric evaluates aggregated model on each site's held-out local validation.",
                  "Communication uses raw tensor byte counts and excludes protocol, headers and transport overhead.",
                  "Coordinator sees individual model updates; there is no DP or secure aggregation.",
                  "Evaluator runs in a separate process with public ground truth. Process separation alone does not enforce filesystem permissions.",
              ]}
    if reference_metrics is not None:
        result["references"] = reference_metrics
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
