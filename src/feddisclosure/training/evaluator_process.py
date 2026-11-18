"""Privileged PUBLIC-data evaluation and reference baselines, isolated as a process.

Never import this module into a coordinator or an attack implementation. The
coordinator invokes it as a separate process and receives scalar scores only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP, evaluate_local, train_local


def model_from_state(weights: Path, features: int, classes: int):
    model = DigitMLP(features, classes)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["test", "references"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--features", type=int, required=True)
    p.add_argument("--classes", type=int, required=True)
    p.add_argument("--config")
    args = p.parse_args()
    torch.set_num_threads(1)
    data = Path(args.data_dir)
    model = model_from_state(Path(args.weights), args.features, args.classes)
    if args.mode == "test":
        x, y = load_site_partition(data / "evaluator_only" / "test.npz")
        result = evaluate_local(model, x, y)
    else:
        if not args.config:
            raise ValueError("references mode requires --config")
        config = yaml.safe_load(Path(args.config).read_text())
        manifest = json.loads((data / "manifest.json").read_text())
        partitions = [load_site_partition(data / part["file"]) for part in manifest["client_partitions"]]
        x, y = np.concatenate([s[0] for s in partitions]), np.concatenate([s[1] for s in partitions])
        test_x, test_y = load_site_partition(data / "evaluator_only" / "test.npz")
        def build():
            return model_from_state(Path(args.weights), args.features, args.classes)
        pooled = build()
        epochs = config["rounds"] * config["local_epochs"]
        train_local(pooled, x, y, epochs=epochs, batch_size=config["batch_size"],
                    lr=config["learning_rate"], seed=config["seed"])
        local_only = []
        for i, (sx, sy) in enumerate(partitions):
            one = build()
            train_local(one, sx, sy, epochs=epochs, batch_size=config["batch_size"],
                        lr=config["learning_rate"], seed=config["seed"] + i)
            local_only.append({"client_id": i, "public_test": evaluate_local(one, test_x, test_y)})
        result = {"pooled_public_data_reference": evaluate_local(pooled, test_x, test_y),
                  "local_only_public_data_reference": local_only,
                  "disclaimer": "Privileged evaluator-only public-data comparison, not a deployed federation. Optimizer schedules differ."}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
