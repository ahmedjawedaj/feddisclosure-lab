"""One-process one-client training worker for the offline torch reference experiment.

This is not a Flower participant. It verifies the local training mathematics and
process boundary while the Flower dependency/network runtime is unavailable.
"""
import argparse
import json
import os
import hashlib
from pathlib import Path
import time
import torch
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP, evaluate_local, train_local, model_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--site-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--weights-in", required=True)
    parser.add_argument("--weights-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proximal-mu", type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    for file_name, expected in ((args.site_file, args.train_sha256),
                                (args.validation_file, args.validation_sha256)):
        if hashlib.sha256(Path(file_name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Site-owned dataset mismatch: {file_name}")
    model = DigitMLP(args.features, args.classes)
    model.load_state_dict(torch.load(args.weights_in, map_location="cpu", weights_only=True))
    val_x, val_y = load_site_partition(args.validation_file)
    if args.eval_only:
        Path(args.metrics_out).write_text(json.dumps({
            "worker_pid": os.getpid(), "evaluation": evaluate_local(model, val_x, val_y),
        }, indent=2) + "\n")
        return
    x, y = load_site_partition(args.site_file)
    metrics = train_local(model, x, y, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, seed=args.seed, proximal_mu=args.proximal_mu)
    metrics.update({"worker_pid": os.getpid(), "local_update_validation": evaluate_local(model, val_x, val_y),
                    "elapsed_seconds": round(time.perf_counter() - start, 5),
                    "communication_upload_bytes": model_bytes(model.state_dict())})
    torch.save(model.state_dict(), args.weights_out)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
