"""Real Flower 1.27 ClientApp (Message API). Requires optional flwr dependency.

A deployment SuperNode must mount one institution's training/validation NPZs.
Only model arrays, scalar metrics, and sample counts are returned to coordinator.
"""
from pathlib import Path
import hashlib
import json
import os
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP, evaluate_local, train_local

app = ClientApp()


def _site_files(context: Context) -> tuple[Path, Path]:
    # Deployment mode: each container mounts its own private files at /site.
    direct = os.getenv("FDL_SITE_DATA_DIR")
    if direct:
        directory = Path(direct)
    else:
        # Simulation-only: separate per-site paths, but a shared filesystem.
        index = int(context.node_config["partition-id"])
        if int(context.node_config["num-partitions"]) != int(context.run_config["clients"]):
            raise RuntimeError("Expected exactly one supervised partition per client")
        directory = Path(str(context.run_config["data-root"])) / f"institution-{index:02d}"
    train_file, val_file = directory / "train.npz", directory / "val.npz"
    if not train_file.is_file() or not val_file.is_file():
        raise FileNotFoundError("SuperNode may access only its own mounted train and val partitions")
    if direct:
        expected = (os.getenv("FDL_SITE_SHA256"), os.getenv("FDL_SITE_VAL_SHA256"))
        if not all(expected) or "FDL_SITE_ID" not in os.environ:
            raise RuntimeError("Deployment SuperNode requires site identity and per-file checksums")
    else:
        # The simulation runner stages a metadata-only manifest; no client data
        # is imported through the app by the coordinator.
        root = Path(str(context.run_config["data-root"]))
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != 2:
            raise RuntimeError("Refusing old manifest with coordinator-visible private label histograms")
        site = manifest["client_partitions"][index]
        expected = (site["sha256"], site["val_sha256"])
    for path, digest in zip((train_file, val_file), expected):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("SuperNode partition integrity verification failed")
    return train_file, val_file


def _model(msg: Message, context: Context) -> DigitMLP:
    model = DigitMLP(int(context.run_config["features"]), int(context.run_config["classes"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


@app.train()
def train(msg: Message, context: Context) -> Message:
    model = _model(msg, context)
    train_file, _ = _site_files(context)
    x, y = load_site_partition(train_file)
    # Deterministic *local* optimizer and data order; no data transfer.
    site_id = int(os.getenv("FDL_SITE_ID", context.node_config.get("partition-id", 0)))
    seed = int(context.run_config["seed"]) + site_id
    metrics = train_local(model, x, y, epochs=int(context.run_config["local-epochs"]),
                          batch_size=int(context.run_config["batch-size"]),
                          lr=float(msg.content["config"]["lr"]), seed=seed)
    reply = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    model = _model(msg, context)
    _, val_file = _site_files(context)
    x, y = load_site_partition(val_file)
    result = evaluate_local(model, x, y)
    return Message(content=RecordDict({"metrics": MetricRecord({
        "eval_acc": result["accuracy"], "eval_loss": result["loss"], "num-examples": result["num-examples"]
    })}), reply_to=msg)
