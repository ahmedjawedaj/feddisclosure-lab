"""Flower upstream SecAgg+ workflow on a separate legacy-style research app.

This does not reproduce any secure-aggregation cryptographic primitive. The
three-client smoke runs only if native Flower launches the upstream workflow.
There is no individual-update capture, plaintext client metrics, or debug hook.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from feddisclosure.training.task import DigitMLP

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    clients = int(cfg["clients"])
    shares = int(cfg["num-shares"])
    threshold = int(cfg["reconstruction-threshold"])
    if clients < 3 or shares != clients or not 2 <= threshold <= shares:
        raise ValueError("SecAgg+ requires >=3 clients and 2 <= threshold <= num-shares=clients")
    torch.manual_seed(int(cfg["seed"]))
    model = DigitMLP(int(cfg["features"]), int(cfg["classes"]))
    parameters = ndarrays_to_parameters([v.detach().cpu().numpy() for v in model.state_dict().values()])
    strategy = FedAvg(fraction_fit=1.0, min_fit_clients=clients,
                      min_available_clients=clients, fraction_evaluate=0.0,
                      initial_parameters=parameters)
    legacy = LegacyContext(context=context,
                           config=ServerConfig(num_rounds=int(cfg["num-server-rounds"])),
                           strategy=strategy)
    workflow = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(
        num_shares=shares, reconstruction_threshold=threshold,
        max_weight=float(cfg["max-weight"]), timeout=float(cfg["timeout"])))
    workflow(grid, legacy)
    output = Path(str(cfg["artifact-dir"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "secagg_workflow_returned.json").write_text(json.dumps({
        "engine": "native_flower_upstream_SecAggPlusWorkflow",
        "workflow_returned_normally": True,
        "clients_requested": clients,
        "num_shares": shares,
        "reconstruction_threshold": threshold,
        "rounds_requested": int(cfg["num-server-rounds"]),
        "limitations": [
            "Native execution requires Flower runtime and independent CI evidence",
            "Successful workflow return alone does not validate cryptography or prove dropout resilience",
            "With all-but-one collusion, aggregate-only confidentiality is not guaranteed",
            "Simulation on one host is not separate-institution deployment",
        ],
    }, indent=2) + "\n")
