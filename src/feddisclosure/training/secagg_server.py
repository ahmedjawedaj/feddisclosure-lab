"""Flower upstream SecAgg+ workflow on a separate legacy-style research app.

This does not reproduce any secure-aggregation cryptographic primitive. The
three-client smoke runs only if native Flower launches the upstream workflow.
There is no individual-update capture, plaintext client metrics, or debug hook.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD

from feddisclosure.training.secagg_params import validate_secagg_params
from feddisclosure.training.task import DigitMLP

app = ServerApp()


class CountingFedAvg(FedAvg):
    """FedAvg that counts rounds in which an aggregate was actually produced."""

    aggregated_rounds = 0

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.aggregated_rounds += 1
        return parameters, metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    clients = int(cfg["clients"])
    shares = int(cfg["num-shares"])
    threshold = int(cfg["reconstruction-threshold"])
    # num-shares equal to clients (every client is in everyone's neighbourhood) is the default. A
    # smaller odd num-shares gives proper neighbourhoods and is meant for larger client counts.
    validate_secagg_params(clients, shares, threshold)
    torch.manual_seed(int(cfg["seed"]))
    model = DigitMLP(int(cfg["features"]), int(cfg["classes"]))
    parameters = ndarrays_to_parameters([v.detach().cpu().numpy() for v in model.state_dict().values()])
    strategy = CountingFedAvg(fraction_fit=1.0, min_fit_clients=clients,
                      min_available_clients=clients, fraction_evaluate=0.0,
                      initial_parameters=parameters)
    legacy = LegacyContext(context=context,
                           config=ServerConfig(num_rounds=int(cfg["num-server-rounds"])),
                           strategy=strategy)
    use_secagg = bool(cfg.get("use-secagg", True))
    if use_secagg:
        workflow = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(
            num_shares=shares, reconstruction_threshold=threshold,
            max_weight=float(cfg["max-weight"]), timeout=float(cfg["timeout"])))
    else:
        # Matched plaintext control: identical app, strategy and seeds, upstream default fit.
        workflow = DefaultWorkflow()
    workflow(grid, legacy)
    output = Path(str(cfg["artifact-dir"]))
    output.mkdir(parents=True, exist_ok=True)
    # Persist the aggregate so it can be compared numerically with plaintext FedAvg.
    names = list(model.state_dict().keys())
    arrays = legacy.state.array_records[MAIN_PARAMS_RECORD].to_numpy_ndarrays()
    if strategy.aggregated_rounds != int(cfg["num-server-rounds"]):
        # A halted secure-aggregation round (e.g. too few shares after a dropout) must fail
        # loudly. Publishing the previous round's model as the result would be a silent partial run.
        raise RuntimeError(f"Only {strategy.aggregated_rounds} of {int(cfg['num-server-rounds'])} "
                           "rounds produced an aggregate; refusing to write a final model")
    initial = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    if all(np.array_equal(a, b) for a, b in zip(arrays, initial)):
        # Every client failed (or no update was aggregated): never publish the untouched
        # initial weights as if they were a trained federated model.
        raise RuntimeError("No client update was aggregated; refusing to write a final model")
    torch.save({n: torch.from_numpy(a.copy()) for n, a in zip(names, arrays)}, output / "final_model.pt")
    (output / "flower_summary.json").write_text(json.dumps({
        "engine": "Flower legacy ServerApp (" + ("upstream SecAggPlusWorkflow" if use_secagg
                                                else "plaintext default workflow control") + ")",
        "rounds": int(cfg["num-server-rounds"]), "clients_requested": clients,
        "secure_aggregation_enabled": use_secagg, "num_shares": shares,
        "reconstruction_threshold": threshold,
        "train_metrics_aggregated": {}, "validation_metrics_aggregated": {},
        "caveat": "No client metrics are exported in SecAgg+ mode; aggregate parameters only.",
    }, indent=2) + "\n")
    (output / "secagg_workflow_returned.json").write_text(json.dumps({
        "engine": ("native_flower_upstream_SecAggPlusWorkflow" if use_secagg
                   else "native_flower_plaintext_default_workflow_control"),
        "secure_aggregation_enabled": use_secagg,
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
