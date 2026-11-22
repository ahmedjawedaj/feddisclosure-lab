"""Flower 1.27 ServerApp: trained weights and aggregate metrics, no client raw datasets."""
import json
from pathlib import Path
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from feddisclosure.training.deterministic_fedavg import DETERMINISTIC, DeterministicFedAvg, resolve_aggregation_mode
from feddisclosure.training.task import DigitMLP

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    torch.manual_seed(int(config["seed"]))
    model = DigitMLP(int(config["features"]), int(config["classes"]))
    # Opt-in only. Default is the unchanged upstream FedAvg. See docs/v0.15-reply-order.md.
    mode = resolve_aggregation_mode(config)
    strategy_class = DeterministicFedAvg if mode == DETERMINISTIC else FedAvg
    strategy = strategy_class(fraction_train=1.0, fraction_evaluate=1.0,
                      min_train_nodes=int(config["clients"]),
                      min_evaluate_nodes=int(config["clients"]),
                      min_available_nodes=int(config["clients"]))
    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
                            train_config=ConfigRecord({"lr": float(config["learning-rate"]), "proximal-mu": float(config.get("proximal-mu", 0.0))}),
                            num_rounds=int(config["num-server-rounds"]))
    out = Path(str(config["artifact-dir"]))
    out.mkdir(parents=True, exist_ok=True)
    # The server has no permission to see or import client train/validation data.
    torch.save(result.arrays.to_torch_state_dict(), out / "final_model.pt")
    def values(metrics):
        return {str(round_number): dict(record) for round_number, record in metrics.items()}
    summary = {
        "engine": "Flower ServerApp/ClientApp", "rounds": int(config["num-server-rounds"]),
        "clients_requested": int(config["clients"]),
        "train_metrics_aggregated": values(result.train_metrics_clientapp),
        "validation_metrics_aggregated": values(result.evaluate_metrics_clientapp),
        "caveat": "FedAvg reports weighted mean validation, not the worst individual institution. No raw samples exposed.",
    }
    if mode == DETERMINISTIC:  # key absent in default runs so existing evidence is unchanged
        summary["aggregation"] = "DeterministicFedAvg (replies sorted by site id, float64 reference arithmetic)"
    (out / "flower_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
