"""Flower 1.27 ServerApp: trained weights and aggregate metrics, no client raw datasets."""
import json
from pathlib import Path
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from feddisclosure.training.task import DigitMLP

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    torch.manual_seed(int(config["seed"]))
    model = DigitMLP(int(config["features"]), int(config["classes"]))
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      min_train_nodes=int(config["clients"]),
                      min_evaluate_nodes=int(config["clients"]),
                      min_available_nodes=int(config["clients"]))
    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
                            train_config=ConfigRecord({"lr": float(config["learning-rate"])}),
                            num_rounds=int(config["num-server-rounds"]))
    out = Path(str(config["artifact-dir"]))
    out.mkdir(parents=True, exist_ok=True)
    # The server has no permission to see or import client train/validation data.
    torch.save(result.arrays.to_torch_state_dict(), out / "final_model.pt")
    def values(metrics):
        return {str(round_number): dict(record) for round_number, record in metrics.items()}
    (out / "flower_summary.json").write_text(json.dumps({
        "engine": "Flower ServerApp/ClientApp", "rounds": int(config["num-server-rounds"]),
        "clients_requested": int(config["clients"]),
        "train_metrics_aggregated": values(result.train_metrics_clientapp),
        "validation_metrics_aggregated": values(result.evaluate_metrics_clientapp),
        "caveat": "FedAvg reports weighted mean validation, not the worst individual institution. No raw samples exposed.",
    }, indent=2) + "\n")
