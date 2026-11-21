"""How much does a ~1 ULP perturbation move the final federated model?

Used to interpret a Flower-vs-reference parity gap. It re-runs the plain-PyTorch weighted FedAvg
(same functions as the reference) twice: unperturbed, and with the initial weights perturbed by
`--ulp-scale` x float32 spacing. If a perturbation far below the parity tolerance already moves the
final weights by more than the tolerance, a larger gap is not evidence of a logic error, but it also
means weight parity alone cannot certify such a run. The check is reported, never used to relax gates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local


def run(data_dir, manifest, cfg, rounds, perturb_ulps):
    torch.manual_seed(cfg["seed"])
    model = DigitMLP(manifest["num_features"], manifest["num_classes"])
    if perturb_ulps:
        g = torch.Generator().manual_seed(12345)
        with torch.no_grad():
            for p in model.parameters():
                spacing = torch.nextafter(p.abs(), torch.tensor(float("inf"))) - p.abs()
                p.add_(torch.randn(p.shape, generator=g).sign() * spacing * perturb_ulps)
    data = [load_site_partition(data_dir / c["file"]) for c in manifest["client_partitions"]]
    for _ in range(rounds):
        states, sizes = [], []
        for c, (x, y) in enumerate(data):
            local = DigitMLP(manifest["num_features"], manifest["num_classes"])
            local.load_state_dict(model.state_dict())
            train_local(local, x, y, epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
                        lr=cfg["learning_rate"], seed=cfg["seed"] + c)
            states.append({k: v.detach().clone() for k, v in local.state_dict().items()})
            sizes.append(len(y))
        model.load_state_dict(fedavg_weighted(states, sizes))
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--reference-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ulps", type=float, default=1.0)
    a = p.parse_args()
    import yaml
    cfg = yaml.safe_load(a.config.read_text())
    cfg = {"seed": cfg["seed"], "local_epochs": cfg["local_epochs"], "batch_size": cfg["batch_size"],
           "learning_rate": cfg["learning_rate"]}
    torch.set_num_threads(1)
    manifest = json.loads((a.data_dir / "manifest.json").read_text())
    rounds = yaml.safe_load(a.config.read_text())["rounds"]
    base = run(a.data_dir, manifest, cfg, rounds, 0)
    pert = run(a.data_dir, manifest, cfg, rounds, a.ulps)
    ref = torch.load(a.reference_checkpoint, weights_only=True)
    d = lambda x, y: max(float((x[k].double() - y[k].double()).abs().max()) for k in x)
    out = {"perturbation_ulps_of_initial_weights": a.ulps, "rounds": rounds,
           "baseline_vs_saved_reference_max_abs_diff": d(base, ref),
           "perturbed_vs_baseline_max_abs_diff": d(pert, base),
           "parity_tolerance_atol": 1e-5,
           "interpretation": "perturbation of ~1 float32 ULP in the initial weights moves the final model by "
                             "this much, so numerical parity at 1e-5 is not attainable for this setting"
                             if d(pert, base) > 1e-5 else
                             "run is numerically stable under ULP-level perturbation, so a larger parity "
                             "gap would indicate a real pipeline difference"}
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
