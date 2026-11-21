"""Replay a Flower run bit for bit in plain PyTorch using Flower's own aggregation arithmetic.

The reference FedAvg accumulates in float64 and casts once. Flower 1.27 accumulates float32
`array * (n_i / total)` in REPLY ORDER, which is not fixed. Floating point addition is not
associative, so the two differ in the last bit every round, and label-skew or unlucky seeds can
amplify that. This script shows the gap is exactly that: with Flower's float32 arithmetic and the
reply order that the receipts imply, every client's per-round trained-weight SHA-256 and the
final model are reproduced exactly. Nothing here relaxes the parity gate.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP, train_local


def digest(model):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def flower_aggregate(states, sizes, order):
    """flwr.serverapp.strategy.strategy_utils.aggregate_arrayrecords, replies in `order`."""
    total = sum(float(n) for n in sizes)
    out = {}
    for idx in order:
        factor = float(sizes[idx]) / total
        for k, v in states[idx].items():
            arr = v.detach().cpu().numpy() * factor
            out[k] = arr if k not in out else out[k] + arr   # the library does an in-place add of the same value
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    manifest = json.loads((a.data_dir / "manifest.json").read_text())
    att = json.loads((a.run_dir / "client_attestations.json").read_text())["sites"]
    rounds, n = int(cfg["rounds"]), len(manifest["client_partitions"])
    receipts = {int(s): v["trained_weight_sha256"] for s, v in att.items()}
    data = [load_site_partition(a.data_dir / c["file"]) for c in manifest["client_partitions"]]
    sizes = [len(y) for _, y in data]
    features, classes = manifest["num_features"], manifest["num_classes"]
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    global_state = {k: v.detach().clone() for k, v in DigitMLP(features, classes).state_dict().items()}

    def train_all(state, clients=None):
        out = {}
        for c in (range(n) if clients is None else clients):
            m = DigitMLP(features, classes)
            m.load_state_dict(state)
            train_local(m, data[c][0], data[c][1], epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
                        lr=cfg["learning_rate"], seed=cfg["seed"] + c)
            out[c] = m
        return out

    report = {"rounds": [], "all_client_hashes_match": True}
    for r in range(rounds):
        models = train_all(global_state)
        hashes = {c: digest(m) for c, m in models.items()}
        match = all(hashes[c] == receipts[c][r] for c in range(n))
        entry = {"round": r + 1, "client_hashes_match_receipts": match}
        states = [models[c].state_dict() for c in range(n)]
        if not match:
            report["all_client_hashes_match"] = False
            entry["note"] = "client training from this global model does not reproduce the receipts"
            report["rounds"].append(entry)
            break
        if r + 1 < rounds:
            candidates = []
            for order in itertools.permutations(range(n)):
                cand = flower_aggregate(states, sizes, order)
                probe = train_all(cand, clients=[0])
                if digest(probe[0]) == receipts[0][r + 1]:
                    candidates.append((order, cand))
            entry["reply_orders_consistent_with_next_round_receipt"] = [list(o) for o, _ in candidates]
            if not candidates:
                report["all_client_hashes_match"] = False
                entry["note"] = "no reply order reproduces the next round receipt"
                report["rounds"].append(entry)
                break
            global_state = candidates[0][1]
            entry["order_used"] = list(candidates[0][0])
        else:
            final = torch.load(a.run_dir / "server" / "final_model.pt", weights_only=True)
            for order in itertools.permutations(range(n)):
                cand = flower_aggregate(states, sizes, order)
                if all(torch.equal(cand[k], final[k]) for k in final):
                    entry["final_model_bitwise_identical_for_order"] = list(order)
                    break
            else:
                entry["final_model_bitwise_identical_for_order"] = None
                report["all_client_hashes_match"] = False
        report["rounds"].append(entry)
        print(entry, flush=True)
    report["bitwise_reproduction_of_flower_run"] = bool(
        report["all_client_hashes_match"] and report["rounds"]
        and report["rounds"][-1].get("final_model_bitwise_identical_for_order") is not None)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print("bitwise_reproduction_of_flower_run", report["bitwise_reproduction_of_flower_run"])


if __name__ == "__main__":
    main()
