"""Independent verification of the upstream Flower SecAgg+ Docker runs.

Nothing here trusts runner flags alone. For each scenario it re-reads the saved evidence,
independently recomputes honest client updates with plain PyTorch (same seeds/data), and
checks (1) numerical equivalence of the SecAgg+ aggregate with plaintext FedAvg within the
protocol's fixed-point quantization error, (2) which client subsets a dropout aggregate
matches, and (3) what colluding clients can recover from the aggregate (an inherent limit,
not a protocol failure).
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local

QUANT_STEP = 2 * 8.0 / 4194304  # default clipping_range=8, quantization_range=2**22


def per_round_bound(n_clients: int, total_samples: int, max_weight: float) -> float:
    """Worst-case |SecAgg+ - exact weighted mean| per round, derived from upstream source.

    Each client multiplies its update by n_i/max_weight, then stochastically rounds it onto a
    grid of size QUANT_STEP (error < 1 step per element). The server divides the sum by
    sum(n_i/max_weight) = total_samples/max_weight, amplifying the summed errors.
    """
    return n_clients * QUANT_STEP * max_weight / total_samples


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def flat(state):
    return torch.cat([v.reshape(-1).double() for v in state.values()])


def maxdiff(a, b):
    return float((flat(a) - flat(b)).abs().max())


def simulate(data_dir, manifest, cfg, subsets):
    """Plain-PyTorch weighted FedAvg where round r aggregates only clients in subsets[r]."""
    torch.manual_seed(cfg["seed"])
    global_model = DigitMLP(manifest["num_features"], manifest["num_classes"])
    updates_by_round = []
    for subset in subsets:
        states, sizes, upd = [], [], {}
        for c in subset:
            x, y = load_site_partition(data_dir / manifest["client_partitions"][c]["file"])
            local = DigitMLP(manifest["num_features"], manifest["num_classes"])
            local.load_state_dict(global_model.state_dict())
            train_local(local, x, y, epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
                        lr=cfg["learning_rate"], seed=cfg["seed"] + c)
            st = {k: v.detach().clone() for k, v in local.state_dict().items()}
            states.append(st); sizes.append(len(y)); upd[c] = (st, len(y))
        global_model.load_state_dict(fedavg_weighted(states, sizes))
        updates_by_round.append(upd)
    return {k: v.detach().clone() for k, v in global_model.state_dict().items()}, updates_by_round


def scenario_flags(run):
    rec = load(run / "server" / "flower_run.json") or {}
    topo = load(run / "docker_topology.json") or {}
    marker = load(run / "server" / "secagg_workflow_returned.json") or {}
    log = (run / "flower_cli_stdout.log").read_text() if (run / "flower_cli_stdout.log").is_file() else ""
    results = [l.split("aggregate_fit:")[1].strip() for l in log.splitlines() if "aggregate_fit:" in l]
    return {"runner_native_flower_verified": rec.get("native_flower_verified"),
            "docker_topology_native_verified": topo.get("native_verified"),
            "client_receipts_verified": rec.get("client_attestation_verified"),
            "flower_exit_code": rec.get("exit_code"), "timed_out": rec.get("timed_out"),
            "seconds": rec.get("seconds"), "fault_injection": rec.get("fault_injection"),
            "workflow_engine": marker.get("engine"), "secure_aggregation_enabled": marker.get("secure_aggregation_enabled"),
            "num_shares": marker.get("num_shares"), "reconstruction_threshold": marker.get("reconstruction_threshold"),
            "secure_aggregation_completed_lines": log.count("Secure aggregation completed"),
            "aggregate_fit_lines": results, "final_model_exists": (run / "server" / "final_model.pt").is_file()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/training/digits-docker-iid-v012")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.data_dir / "manifest.json").read_text())
    cfg = {"seed": 20261118, "local_epochs": 1, "batch_size": 32, "learning_rate": 0.15}
    report = {"schema_version": 1, "quantization_step_per_update": QUANT_STEP, "scenarios": {}}
    total = sum(p_["samples"] for p_ in manifest["client_partitions"])
    bound1 = per_round_bound(3, total, 1000.0)
    report["tolerance_derivation"] = {
        "formula": "n_clients * quantization_step * max_weight / total_samples (per round, worst case)",
        "n_clients": 3, "total_samples": total, "max_weight": 1000.0,
        "per_round_bound": bound1,
        "note": "The first a-priori tolerance (2e-5) ignored the max_weight/total_samples amplification "
                "and was too tight; it is superseded by this derivation from the upstream source."}
    tol = 2 * bound1  # two rounds; training map is not expansive at lr=0.15 on this model
    on, ctl = a.base / "secagg-on-t2-n3", a.base / "plaintext-control"
    if (on / "server/final_model.pt").is_file() and (ctl / "server/final_model.pt").is_file():
        s_on = torch.load(on / "server/final_model.pt", weights_only=True)
        s_ctl = torch.load(ctl / "server/final_model.pt", weights_only=True)
        ref, _ = simulate(a.data_dir, manifest, cfg, [(0, 1, 2)] * 2)
        report["equivalence_2_rounds"] = {
            "secagg_vs_plaintext_control_max_abs_diff": maxdiff(s_on, s_ctl),
            "secagg_vs_independent_pytorch_max_abs_diff": maxdiff(s_on, ref),
            "plaintext_control_vs_independent_pytorch_max_abs_diff": maxdiff(s_ctl, ref),
            "tolerance": tol,
            "within_tolerance": maxdiff(s_on, ref) <= tol and maxdiff(s_on, s_ctl) <= tol}
    one_r = a.base / "secagg-1round-collusion" / "server" / "final_model.pt"
    mw = a.base / "secagg-1round-maxweight200" / "server" / "final_model.pt"
    if one_r.is_file():
        ref1, _ = simulate(a.data_dir, manifest, cfg, [(0, 1, 2)])
        entry = {"max_weight_1000": {"observed_max_abs_diff": maxdiff(torch.load(one_r, weights_only=True), ref1),
                                     "analytic_bound": bound1}}
        if mw.is_file():
            b200 = per_round_bound(3, total, 200.0)
            entry["max_weight_200"] = {"observed_max_abs_diff": maxdiff(torch.load(mw, weights_only=True), ref1),
                                       "analytic_bound": b200}
        for v in entry.values():
            v["within_bound"] = v["observed_max_abs_diff"] <= v["analytic_bound"]
        report["single_round_error_vs_bound"] = entry
    for name in ("secagg-on-t2-n3", "plaintext-control", "secagg-1round-collusion", "secagg-1round-maxweight200",
                 "secagg-dropout-before-run", "secagg-dropout-mid-protocol"):
        if (a.base / name / "server" / "flower_run.json").is_file():
            report["scenarios"][name] = scenario_flags(a.base / name)
    # dropout: which client subsets does the aggregate match?
    for name in ("secagg-dropout-before-run", "secagg-dropout-mid-protocol"):
        mp = a.base / name / "server" / "final_model.pt"
        if mp.is_file():
            agg = torch.load(mp, weights_only=True)
            subsets = [s for k in (2, 3) for s in itertools.combinations(range(3), k)]
            matches = []
            for s1, s2 in itertools.product(subsets, subsets):
                sim, _ = simulate(a.data_dir, manifest, cfg, [s1, s2])
                d = maxdiff(agg, sim)
                matches.append({"round1_clients": list(s1), "round2_clients": list(s2), "max_abs_diff": d})
            matches.sort(key=lambda m: m["max_abs_diff"])
            report["scenarios"][name]["best_matching_participation"] = matches[0]
            report["scenarios"][name]["matches_a_participation_within_tolerance"] = matches[0]["max_abs_diff"] <= tol
    # collusion limit on the single-round run
    one = a.base / "secagg-1round-collusion" / "server" / "final_model.pt"
    if one.is_file():
        agg = torch.load(one, weights_only=True)
        sim_full, ups = simulate(a.data_dir, manifest, {**cfg}, [(0, 1, 2)])
        u = ups[0]
        total = sum(n for _, n in u.values())
        victim = 2
        known = {c: u[c] for c in (0, 1)}
        known_sum = sum((flat(st) * n for st, n in known.values()), torch.zeros_like(flat(agg)))
        recovered = (flat(agg) * total - known_sum) / u[victim][1]
        truth = flat(u[victim][0])
        one_known = (flat(agg) * total - flat(u[0][0]) * u[0][1])
        est_one = one_known / (u[1][1] + u[2][1])
        report["collusion_single_round"] = {
            "victim": victim,
            "aggregate_vs_independent_pytorch_max_abs_diff": maxdiff(agg, sim_full),
            "baseline_error_if_victim_assumed_equal_to_aggregate": float((flat(agg) - truth).abs().max()),
            "error_with_server_plus_1_colluder": float((est_one - truth).abs().max()),
            "error_with_server_plus_2_colluders_n_minus_1": float((recovered - truth).abs().max()),
            "interpretation": "With n-1 colluders the victim update is recovered to quantization precision "
                              "regardless of threshold: aggregate-only protection is inherently void under "
                              "n-1 collusion. This is a limit of secure aggregation, not a Flower defect."}
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "scenarios"}, indent=2))
    for k, v in report["scenarios"].items():
        print(k, v["runner_native_flower_verified"], v["flower_exit_code"], v.get("best_matching_participation"))


if __name__ == "__main__":
    main()
