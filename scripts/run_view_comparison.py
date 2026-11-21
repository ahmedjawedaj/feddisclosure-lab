"""Protected vs unprotected update-inversion comparison (see feddisclosure.attacks.view_study)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from feddisclosure.attacks.view_study import (attack, chance_floor, client_step, flatten, score_victim,
                                               secagg_aggregate, unflatten)
from feddisclosure.training.datasets import _source
from feddisclosure.training.task import DigitMLP

ROOT = Path(__file__).resolve().parents[1]
LR, B, N = 0.15, 8, 4


def single_step_epsilon(sigma: float, delta: float = 1e-5) -> float:
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant(); acc.history = [(sigma, 1.0, 1)]
    return float(acc.get_epsilon(delta))


def one_seed(seed: int, sigmas, max_weights, iterations: int) -> dict:
    torch.set_num_threads(1)
    x, y, _ = _source("sklearn_digits", ROOT / "data/training/_lira_cache")
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(y)); idx = perm[: N * B]
    cx, cy = x[idx].reshape(N, B, 64), y[idx].reshape(N, B)          # client 0 is the victim
    torch.manual_seed(seed); init = {k: v.clone() for k, v in DigitMLP(64, 10).state_dict().items()}
    sizes = [B] * N
    victim_x = cx[0]; all_x = cx.reshape(-1, 64); all_y = cy.reshape(-1)
    out = {"seed": seed, "floor": chance_floor(victim_x, x[perm[N * B:]], rng), "views": {}}

    def delta_of(state): return {k: init[k] - state[k] for k in init}

    def run(name, observer, delta, labels, truth_x, victim_rows):
        t = time.perf_counter()
        rec = attack(init, delta, labels, observer, lr=LR, iterations=iterations, seed=seed + 1)
        out["views"][name] = {**score_victim(truth_x, rec, victim_rows), "seconds": round(time.perf_counter() - t, 1)}

    def build(sigma):
        return [client_step(init, cx[i], cy[i], lr=LR, sigma=sigma, seed=seed * 10 + i) for i in range(N)]

    def aggregate(states, mw):
        vecs = [flatten(s) for s in states]
        return unflatten(secagg_aggregate(vecs, sizes, max_weight=mw, rng_seed=seed), init)

    def collude(states, mw):
        agg = flatten(aggregate(states, mw)); total = sum(sizes)
        others = sum(flatten(states[j]) * sizes[j] / total for j in range(1, N))
        return unflatten((agg - others) * total / sizes[0], init)

    plain = build(None)
    run("plaintext_individual", "server_individual_model_delta", delta_of(plain[0]), cy[0], victim_x, B)
    for mw in max_weights:
        run(f"secagg_aggregate_maxw{mw:g}", "server_aggregate_model_delta_joint_oracle",
            delta_of(aggregate(plain, mw)), all_y, all_x, B)
        run(f"secagg_collusion_n-1_maxw{mw:g}", "server_collusion_isolated_delta",
            delta_of(collude(plain, mw)), cy[0], victim_x, B)
    mw = max_weights[-1]
    for s in sigmas:
        dp = build(s)
        run(f"dp{s:g}_individual", "server_individual_model_delta", delta_of(dp[0]), cy[0], victim_x, B)
        run(f"dp{s:g}_secagg_aggregate_maxw{mw:g}", "server_aggregate_model_delta_joint_oracle",
            delta_of(aggregate(dp, mw)), all_y, all_x, B)
        run(f"dp{s:g}_secagg_collusion_maxw{mw:g}", "server_collusion_isolated_delta",
            delta_of(collude(dp, mw)), cy[0], victim_x, B)
    return out


def _ci(v):
    from scipy import stats
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.2, 0.5, 1.0])
    p.add_argument("--max-weights", nargs="+", type=float, default=[1000.0, 16.0])
    p.add_argument("--output", type=Path, default=ROOT / "reports/results/privacy/v014/view_comparison")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for s in range(a.seeds):
        f = a.output / f"seed{s}.json"
        if f.is_file():
            runs.append(json.loads(f.read_text())); continue
        r = one_seed(s, a.sigmas, a.max_weights, a.iterations)
        f.write_text(json.dumps(r, indent=1)); runs.append(r); print("seed", s, "done", flush=True)
    names = list(runs[0]["views"])
    plain = np.array([r["views"]["plaintext_individual"]["victim_mse"] for r in runs])
    summary = {"schema_version": 1, "clients": N, "records_per_client": B, "seeds": a.seeds,
               "lr": LR, "attack_iterations": a.iterations, "labels_known_to_attacker": True,
               "dp": {"mechanism": "Opacus DP-SGD single step, per-sample clip 1.0",
                      "single_step_epsilon_delta_1e-5_no_subsampling": {str(s): single_step_epsilon(s) for s in a.sigmas}},
               "secagg": "upstream Flower SecAgg+ quantiser arithmetic, clipping_range 8, quantization_range 2**22",
               "floor": {k: _ci([r["floor"][k] for r in runs]) for k in runs[0]["floor"]}, "views": {}}
    for n in names:
        mse = np.array([r["views"][n]["victim_mse"] for r in runs])
        rate = np.array([r["views"][n]["victim_rate_mse_below_0_05"] for r in runs])
        summary["views"][n] = {"victim_mse": _ci(mse), "rate_mse_below_0_05": _ci(rate),
                               "paired_mse_minus_plaintext": _ci(mse - plain)}
    summary["limitations"] = [
        "Observation-model study with real quantiser arithmetic, not an attack on captured Docker traffic",
        "One local SGD step on 8 distinct records per client, known labels, known lr and initial model",
        "DP noise is Opacus single-step DP-SGD, the attacker matches naively without modelling clipping or noise",
        "A failed reconstruction is evidence about this attack only, not a privacy guarantee"]
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for n, v in summary["views"].items():
        print("%-40s mse %.4f  rate %.2f" % (n, v["victim_mse"][0], v["rate_mse_below_0_05"][0]))
    print("floor", summary["floor"])


if __name__ == "__main__":
    main()
