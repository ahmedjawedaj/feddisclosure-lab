"""Does round-1 SecAgg+ quantisation error grow through round-2 local training?

The 2-round tolerance used so far (2 x the one-round worst-case bound) assumed the training map
is not expansive. This probe tests that assumption on saved evidence, with plaintext PyTorch only.
It takes the REAL one-round SecAgg+ model, starts round 2 from it (and from the exact round-1
reference), runs the same round-2 training on both, and reports the growth of the difference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from verify_secagg_native import maxdiff, per_round_bound, simulate  # noqa: E402
from verify_secagg_dropout_matrix import CFG  # noqa: E402
from feddisclosure.training.datasets import load_site_partition  # noqa: E402
from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local  # noqa: E402


def round2_from(state, d, m, clients):
    g = DigitMLP(m["num_features"], m["num_classes"]); g.load_state_dict(state)
    states, sizes = [], []
    for c in clients:
        x, y = load_site_partition(d / m["client_partitions"][c]["file"])
        l = DigitMLP(m["num_features"], m["num_classes"]); l.load_state_dict(g.state_dict())
        train_local(l, x, y, epochs=CFG["local_epochs"], batch_size=CFG["batch_size"], lr=CFG["learning_rate"], seed=CFG["seed"] + c)
        states.append({k: v.detach().clone() for k, v in l.state_dict().items()}); sizes.append(len(y))
    return fedavg_weighted(states, sizes)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, default=ROOT / "reports/results/secagg/v014_dropout_matrix/error_propagation.json")
    a = a.parse_args()
    d = ROOT / "data/training/digits-docker-iid-4c-v014"; m = json.loads((d / "manifest.json").read_text())
    sizes = [p["samples"] for p in m["client_partitions"]]; b1 = per_round_bound(4, sum(sizes), 1000.0)
    V = ROOT / "reports/results/secagg/v014_dropout_matrix"
    one = torch.load(V / "n4-1round-secagg/server/final_model.pt", weights_only=True)
    ref1, _ = simulate(d, m, CFG, [(0, 1, 2, 3)])
    ref2, _ = simulate(d, m, CFG, [(0, 1, 2, 3)] * 2)
    start_diff = maxdiff(one, ref1)
    after = round2_from(one, d, m, range(4))           # round 2 plaintext, started from the real SecAgg round-1 model
    grown = maxdiff(after, ref2)                         # difference after one more round of exact aggregation
    base = torch.load(ROOT / "reports/results/secagg/v014_n4/n4-t2-baseline/server/final_model.pt", weights_only=True)
    out = {"one_round_bound": b1, "round1_secagg_vs_reference": start_diff,
           "round1_error_after_one_exact_round2": grown, "growth_factor": grown / start_diff,
           "two_round_secagg_vs_reference_observed": maxdiff(base, ref2),
           "naive_two_round_tolerance_2x_bound": 2 * b1,
           "conclusion_rule": "growth_factor > 1 means the training map expands quantisation error, so 2 x one-round bound is not a valid multi-round bound"}
    out["relu_flip_study"] = relu_flip_study()
    args = a.output; args.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2))



def relu_flip_study(trials: int = 200, seed: int = 7):
    """Inject quantisation-sized noise (within the one-round bound) into the exact round-1 model, run exact round 2,
    and report how often the round-2 difference exceeds 2 x the one-round bound. Shows whether non-smooth
    training (ReLU activation flips) can turn bounded round-1 error into a larger round-2 difference."""
    import numpy as np
    d = ROOT / "data/training/digits-docker-iid-4c-v014"; m = json.loads((d / "manifest.json").read_text())
    sizes = [p["samples"] for p in m["client_partitions"]]; total = sum(sizes)
    step_eff = 2 * 8.0 / 4194304 * 1000.0 / total          # per-client worst-case error after server rescaling
    ref1, _ = simulate(d, m, CFG, [(0, 1, 2, 3)]); ref2 = round2_from(ref1, d, m, range(4))
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(trials):
        noisy = {k: v + torch.tensor(sum(rng.uniform(-1, 1, tuple(v.shape)) for _ in range(4)) * step_eff, dtype=v.dtype)
                 for k, v in ref1.items()}
        diffs.append(maxdiff(round2_from(noisy, d, m, range(4)), ref2))
    b1 = per_round_bound(4, total, 1000.0); a = np.array(diffs)
    return {"trials": trials, "one_round_bound": b1, "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "fraction_above_2x_bound": float((a > 2 * b1).mean()),
            "fraction_above_observed_baseline_1.89e-4": float((a > 1.8877e-4).mean())}


if __name__ == "__main__":
    main()
