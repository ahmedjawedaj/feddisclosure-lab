"""Reply-order study (v0.15 WP7: P1, P2, P3). Plain PyTorch replay, not a Flower run.

Flower 1.27 FedAvg sums float32 `array * weight` in the order the replies arrive. This script takes
the training setup of each saved native run (or of a prepared 4-client dataset) and replays it in
plain PyTorch, retraining every client from the aggregated model in every round. For EVERY reply
order (n! permutations) it aggregates with
  upstream       the float32 reply-order arithmetic of flwr 1.27 `aggregate_arrayrecords`
  deterministic  `DeterministicFedAvg` arithmetic (replies sorted by site id, float64 reference)
and compares with the float64 reference trajectory (`task.fedavg_weighted`, the aggregation of the
repository's torch reference) using the UNCHANGED parity gate: `torch.allclose(atol=1e-5, rtol=1e-5)`
on every tensor.

Two views per run:
  one_round   every round aggregated once from the reference trajectory's client updates, so only the
              arithmetic of that round differs (no accumulation of earlier differences)
  trajectory  the SAME permutation applied in every round, clients retrained from the resulting model.
              This is a restricted family of reply schedules (n! of the n!^rounds possible ones)
For native runs a third check replays the per-round orders implied by the receipts
(`bitwise_replay.json`) and compares with the saved native final model.

P3: does reply order move a privacy metric beyond seed noise? See `p3_study`.

    PYTHONPATH=src:. python scripts/study_reply_order.py replay --run run-iid-d
    PYTHONPATH=src:. python scripts/study_reply_order.py p3
    PYTHONPATH=src:. python scripts/study_reply_order.py summarize
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from feddisclosure.training.datasets import load_site_partition  # noqa: E402
from feddisclosure.training.deterministic_fedavg import (  # noqa: E402
    deterministic_weighted_average, sorted_by_site, upstream_weighted_average)
from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local  # noqa: E402

OUT = ROOT / "reports/results/native_flower/v0150/reply_order"
PARITY_JSON = ROOT / "reports/results/privacy/v0150/prereg_check_parity.json"
LIVE_DIR = ROOT / "reports/results/native_flower/v0150/live_parity"
NATIVE = ROOT / "reports/results/native_flower/v014"
ATOL = RTOL = 1e-5

# Native runs come from the replay gate script so the two lists cannot drift apart.
from verify_flower_replay_gate import RUNS as NATIVE_RUNS  # noqa: E402

# Replay-only runs with 4 clients. There is no saved native 4-client run, so these have no receipts.
# `primary` is the shipped 4-client config and is the one P2 is judged on. The others are exploratory.
REPLAY_ONLY = {
    "c4-digits-iid-2round": ("configs/training/fedavg_digits_iid_4clients.yaml", "data/training/digits-docker-iid-4c-v014", True),
    "c4-digits-iid-10round": ("configs/training/fedavg_digits_iid_4clients_10round.yaml", "data/training/digits-docker-iid-4c-v014", False),
    "c4-mnist-iid-10round": ("configs/training/fedavg_mnist_iid_4clients_10round.yaml", "data/training/mnist-full-iid-4c-v015", False),
}


# ---------------------------------------------------------------- annotations kept in code
# These sentences used to be added to prereg_check_parity.json by hand after a run, and `summarize` then overwrote
# them. They live here so that rerunning `summarize` on the saved inputs reproduces the committed file
# (tests/training/test_reply_order_study.py compares the two). Editing a sentence is a visible code change.
# No threshold and no result number is stored here, only wording.
NOTE = ("Thresholds are copied verbatim from the pre-registration and were not changed. P1 to P3 are judged in replay in "
        "plain PyTorch. Live hosted runs are in live_hosted_run. Wording fixes added after review R2, no threshold and no "
        "result number changed.")
P1_SCOPE = ("Judged in replay on the 7 native runs plus 3 live Digits configs (6 hosted runs, strategy on and off). The live "
            "configs do not include a config that fails upstream, because the MNIST configs that failed upstream were not run "
            "live. The strategy is the reference arithmetic by construction.")
P1_LABEL = "PASS in replay only. Live runs confirm transport and client determinism but exercise no config that fails upstream"
P2_NOTE = ("The primary config was fixed before the results were seen. Live run 45076810526 has a native 4-client Digits iid "
           "10-round run in each mode with one reply order each. It passes upstream and cannot show order dependence.")
P3_SENSITIVITY_NOTE = ("In the Gaussian perturbation control the order-effect analogue stays below the seed SD even at a "
                       "perturbation scale of 1e-2, so this metric could not exceed the seed SD on this setup.")
P3_LABEL = "PASS, metric insensitive on this setup, cannot fail"
LIVE_RUNS = ("iid3-upstream", "iid3-deterministic", "noniid3-upstream", "noniid3-deterministic",
             "iid4-10r-upstream", "iid4-10r-deterministic")
LIVE_TEXT = {
    "workflow": ".github/workflows/parity-live.yml",
    "run_id": 45076810526,
    "gate": "unchanged 1e-5 absolute and relative tolerance",
    "reading": ("All six live runs pass the unchanged gate. With the strategy on, all three configs are bit for bit equal to the "
                "reference. Upstream passes at about 1e-8. The shipped Digits configs do not trigger the reply order effect, so "
                "this live run neither confirms nor refutes P2. The MNIST configs that failed upstream in earlier native runs "
                "were not run live."),
    "earlier_attempt": ("Run 45074469691 passed both 3 client configs and failed the 4 client config in both modes at the "
                        "evidence check, because the check hardcoded 8 containers. Fixed in PR 7 before any numeric comparison."),
}


# ---------------------------------------------------------------- pure helpers (unit tested)

def all_orders(n: int) -> list[tuple[int, ...]]:
    if n < 2:
        raise ValueError("need at least two clients")
    return list(itertools.permutations(range(n)))


def state_digest(state: dict) -> str:
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode())
        h.update(torch.as_tensor(v).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gate(a: dict, b: dict, atol: float = ATOL, rtol: float = RTOL) -> tuple[bool, float]:
    """The parity gate of scripts/verify_flower_parity.py: allclose on every tensor. Also max abs diff."""
    if list(a) != list(b):
        raise ValueError("model keys differ")
    ok, worst = True, 0.0
    for k in a:
        x, y = torch.as_tensor(a[k]), torch.as_tensor(b[k])
        worst = max(worst, float((x.double() - y.double()).abs().max()))
        ok &= bool(torch.allclose(x, y, atol=atol, rtol=rtol))
    return ok, worst


def pass_rate(flags: list[bool]) -> float:
    if not flags:
        raise ValueError("no orders")
    return sum(bool(f) for f in flags) / len(flags)


def np_state(state: dict) -> dict:
    return {k: torch.as_tensor(v).detach().cpu().numpy() for k, v in state.items()}


def torch_state(arrays: dict) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).clone() for k, v in arrays.items()}


def aggregate_upstream(states: list[dict], sizes: list[int], order) -> dict:
    return torch_state(upstream_weighted_average([np_state(s) for s in states], [float(n) for n in sizes], order))


def aggregate_deterministic(states: list[dict], sizes: list[int], order, site_ids: list[int] | None = None) -> dict:
    """Replies arrive in `order`. DeterministicFedAvg sorts them by site id, then averages."""
    site_ids = list(range(len(states))) if site_ids is None else site_ids
    arrived = [np_state(states[i]) for i in order]
    weights = [float(sizes[i]) for i in order]
    sort = sorted_by_site([site_ids[i] for i in order])
    return torch_state(deterministic_weighted_average([arrived[i] for i in sort], [weights[i] for i in sort]))


def aggregate_reference(states: list[dict], sizes: list[int]) -> dict:
    """Repository float64 reference used by the torch reference: `task.fedavg_weighted`, site order."""
    return dict(fedavg_weighted(states, sizes))


def summarize_orders(records: list[dict]) -> dict:
    """records: one dict per order with `passes` (bool) and `max_abs_diff` (float)."""
    diffs = [r["max_abs_diff"] for r in records]
    return {"orders": len(records), "pass_rate": pass_rate([r["passes"] for r in records]),
            "orders_passing": sum(bool(r["passes"]) for r in records),
            "max_abs_diff_worst": max(diffs), "max_abs_diff_best": min(diffs),
            "max_abs_diff_median": float(np.median(diffs))}


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def p3_verdict(mse: dict[str, list[float]]) -> dict:
    """mse[order_key] = victim MSE per seed (same seeds for every order).

    order_effect  = max over orders of the seed-mean MSE minus the min over orders.
    seed_sd       = sample SD over seeds of the per-seed MSE, averaged over orders.
    P3 passes when order_effect < seed_sd, as written in the pre-registration.
    """
    seeds = {len(v) for v in mse.values()}
    if len(seeds) != 1 or seeds.pop() < 5:
        raise ValueError("need the same number of at least 5 seeds for every order")
    means = {k: float(np.mean(v)) for k, v in mse.items()}
    sds = [sample_sd(v) for v in mse.values()]
    per_seed_spread = [max(v[s] for v in mse.values()) - min(v[s] for v in mse.values())
                       for s in range(len(next(iter(mse.values()))))]
    effect = max(means.values()) - min(means.values())
    seed_sd = float(np.mean(sds))
    return {"orders": len(mse), "seeds": len(next(iter(mse.values()))),
            "order_effect_max_minus_min_of_seed_mean_mse": effect,
            "seed_sd_mean_over_orders": seed_sd, "seed_sd_min_over_orders": float(min(sds)),
            "per_seed_spread_over_orders_max": float(max(per_seed_spread)),
            "ratio_effect_to_seed_sd": effect / seed_sd if seed_sd > 0 else float("inf"),
            "p3_pass": bool(effect < seed_sd)}


# ---------------------------------------------------------------- replay engine

class Replay:
    """Training of every client from a given global state, cached by (client, state digest).

    Training is deterministic (seed + site id), so two orders that produce bitwise identical
    global models share their client updates. This changes the cost only, never a result.
    """

    def __init__(self, cfg: dict, manifest: dict, data_dir: Path):
        torch.set_num_threads(1)
        self.cfg = cfg
        self.features, self.classes = manifest["num_features"], manifest["num_classes"]
        self.data = [load_site_partition(data_dir / c["file"]) for c in manifest["client_partitions"]]
        self.sizes = [len(y) for _, y in self.data]
        self.n = len(self.data)
        self.rounds = int(cfg["rounds"])
        torch.manual_seed(cfg["seed"])
        self.init = {k: v.detach().clone() for k, v in DigitMLP(self.features, self.classes).state_dict().items()}
        self._cache: dict = {}
        self.trainings = 0

    def train(self, client: int, state: dict) -> dict:
        key = (client, state_digest(state))
        if key not in self._cache:
            m = DigitMLP(self.features, self.classes)
            m.load_state_dict(state)
            train_local(m, self.data[client][0], self.data[client][1], epochs=self.cfg["local_epochs"],
                        batch_size=self.cfg["batch_size"], lr=self.cfg["learning_rate"], seed=self.cfg["seed"] + client)
            self._cache[key] = {k: v.detach().clone() for k, v in m.state_dict().items()}
            self.trainings += 1
        return self._cache[key]

    def train_all(self, state: dict) -> list[dict]:
        return [self.train(c, state) for c in range(self.n)]

    def run(self, aggregate, orders_per_round) -> list[dict]:
        """Global model after each round. orders_per_round: one order per round."""
        g, out = self.init, []
        for order in orders_per_round:
            g = aggregate(self.train_all(g), self.sizes, order)
            out.append(g)
        return out


def replay_run(name: str, cfg_path: str, data_path: str, *, native: bool, primary: bool = False) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    data_dir = ROOT / data_path
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rp = Replay(cfg, manifest, data_dir)
    n, R = rp.n, rp.rounds
    orders = all_orders(n)
    # Reference trajectory (float64 reference, site order) and one-round client updates along it.
    ref, client_updates, g = [], [], rp.init
    for _ in range(R):
        ups = rp.train_all(g)
        client_updates.append(ups)
        g = aggregate_reference(ups, rp.sizes)
        ref.append(g)
    result = {"name": name, "config": cfg_path, "data_dir": data_path, "clients": n, "rounds": R,
              "sample_counts": rp.sizes, "native_run": native, "primary_for_p2": primary,
              "orders_total": len(orders), "gate": {"atol": ATOL, "rtol": RTOL}}

    # One-round view per round.
    for label, agg in (("upstream", aggregate_upstream), ("deterministic", aggregate_deterministic)):
        rounds_out = []
        for r in range(R):
            recs = []
            for o in orders:
                ok, d = gate(agg(client_updates[r], rp.sizes, o), ref[r])
                recs.append({"order": list(o), "passes": ok, "max_abs_diff": d})
            rounds_out.append({"round": r + 1, **summarize_orders(recs), "per_order": recs})
        result.setdefault("one_round", {})[label] = rounds_out

    # Trajectory view: same permutation every round, retraining every round.
    for label, agg in (("upstream", aggregate_upstream), ("deterministic", aggregate_deterministic)):
        per_order, final_recs = [], []
        for o in orders:
            traj = rp.run(agg, [o] * R)
            rounds_gate = [gate(traj[r], ref[r]) for r in range(R)]
            per_order.append({"order": list(o), "rounds": [{"round": r + 1, "passes": ok, "max_abs_diff": d}
                                                           for r, (ok, d) in enumerate(rounds_gate)]})
            final_recs.append({"order": list(o), "passes": rounds_gate[-1][0], "max_abs_diff": rounds_gate[-1][1],
                               "final_digest": state_digest(traj[-1])})
        by_round = []
        for r in range(R):
            by_round.append({"round": r + 1, **summarize_orders(
                [{"passes": p["rounds"][r]["passes"], "max_abs_diff": p["rounds"][r]["max_abs_diff"]} for p in per_order])})
        result.setdefault("trajectory", {})[label] = {
            "final": {**summarize_orders(final_recs), "distinct_final_models": len({f["final_digest"] for f in final_recs}),
                      "per_order": final_recs},
            "by_round": by_round, "per_order_rounds": per_order}
    # DeterministicFedAvg must be bitwise order invariant and equal to the reference.
    det_digests = {f["final_digest"] for f in result["trajectory"]["deterministic"]["final"]["per_order"]}
    result["deterministic_bitwise_equal_to_reference"] = det_digests == {state_digest(ref[-1])}

    if native:
        result["native_validation"] = validate_native(name, rp, ref, cfg)
    result["client_trainings"] = rp.trainings
    result["seconds"] = round(time.time() - t0, 1)
    return result


def missing_replay_inputs(name: str, data_path: str, native: bool) -> list[str]:
    """Inputs that `replay` needs and that Git does not hold. The prepared partitions under data/ and every *.pt
    (native final models and reference checkpoints) are git ignored, so they are absent in a fresh checkout."""
    need = [ROOT / data_path / "manifest.json"]
    if native:
        run = NATIVE / name
        need += [run / "server/final_model.pt", run / "bitwise_replay.json", run / "flower_parity.json"]
        if not list((run / "checkpoints").glob("*_final.pt")):
            need.append(run / "checkpoints/<id>_final.pt")
    return [str(p.relative_to(ROOT)) for p in need if "<id>" in str(p) or not p.is_file()]


def require_replay_inputs(name: str, data_path: str, native: bool) -> None:
    missing = missing_replay_inputs(name, data_path, native)
    if missing:
        raise SystemExit(
            f"checkpoints not in checkout: {name} cannot be replayed because these inputs are missing: "
            + ", ".join(missing) + ". The prepared partitions (data/) and the native checkpoints (*.pt) are git ignored. "
            "Produce them with the Docker runs in docs/v0.14-reproduce.md, or copy them from a full evidence bundle. "
            "`summarize` needs only the saved replay_*.json files and works in a fresh checkout.")


def validate_native(name: str, rp: Replay, ref: list[dict], cfg: dict) -> dict:
    """Anchor the replay to the saved native evidence: observed orders, saved models, saved gate."""
    run = NATIVE / name
    replay = json.loads((run / "bitwise_replay.json").read_text())
    parity = json.loads((run / "flower_parity.json").read_text())
    orders_seen = []
    for entry in replay["rounds"]:
        o = entry.get("order_used") or entry.get("final_model_bitwise_identical_for_order")
        orders_seen.append(tuple(o))
    traj = rp.run(aggregate_upstream, orders_seen)
    native_final = torch.load(run / "server/final_model.pt", weights_only=True)
    ref_ckpt_dir = run / "checkpoints"
    ref_final = torch.load(next(ref_ckpt_dir.glob("*_final.pt")), weights_only=True)
    ok_obs, d_obs = gate(traj[-1], ref_final)
    return {
        "observed_orders_per_round": [list(o) for o in orders_seen],
        "replay_with_observed_orders_equals_native_final_bitwise": all(torch.equal(traj[-1][k], native_final[k]) for k in native_final),
        "float64_reference_replay_equals_saved_reference_checkpoint_bitwise": all(torch.equal(ref[-1][k], ref_final[k]) for k in ref_final),
        "native_gate_saved_pass": parity["verified_numerical_parity"],
        "native_gate_saved_max_abs_diff": max(v["max_abs_diff"] for v in parity["layers"].values()),
        "replay_observed_orders_gate_pass": ok_obs, "replay_observed_orders_max_abs_diff": d_obs,
    }


# ---------------------------------------------------------------- P3

def p3_study(seeds: int, iterations: int, batch: int, stress_rounds: int, controls: bool) -> dict:
    """Victim MSE of repo gradient inversion on a one-step plaintext update, starting model varies by reply order.

    Metric: mean per-image MSE (optimal matching, evaluator side) of the repository DLG-style
    gradient-matching attack (`view_study.attack`, `reconstruct`) with known labels, applied to the
    victim's one-step full-batch plaintext model delta. Everything is fixed except the starting
    model: it is the global model after aggregating the 4 clients' updates in one of the 24 reply
    orders with upstream arithmetic. The seed varies the victim batch (8 records drawn from site 0)
    and the attack initialisation. The attack is deterministic given inputs.
    """
    from feddisclosure.attacks import view_study as vs
    cfg_path, data_path = "configs/training/fedavg_digits_iid_4clients.yaml", "data/training/digits-docker-iid-4c-v014"
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    data_dir = ROOT / data_path
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rp = Replay(cfg, manifest, data_dir)
    orders = all_orders(rp.n)
    victim_x, victim_y = rp.data[0]
    lr = float(cfg["learning_rate"])
    cache: dict = {}

    def victim_mse(start: dict, s: int) -> float:
        key = (state_digest(start), s)
        if key not in cache:
            idx = np.random.default_rng(10_000 + s).choice(len(victim_y), batch, replace=False)
            xb, yb = victim_x[idx], victim_y[idx]
            new = vs.client_step(start, xb, yb, lr=lr, sigma=None, seed=s)
            delta = {k: new[k] - start[k] for k in new}
            rec = vs.attack(start, delta, yb, "server_individual_model_delta", lr=lr, iterations=iterations, seed=1000 + s)
            cache[key] = vs.score_victim(xb, rec, batch)["victim_mse"]
        return cache[key]

    def block(starts: dict[str, dict]) -> dict:
        mse = {k: [victim_mse(st, s) for s in range(seeds)] for k, st in starts.items()}
        out = p3_verdict(mse)
        out["distinct_starting_models"] = len({state_digest(st) for st in starts.values()})
        out["mean_mse_by_order"] = {k: float(np.mean(v)) for k, v in mse.items()}
        out["per_seed_mse_by_order"] = mse
        return out

    t0 = time.time()
    # Context for the MSE scale: a no-information reconstruction (mean public digit from the other sites).
    pool = np.concatenate([rp.data[c][0] for c in range(1, rp.n)])
    floors = []
    for s in range(seeds):
        idx = np.random.default_rng(10_000 + s).choice(len(victim_y), batch, replace=False)
        floors.append(vs.chance_floor(victim_x[idx], pool, np.random.default_rng(s))["mean_image_mse"])
    updates1 = rp.train_all(rp.init)
    main_starts = {"-".join(map(str, o)): aggregate_upstream(updates1, rp.sizes, o) for o in orders}
    result = {"config": cfg_path, "data_dir": data_path, "seeds": seeds, "attack_iterations": iterations,
              "victim_batch": batch, "learning_rate": lr, "clients": rp.n,
              "mean_image_floor_mse_mean_over_seeds": float(np.mean(floors)),
              "prediction": "P3: difference in mean victim MSE between reply orders is below the seed standard deviation",
              "main_after_round_1": block(main_starts)}
    # Stress: models after `stress_rounds` rounds with the same permutation every round, where
    # earlier last-bit differences have had time to grow.
    stress = {}
    for o in orders:
        stress["-".join(map(str, o))] = rp.run(aggregate_upstream, [o] * stress_rounds)[-1]
    result["stress_after_%d_rounds" % stress_rounds] = block(stress)
    ref_stress_gap = max(gate(stress[k], next(iter(stress.values())))[1] for k in stress)
    result["stress_max_abs_weight_spread_between_orders"] = ref_stress_gap
    if controls:
        # Sensitivity control: how large a change of the starting model does move this metric?
        base = main_starts["0-1-2-3"]
        gen = torch.Generator().manual_seed(77)
        ctrl = {}
        for scale in (1e-8, 1e-7, 1e-6, 1e-4, 1e-2):
            starts = {}
            for j in range(8):
                starts[f"p{j}"] = {k: v + scale * torch.randn(v.shape, generator=gen) for k, v in base.items()}
            blk = block(starts)
            ctrl[f"gaussian_{scale:g}"] = {k: blk[k] for k in ("order_effect_max_minus_min_of_seed_mean_mse",
                                                               "seed_sd_mean_over_orders", "p3_pass")}
        result["sensitivity_control_perturbed_start"] = ctrl
    result["seconds"] = round(time.time() - t0, 1)
    result["limitations"] = [
        "One attack (repository DLG style gradient matching), one metric, one 4-client IID digits setup, one local step",
        "Seeds change the victim batch and the attack initialisation, not the training data partition or the model init",
        "Attack is deterministic given its inputs, so identical starting models share one run",
        "Reply orders compared are fixed permutations repeated every round, not every possible schedule",
        "Plaintext one-step update, so this says nothing about SecAgg+ or DP",
    ]
    return result


# ---------------------------------------------------------------- summary and CLI

def summarize(out_dir: Path = OUT) -> dict:
    rows, parity = {}, {}
    for path in sorted(out_dir.glob("replay_*.json")):
        r = json.loads(path.read_text())
        up, de = r["trajectory"]["upstream"]["final"], r["trajectory"]["deterministic"]["final"]
        row = {"clients": r["clients"], "rounds": r["rounds"], "native_run": r["native_run"],
               "primary_for_p2": r["primary_for_p2"], "orders": r["orders_total"],
               "upstream_final_pass_rate": up["pass_rate"], "upstream_final_worst_diff": up["max_abs_diff_worst"],
               "upstream_final_best_diff": up["max_abs_diff_best"],
               "upstream_one_round_min_pass_rate": min(x["pass_rate"] for x in r["one_round"]["upstream"]),
               "upstream_one_round_worst_diff": max(x["max_abs_diff_worst"] for x in r["one_round"]["upstream"]),
               "deterministic_final_pass_rate": de["pass_rate"], "deterministic_final_worst_diff": de["max_abs_diff_worst"],
               "deterministic_one_round_min_pass_rate": min(x["pass_rate"] for x in r["one_round"]["deterministic"]),
               "deterministic_bitwise_equal_to_reference": r["deterministic_bitwise_equal_to_reference"]}
        if r["native_run"]:
            nv = r["native_validation"]
            row.update({k: nv[k] for k in ("native_gate_saved_pass", "native_gate_saved_max_abs_diff",
                                           "replay_with_observed_orders_equals_native_final_bitwise",
                                           "float64_reference_replay_equals_saved_reference_checkpoint_bitwise",
                                           "replay_observed_orders_gate_pass")})
        rows[r["name"]] = row
    native = {k: v for k, v in rows.items() if v["native_run"]}
    p1_runs = {k: v["deterministic_final_pass_rate"] == 1.0 and v["deterministic_bitwise_equal_to_reference"]
               and v["float64_reference_replay_equals_saved_reference_checkpoint_bitwise"] for k, v in native.items()}
    c4 = {k: v for k, v in rows.items() if v["clients"] == 4}
    primary = [k for k, v in c4.items() if v["primary_for_p2"]]
    p2_primary = {k: c4[k]["upstream_final_pass_rate"] < 1.0 for k in primary}
    summary = {"schema_version": 1, "runs": rows,
               "P1": {"rule": "7 of 7 native runs pass the unchanged 1e-5 gate with DeterministicFedAvg (replay)",
                      "native_runs_in_study": len(native), "passing": sum(p1_runs.values()), "per_run": p1_runs,
                      "result": "PASS" if native and len(native) == 7 and all(p1_runs.values()) else "FAIL"},
               "P2": {"rule": "upstream pass rate over reply orders strictly below 100 percent for 4 clients (judged on the shipped 4-client config)",
                      "primary": p2_primary,
                      "result": "PASS" if primary and all(p2_primary.values()) else "FAIL",
                      "exploratory_4_client_runs_below_100_percent": {k: v["upstream_final_pass_rate"] < 1.0 for k, v in c4.items() if not v["primary_for_p2"]}},
               "baseline_upstream_float64_gate_native": {k: {"saved_pass": v["native_gate_saved_pass"], "saved_max_abs_diff": v["native_gate_saved_max_abs_diff"]} for k, v in native.items()}}
    p3 = out_dir / "p3_victim_mse.json"
    if p3.is_file():
        d = json.loads(p3.read_text())
        summary["P3"] = {"rule": d["prediction"], "main_after_round_1": {k: d["main_after_round_1"][k] for k in (
            "order_effect_max_minus_min_of_seed_mean_mse", "seed_sd_mean_over_orders", "p3_pass", "distinct_starting_models")},
            "result": "PASS" if d["main_after_round_1"]["p3_pass"] else "FAIL"}
    return summary


def live_hosted_run(live_dir: Path = LIVE_DIR) -> dict:
    """The six hosted runs of parity-live.yml, read from their saved flower_parity.json files."""
    runs = {}
    for name in LIVE_RUNS:
        f = live_dir / name / "flower_parity.json"
        if not f.is_file():
            raise FileNotFoundError(f"live parity evidence missing: {f}")
        d = json.loads(f.read_text())
        runs[name] = {"verified_numerical_parity": d["verified_numerical_parity"],
                      "worst_max_abs_diff": max(v["max_abs_diff"] for v in d["layers"].values())}
    return {"workflow": LIVE_TEXT["workflow"], "run_id": LIVE_TEXT["run_id"], "gate": LIVE_TEXT["gate"], "runs": runs,
            "reading": LIVE_TEXT["reading"], "earlier_attempt": LIVE_TEXT["earlier_attempt"]}


def prereg_check(s: dict, out_dir: Path = OUT, live_dir: Path = LIVE_DIR) -> dict:
    """Same layout as the other prereg_check_*.json files. Thresholds are copied, never changed.
    Everything in the committed file comes from the saved inputs or from the constants at the top of this file."""
    rows = s["runs"]
    c4 = {k: v for k, v in rows.items() if v["clients"] == 4}
    out = {"schema_version": 1, "preregistration": "docs/preregistration/v0.15.md",
           "note": NOTE,
           "P1": {"prediction": "DeterministicFedAvg (replies ordered by site id) closes the float64 parity gap",
                  "pass_rule": "7 of 7 native runs pass the unchanged 1e-5 gate",
                  "observed": {"native_runs_replayed": s["P1"]["native_runs_in_study"], "passing_with_deterministic": s["P1"]["passing"],
                               "worst_deterministic_max_abs_diff": max(v["deterministic_final_worst_diff"] for v in rows.values() if v["native_run"]),
                               "upstream_saved_native_gate_passes": sum(bool(v["native_gate_saved_pass"]) for v in rows.values() if v["native_run"]),
                               "replay_reproduces_saved_native_final_model_bitwise": sum(bool(v["replay_with_observed_orders_equals_native_final_bitwise"]) for v in rows.values() if v["native_run"]),
                               "per_run": s["P1"]["per_run"]},
                  "pass": s["P1"]["result"] == "PASS",
                  "scope": P1_SCOPE, "label": P1_LABEL},
           "P2": {"prediction": "Upstream FedAvg parity depends on reply order",
                  "pass_rule": "In the bitwise replay, the pass rate over reply orders is strictly below 100% for 4 clients",
                  "observed": {"primary_shipped_4_client_config": {k: {"upstream_pass_rate": c4[k]["upstream_final_pass_rate"],
                                                                      "worst_max_abs_diff": c4[k]["upstream_final_worst_diff"]} for k in s["P2"]["primary"]},
                               "exploratory_4_client_configs": {k: {"upstream_pass_rate": c4[k]["upstream_final_pass_rate"],
                                                                   "worst_max_abs_diff": c4[k]["upstream_final_worst_diff"]}
                                                                for k in c4 if not c4[k]["primary_for_p2"]},
                               "note": P2_NOTE},
                  "pass": s["P2"]["result"] == "PASS"},
           "baseline_float64_gate_native_runs": s["baseline_upstream_float64_gate_native"]}
    if "P3" in s:
        p3 = json.loads((out_dir / "p3_victim_mse.json").read_text())["main_after_round_1"]
        out["P3"] = {"prediction": "Reply order does not move privacy metrics beyond seed noise",
                     "pass_rule": "Difference in mean victim MSE between reply orders is below the seed standard deviation",
                     "observed": {k: p3[k] for k in ("orders", "seeds", "distinct_starting_models",
                                                     "order_effect_max_minus_min_of_seed_mean_mse", "seed_sd_mean_over_orders",
                                                     "ratio_effect_to_seed_sd")} | {"sensitivity_note": P3_SENSITIVITY_NOTE},
                     "pass": s["P3"]["result"] == "PASS", "label": P3_LABEL}
    out["live_hosted_run"] = live_hosted_run(live_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--run", required=True, help="native run name or replay-only name")
    p = sub.add_parser("p3")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--iterations", type=int, default=350)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--stress-rounds", type=int, default=10)
    p.add_argument("--no-controls", action="store_true")
    sub.add_parser("summarize")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "replay":
        if a.run in NATIVE_RUNS:
            cfg, data = NATIVE_RUNS[a.run]
            require_replay_inputs(a.run, data, True)
            res = replay_run(a.run, cfg, data, native=True)
        elif a.run in REPLAY_ONLY:
            cfg, data, primary = REPLAY_ONLY[a.run]
            require_replay_inputs(a.run, data, False)
            res = replay_run(a.run, cfg, data, native=False, primary=primary)
        else:
            raise SystemExit(f"unknown run {a.run}")
        (OUT / f"replay_{a.run}.json").write_text(json.dumps(res, indent=1) + "\n")
        print(a.run, "upstream final", res["trajectory"]["upstream"]["final"]["pass_rate"],
              "deterministic final", res["trajectory"]["deterministic"]["final"]["pass_rate"], res["seconds"], "s")
    elif a.cmd == "p3":
        require_replay_inputs("p3", "data/training/digits-docker-iid-4c-v014", False)
        res = p3_study(a.seeds, a.iterations, a.batch, a.stress_rounds, not a.no_controls)
        (OUT / "p3_victim_mse.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res["main_after_round_1"] | {"per_seed_mse_by_order": "..."}, indent=1, default=str)[:1500])
    else:
        s = summarize()
        PARITY_JSON.parent.mkdir(parents=True, exist_ok=True)
        PARITY_JSON.write_text(json.dumps(prereg_check(s), indent=2) + "\n")
        (OUT / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps({k: s[k] for k in ("P1", "P2", "P3") if k in s}, indent=1)[:2500])


if __name__ == "__main__":
    main()
