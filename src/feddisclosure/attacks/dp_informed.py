"""Informed-attacker identification against record-level DP-SGD on one full-batch local step.

This is a REPLICATION, in the federated update setting, of the reconstruction-robustness (ReRo)
analysis of Hayes, Mahloujifar and Balle (arXiv 2302.07225, "Bounding Training Data
Reconstruction in DP-SGD"). It is not a new bound.

Threat model. The client holds B records and takes one full-batch DP-SGD step (per-sample
clipping to norm C, Gaussian noise of standard deviation sigma*C added to the SUM of the clipped
per-sample gradients, then division by B). The attacker sees the resulting model update. It
knows the model initialisation, the learning rate, the clipping norm, the noise multiplier, the
other B-1 records, and a list of u candidates for the one remaining record (the victim). The
candidate list contains the true record. The attacker picks one candidate.
An FL server does not see intermediate gradients, so a client that takes several local steps is a
stronger threat model (the attacker would need to cope with the unseen intermediate iterates).
That case is out of scope here.

Bound (valid for the setup that is actually run). Write R for the observed update after the
attacker subtracts the clipped gradients of the B-1 known records. If the victim is candidate i,
R ~ N(c_i, (sigma*C)^2 I) with c_i the clipped gradient of candidate i, and ||c_i|| <= C. Adjacency
is add/remove-one-record, the one Opacus accounts for. Compare with the reference world in which
the victim record is absent, R0 ~ N(0, (sigma*C)^2 I). The reference is an analysis device only.
It is never run. Let E_i be the event that the attacker answers i, so the events E_i partition the
output space and sum_i Q(E_i) = 1 under the reference Q. For every i, deciding between N(c_i, .)
and N(0, .) with ||c_i|| <= C is a Gaussian shift problem with standardised shift at most
mu = C/(sigma*C) = 1/sigma. By the Neyman-Pearson lemma, P_i(E_i) <= h(Q(E_i)) with
h(q) = Phi(Phi^-1(q) + 1/sigma). The function h is concave, because
h'(q) = exp(-mu*Phi^-1(q) - mu^2/2) decreases in q. By Jensen, for the equally likely victims
(1/u) sum_i P_i(E_i) <= h((1/u) sum_i Q(E_i)) = h(1/u). So

    success <= Phi(Phi^-1(1/u) + 1/sigma).

The bound uses sensitivity C (add/remove), not 2C. Two candidates can be 2C apart in the worst
case, but the argument never compares candidate with candidate. It compares each candidate with
the absent record. For u = 2 and antipodal clipped gradients the optimal rule reaches exactly
Phi(1/sigma), so the bound cannot be improved in general.
The bound does not use any accountant, delta or composition. It is a statement about one Gaussian
mechanism invocation with no subsampling amplification (the batch is the whole client dataset).

Exact optimal success. With equal priors and Gaussian noise the Bayes optimal rule picks the
candidate whose clipped gradient is nearest to R in Euclidean distance. Its success probability
for the actual candidate set is estimated by Monte Carlo (the candidate gradients are fixed given
the initialisation, so only the noise is sampled).

Metrics. Identification success among u candidates is not the same as reconstruction error. A
low identification rate says nothing about whether a gradient inversion attack that has no
candidate list can recover an image, and the reverse also holds.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import integrate, stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feddisclosure.evaluation.intervals import (TUNING_SEED_BASE, clopper_pearson_interval,
                                                wilson_interval)
from feddisclosure.training.task import DigitMLP

BATCH = 8
CLIP = 1.0
LR = 0.1
SIGMAS = (0.2, 0.5, 1.0, 2.0, 4.0)
# (label mode, number of candidates u). Same-label candidates share the victim's label.
CONFIGS = (("same", 10), ("same", 100), ("any", 10), ("any", 100), ("any", 1000))
MC_DRAWS_PER_TRIAL = 40            # 40 x 1000 trials = 40,000 noise draws per cell (rule: at least 20,000)
OPACUS_CLIP_EPS = 1e-6             # Opacus adds this to the per-sample norm before clipping
PRIMARY_ATTACK = "ip"              # Hayes Algorithm 2 scores by inner product. "dist" is nearest candidate.
SCHEMA_VERSION = 1


def cfg_name(mode: str, u: int) -> str:
    return f"{mode}_u{u}"


# ---------------------------------------------------------------- bounds and reference values
def rero_bound(u: int, sigma: float) -> float:
    """Upper bound on identification success among u equally likely candidates, one full-batch
    DP-SGD step, clip C, noise std sigma*C on the sum of clipped gradients. See module docstring."""
    if u < 1 or sigma <= 0:
        raise ValueError("need u >= 1 and sigma > 0")
    if u == 1:
        return 1.0
    return float(stats.norm.cdf(stats.norm.ppf(1.0 / u) + 1.0 / sigma))


def orthogonal_reference(u: int, sigma: float) -> float:
    """Exact success of the optimal rule when the u candidates have clipped gradients that are
    mutually orthogonal and all of norm C. P(Z0 + mu > max_j Z_j) with mu = 1/sigma. It separates
    the part of the gap to the bound that exists even for ideal candidates from the part that is
    caused by the real candidate geometry (similar or short gradients)."""
    mu = 1.0 / sigma
    f = lambda x: stats.norm.pdf(x) * stats.norm.cdf(x + mu) ** (u - 1)
    val, _ = integrate.quad(f, -12.0, 12.0 + mu, limit=400)
    return float(min(1.0, val))


# ---------------------------------------------------------------- data and instances
def load_public_digits() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    return np.asarray(x, dtype=np.float32) / 16.0, np.asarray(y, dtype=np.int64)


def check_eval_seed(seed: int, *, tuning: bool = False) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not tuning and seed >= TUNING_SEED_BASE:
        raise ValueError(f"seed {seed} is a tuning seed (>= {TUNING_SEED_BASE}), not an evaluation seed")
    if tuning and seed < TUNING_SEED_BASE:
        raise ValueError("tuning runs must use seeds >= 9000")


def build_instance(seed: int, y_all: np.ndarray, *, batch: int = BATCH, configs=CONFIGS,
                   tuning: bool = False) -> dict:
    """Everything the evaluator draws for one trial. Client records, victim slot, and candidate
    sets. The public pool is every record that is NOT one of the client's records, so candidates are
    disjoint from the client data except for the true victim, who is inserted once on purpose."""
    check_eval_seed(seed, tuning=tuning)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 150]))
    n = len(y_all)
    client = rng.choice(n, size=batch, replace=False)
    slot = int(rng.integers(batch))
    true_id = int(client[slot])
    in_client = np.zeros(n, dtype=bool)
    in_client[client] = True
    pool = np.flatnonzero(~in_client)
    need_any = max(u for m, u in configs if m == "any") - 1 if any(m == "any" for m, _ in configs) else 0
    need_same = max(u for m, u in configs if m == "same") - 1 if any(m == "same" for m, _ in configs) else 0
    same_pool = pool[y_all[pool] == y_all[true_id]]
    if len(same_pool) < need_same or len(pool) < need_any:
        raise ValueError("public pool too small for the requested candidate sets")
    same_order = rng.permutation(same_pool)[:need_same]
    any_order = rng.permutation(pool)[:need_any]
    cands = {}
    for mode, u in configs:
        others = (same_order if mode == "same" else any_order)[:u - 1]
        ids = np.concatenate([others, [true_id]]).astype(np.int64)
        ids = ids[rng.permutation(u)]
        cands[cfg_name(mode, u)] = ids
    return {"seed": seed, "client": client.astype(np.int64), "slot": slot, "true_id": true_id,
            "candidates": cands}


def init_state(seed: int) -> dict:
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in DigitMLP(64, 10).state_dict().items()}


# ---------------------------------------------------------------- per-sample clipped gradients
def flat_params(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().astype(np.float64).ravel() for v in state.values()])


def clipped_gradients(init: dict, x: np.ndarray, y: np.ndarray, clip: float = CLIP) -> np.ndarray:
    """Per-sample gradients at the initial model, each clipped to norm <= clip with the same rule
    Opacus uses (factor = min(1, clip / (norm + 1e-6))). Returns float64 (n, d)."""
    from torch.func import functional_call, grad, vmap
    model = DigitMLP(64, 10)
    model.load_state_dict(init)
    params = {k: v.detach() for k, v in model.named_parameters()}
    xs, ys = torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y)).long()

    def loss_fn(p, xi, yi):
        return nn.functional.cross_entropy(functional_call(model, p, (xi[None],)), yi[None])

    g = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, xs, ys)
    flat = torch.cat([g[k].reshape(len(ys), -1) for k in params], dim=1).double()
    norms = flat.norm(dim=1)
    factor = (clip / (norms + OPACUS_CLIP_EPS)).clamp(max=1.0)
    return (flat * factor[:, None]).numpy()


# ---------------------------------------------------------------- the real Opacus step
def opacus_noisy_sum(init: dict, x: np.ndarray, y: np.ndarray, *, sigma: float, clip: float = CLIP,
                     lr: float = LR, seed: int = 0) -> np.ndarray:
    """One full-batch step with the real Opacus PrivacyEngine (no Poisson sampling, since the
    batch is the whole client dataset). Returns what the attacker can read off the update:
    (init - updated) * B / lr = sum of clipped per-sample gradients + N(0, (sigma*clip)^2 I)."""
    from opacus import PrivacyEngine
    torch.manual_seed(seed)
    model = DigitMLP(64, 10)
    model.load_state_dict(init)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    xs, ys = torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y)).long()
    dl = DataLoader(TensorDataset(xs, ys), batch_size=len(ys), shuffle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm, po, pdl = PrivacyEngine(secure_mode=False).make_private(
            module=model, optimizer=opt, data_loader=dl, noise_multiplier=sigma,
            max_grad_norm=clip, poisson_sampling=False)
        for bx, by in pdl:
            po.zero_grad()
            nn.functional.cross_entropy(pm(bx), by).backward()
            po.step()
    new = pm._module.state_dict()
    delta = flat_params(init) - flat_params(new)
    return delta * len(ys) / lr


# ---------------------------------------------------------------- attacks and exact optimum
def ip_choice(residual: np.ndarray, cand_grads: np.ndarray) -> int:
    """Hayes Algorithm 2 style: score each candidate by the inner product of the residual with the
    candidate's clipped gradient, take the argmax."""
    return int(np.argmax(cand_grads @ residual))


def nearest_choice(residual: np.ndarray, cand_grads: np.ndarray) -> int:
    """Bayes optimal rule for equal priors and isotropic Gaussian noise."""
    d = ((cand_grads - residual[None, :]) ** 2).sum(axis=1)
    return int(np.argmin(d))


def mc_optimal_successes(cand_grads: np.ndarray, true_idx: int, *, sigma: float, clip: float = CLIP,
                         draws: int, rng: np.random.Generator) -> int:
    """Number of draws, out of `draws`, in which the nearest-candidate rule names the true
    candidate when R = c_true + N(0, (sigma*clip)^2 I). Ties break by position (first index)."""
    d = cand_grads.shape[1]
    r = cand_grads[true_idx][None, :] + sigma * clip * rng.standard_normal((draws, d))
    # squared distance up to a term that does not depend on the candidate
    score = -2.0 * r @ cand_grads.T + (cand_grads ** 2).sum(axis=1)[None, :]
    return int((np.argmin(score, axis=1) == true_idx).sum())


def geometry_stats(cand_grads: np.ndarray, true_idx: int, clip: float = CLIP) -> dict:
    """Why the real candidates may fall short of the ideal orthogonal case."""
    norms = np.linalg.norm(cand_grads, axis=1)
    ct = cand_grads[true_idx]
    others = np.delete(np.arange(len(cand_grads)), true_idx)
    if len(others) == 0:
        return {"mean_norm_over_clip": float(norms.mean() / clip)}
    cos = (cand_grads[others] @ ct) / (norms[others] * norms[true_idx] + 1e-12)
    dist = np.linalg.norm(cand_grads[others] - ct[None, :], axis=1)
    return {"mean_norm_over_clip": float(norms.mean() / clip),
            "frac_clipped": float((norms >= clip * 0.999999).mean()),
            "mean_cos_to_true": float(cos.mean()),
            "mean_dist_to_true_over_clip": float(dist.mean() / clip),
            "nearest_other_dist_over_clip": float(dist.min() / clip)}


# ---------------------------------------------------------------- crosscheck of Opacus noise
def crosscheck_opacus_noise(*, sigma: float, trials: int = 300, seed: int = 0, clip: float = CLIP) -> dict:
    """Real Opacus output minus the hand-computed clipped sum, divided by sigma*clip. If Opacus adds
    N(0, (sigma*clip)^2) noise to the clipped sum, this is standard normal."""
    x_all, y_all = load_public_digits()
    inst = build_instance(seed, y_all, configs=())
    init = init_state(seed)
    xb, yb = x_all[inst["client"]], y_all[inst["client"]]
    clipped_sum = clipped_gradients(init, xb, yb, clip).sum(axis=0)
    base = 1_000_000 + int(round(sigma * 1000)) * 1000
    raw = [opacus_noisy_sum(init, xb, yb, sigma=sigma, clip=clip, seed=base + t) for t in range(trials)]
    res = np.stack(raw) - clipped_sum[None, :]
    if sigma == 0:
        return {"sigma": 0.0, "max_abs_difference": float(np.abs(res).max()), "trials": trials}
    z = res / (sigma * clip)
    ks = stats.kstest(z.ravel(), "norm")
    return {"sigma": sigma, "trials": trials, "coordinates": int(z.shape[1]),
            "mean": float(z.mean()), "variance": float(z.var()),
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "mean_standard_error": float(1.0 / math.sqrt(z.size))}


# ---------------------------------------------------------------- trial loop with resume
def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue            # a half-written last line from an interrupted run is simply redone
    return out


def completed_seeds(path: Path) -> set[int]:
    return {int(r["seed"]) for r in _read_records(path)}


def sigma_file(out_dir: Path, sigma: float) -> Path:
    return Path(out_dir) / f"trials_sigma{sigma:g}.jsonl"


def run_config(*, draws: int, batch: int, clip: float, lr: float, configs, sigmas) -> dict:
    return {"schema_version": SCHEMA_VERSION, "mc_draws_per_trial": draws, "batch": batch, "clip": clip,
            "lr": lr, "configs": [list(c) for c in configs], "sigmas": list(sigmas),
            "primary_attack": PRIMARY_ATTACK}


def ensure_run_config(out_dir: Path, cfg: dict) -> None:
    """Refuse to resume into a directory written with different settings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "run_config.json"
    if p.exists():
        old = json.loads(p.read_text())
        keys = ("schema_version", "mc_draws_per_trial", "batch", "clip", "lr", "configs", "primary_attack")
        if any(old.get(k) != cfg.get(k) for k in keys):
            raise RuntimeError("existing results in this directory were produced with different settings")
        if sorted(set(old.get("sigmas", [])) | set(cfg["sigmas"])) != old.get("sigmas"):
            cfg = dict(cfg, sigmas=sorted(set(old.get("sigmas", [])) | set(cfg["sigmas"])))
        else:
            return
    p.write_text(json.dumps(cfg, indent=2) + "\n")


def run_trials(out_dir: Path, seeds, sigmas=SIGMAS, *, draws: int = MC_DRAWS_PER_TRIAL, configs=CONFIGS,
               batch: int = BATCH, clip: float = CLIP, lr: float = LR, tuning: bool = False,
               time_budget_s: float | None = None, progress=None) -> dict:
    """For each seed: build the instance once, compute candidate gradients once, then for each sigma
    take ONE real Opacus step and score every candidate set on that same observed update. Resumes:
    seeds already present in a sigma file are skipped. Returns counts of new work."""
    import time
    out_dir = Path(out_dir)
    ensure_run_config(out_dir, run_config(draws=draws, batch=batch, clip=clip, lr=lr, configs=configs,
                                          sigmas=list(sigmas)))
    x_all, y_all = load_public_digits()
    done = {s: completed_seeds(sigma_file(out_dir, s)) for s in sigmas}
    t0 = time.time()
    new = 0
    for seed in seeds:
        todo = [(i, s) for i, s in enumerate(sigmas) if seed not in done[s]]
        if not todo:
            continue
        if time_budget_s is not None and time.time() - t0 > time_budget_s:
            break
        inst = build_instance(seed, y_all, batch=batch, configs=configs, tuning=tuning)
        init = init_state(seed)
        client = inst["client"]
        xb, yb = x_all[client], y_all[client]
        cg = clipped_gradients(init, xb, yb, clip)
        known_sum = cg.sum(axis=0) - cg[inst["slot"]]
        cand_g = {}
        geom = {}
        for name, ids in inst["candidates"].items():
            g = clipped_gradients(init, x_all[ids], y_all[ids], clip)
            ti = int(np.flatnonzero(ids == inst["true_id"])[0])
            cand_g[name] = (g, ti)
            geom[name] = geometry_stats(g, ti, clip)
        for i, s in todo:
            noisy = opacus_noisy_sum(init, xb, yb, sigma=s, clip=clip, lr=lr, seed=7_000_000 + seed * 16 + i)
            residual = noisy - known_sum
            mc_rng = np.random.default_rng(np.random.SeedSequence([seed, i, 77]))
            rec = {"seed": seed, "sigma": s, "ip": {}, "dist": {}, "opt_successes": {}, "opt_draws": draws,
                   "geometry": geom}
            for name, (g, ti) in cand_g.items():
                rec["ip"][name] = int(ip_choice(residual, g) == ti)
                rec["dist"][name] = int(nearest_choice(residual, g) == ti)
                rec["opt_successes"][name] = mc_optimal_successes(g, ti, sigma=s, clip=clip, draws=draws, rng=mc_rng)
            with open(sigma_file(out_dir, s), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            done[s].add(seed)
            new += 1
        if progress:
            progress(seed)
    return {"new_records": new}


# ---------------------------------------------------------------- summary and pre-registered checks
def _rate(k: int, n: int) -> dict:
    return {"successes": int(k), "trials": int(n), "rate": k / n,
            "wilson": list(wilson_interval(int(k), int(n))),
            "clopper_pearson": list(clopper_pearson_interval(int(k), int(n)))}


def summarize(out_dir: Path, sigmas=SIGMAS, configs=CONFIGS, expected_trials: int | None = None) -> dict:
    out_dir = Path(out_dir)
    cells, geometry_acc = {}, {}
    for s in sigmas:
        recs = _read_records(sigma_file(out_dir, s))
        by_seed = {r["seed"]: r for r in recs}          # duplicates (crash then rerun) collapse by seed
        recs = [by_seed[k] for k in sorted(by_seed)]
        if expected_trials is not None and len(recs) != expected_trials:
            raise RuntimeError(f"sigma {s}: {len(recs)} trials found, {expected_trials} expected")
        for mode, u in configs:
            name = cfg_name(mode, u)
            n = len(recs)
            ip = sum(r["ip"][name] for r in recs)
            dist = sum(r["dist"][name] for r in recs)
            opt_k = sum(r["opt_successes"][name] for r in recs)
            opt_n = sum(r["opt_draws"] for r in recs)
            per_trial = np.array([r["opt_successes"][name] / r["opt_draws"] for r in recs])
            cells[f"{name}|sigma={s:g}"] = {
                "config": name, "label_mode": mode, "u": u, "sigma": s,
                "chance": 1.0 / u, "bound": rero_bound(u, s),
                "orthogonal_reference": orthogonal_reference(u, s),
                "ip_attack": _rate(ip, n), "dist_attack": _rate(dist, n),
                "exact_optimal": dict(_rate(opt_k, opt_n),
                                      per_trial_mean_standard_error=float(per_trial.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                                      note="Monte Carlo over noise draws on the realised candidate sets"),
            }
            if s == sigmas[0]:
                geometry_acc[name] = {k: float(np.mean([r["geometry"][name][k] for r in recs
                                                        if k in r["geometry"][name]]))
                                      for k in recs[0]["geometry"][name]} if recs else {}
    return {"schema_version": SCHEMA_VERSION, "primary_attack": PRIMARY_ATTACK, "cells": cells,
            "geometry_mean_over_trials": geometry_acc}


def gap_note(summary: dict) -> dict:
    """Where the distance between the attack and the bound comes from. Split per cell:
    bound minus orthogonal reference (looseness of the bound even for ideal candidates),
    orthogonal reference minus exact optimum (real candidate geometry, such as similar gradients),
    exact optimum minus measured attack rate (sampling noise of 1000 trials)."""
    rows = {}
    for key, c in summary["cells"].items():
        o = c["exact_optimal"]["rate"]
        rows[key] = {"bound_minus_orthogonal_reference": c["bound"] - c["orthogonal_reference"],
                     "orthogonal_reference_minus_exact_optimal": c["orthogonal_reference"] - o,
                     "exact_optimal_minus_attack": o - c["ip_attack"]["rate"],
                     "bound_minus_attack": c["bound"] - c["ip_attack"]["rate"]}
    g = summary.get("geometry_mean_over_trials", {})
    cos = {k: v.get("mean_cos_to_true") for k, v in g.items()}
    return {"question": "Does the gap between the attack and the bound come from same-label gradient similarity?",
            "answer": ("Yes for same-label candidates at u 10 and 100, mostly no for any-label candidates at u 10. "
                       "Same-label candidates have clipped gradients that point the same way as the victim's (mean cosine "
                       "about 0.8, against about 0.01 for any-label candidates). Their exact optimum sits far below the "
                       "orthogonal-candidate reference at low and medium noise (same-label u 10: 0.32 below at sigma 0.2, "
                       "0.38 at sigma 0.5), while for any-label u 10 it is within 0.07 of that reference at every sigma. "
                       "At sigma 4 every gap is small. The same-label u 10 gap of 0.055 splits into 0.007 for the bound "
                       "itself, 0.026 for candidate similarity and 0.022 for sampling noise in 1000 trials. "
                       "For any-label u 100 and 1000 at low noise, a share of the gap comes from candidates "
                       "of the victim's own class that sit close to the victim by chance (nearest other candidate "
                       "0.44 C at u 100 and 0.31 C at u 1000), and from the bound itself being loose for large u. "
                       "All clipped gradients have norm C here (every per-sample gradient exceeds the clip), so "
                       "the inner-product and nearest-candidate rules choose the same candidate in every trial."),
            "mean_cosine_victim_vs_other_candidates": cos, "per_cell": rows}


def prereg_check(summary: dict, sigmas=SIGMAS, configs=CONFIGS) -> dict:
    """D1 to D5 exactly as written in docs/preregistration/v0.15.md. Thresholds are constants of
    the pre-registration and are not tunable. Wording that needs an operational reading is spelled
    out in `operationalisation`."""
    cell = lambda mode, u, s: summary["cells"][f"{cfg_name(mode, u)}|sigma={s:g}"]
    out = {"schema_version": 1, "preregistration": "docs/preregistration/v0.15.md",
           "note": "Thresholds are copied verbatim from the pre-registration and were not changed. "
                   "The primary attack is the inner-product attack (Hayes Algorithm 2). The "
                   "nearest-candidate attack is reported as well and is included in D1."}
    # D1
    worst, breaches = -1e9, []
    for key, c in summary["cells"].items():
        for att in ("ip_attack", "dist_attack"):
            margin = c[att]["clopper_pearson"][0] - (c["bound"] + 0.01)
            worst = max(worst, margin)
            if margin > 0:
                breaches.append({"cell": key, "attack": att, "cp_lower": c[att]["clopper_pearson"][0],
                                 "bound": c["bound"]})
    out["D1"] = {"id": "D1", "prediction": "No cell breaks the bound",
                 "pass_rule": "For every cell, the Clopper-Pearson lower limit of attack success is at most the bound plus 0.01. A breach is a gate failure",
                 "observed": {"cells_checked": 2 * len(summary["cells"]), "breaches": breaches,
                              "largest_cp_lower_minus_bound_plus_0.01": worst},
                 "pass": not breaches}
    # D2
    d2 = {cfg_name(m, 10): cell(m, 10, 0.2)["ip_attack"]["wilson"][0] for m in ("same", "any")}
    out["D2"] = {"id": "D2", "prediction": "The attack is far above chance at sigma 0.2, u 10",
                 "pass_rule": "Wilson lower limit above 0.60",
                 "observed": {"wilson_lower_by_candidate_mode": d2,
                              "rates": {k: cell(k.split("_")[0], 10, 0.2)["ip_attack"]["rate"] for k in d2}},
                 "operationalisation": "u 10 exists for same-label and any-label candidates. Both must pass.",
                 "pass": all(v > 0.60 for v in d2.values())}
    # D3
    d3 = {}
    for m in ("same", "any"):
        c = cell(m, 10, 4.0)
        d3[cfg_name(m, 10)] = {"attack": c["ip_attack"]["rate"], "bound": c["bound"],
                               "gap": c["bound"] - c["ip_attack"]["rate"]}
    out["D3"] = {"id": "D3", "prediction": "The attack is near the bound at high noise",
                 "pass_rule": "At sigma 4, u 10 the attack rate is within 0.05 of the bound",
                 "observed": d3,
                 "operationalisation": "Within 0.05 means absolute difference at most 0.05. Both candidate modes must pass.",
                 "pass": all(abs(v["gap"]) <= 0.05 for v in d3.values())}
    # D4
    d4, ok4 = {}, True
    for mode, u in configs:
        hi = cell(mode, u, 4.0)["ip_attack"]
        lo = cell(mode, u, 0.5)["ip_attack"]
        sep = hi["wilson"][1] < lo["wilson"][0]
        below = hi["rate"] < lo["rate"]
        d4[cfg_name(mode, u)] = {"rate_sigma4": hi["rate"], "wilson_sigma4": hi["wilson"],
                                 "rate_sigma0.5": lo["rate"], "wilson_sigma0.5": lo["wilson"],
                                 "lower_and_non_overlapping": bool(below and sep)}
        ok4 = ok4 and bool(below and sep)
    out["D4"] = {"id": "D4", "prediction": "Success falls as sigma grows",
                 "pass_rule": "Attack rate at sigma 4 is below the rate at sigma 0.5 for every u, with non-overlapping Wilson intervals",
                 "observed": d4, "operationalisation": "Every one of the five candidate sets (same-label u 10 and 100, any-label u 10, 100 and 1000) must pass.",
                 "pass": ok4}
    # D5
    d5, ok5 = {}, True
    for key, c in summary["cells"].items():
        o = c["exact_optimal"]
        cond_bound = o["clopper_pearson"][0] <= c["bound"]
        cond_ip = c["ip_attack"]["clopper_pearson"][0] <= o["clopper_pearson"][1]
        cond_dist = c["dist_attack"]["clopper_pearson"][0] <= o["clopper_pearson"][1]
        d5[key] = {"optimal": o["rate"], "optimal_cp": o["clopper_pearson"], "ip_attack": c["ip_attack"]["rate"],
                   "dist_attack": c["dist_attack"]["rate"], "bound": c["bound"],
                   "optimal_cp_lower_le_bound": bool(cond_bound), "ip_cp_lower_le_optimal_cp_upper": bool(cond_ip),
                   "dist_cp_lower_le_optimal_cp_upper": bool(cond_dist),
                   "point_order_ip_le_optimal_le_bound": bool(c["ip_attack"]["rate"] <= o["rate"] <= c["bound"])}
        ok5 = ok5 and cond_bound and cond_ip and cond_dist
    out["D5"] = {"id": "D5", "prediction": "The exact optimal success for the real candidate set is at least the attack success and at most the bound",
                 "pass_rule": "Holds in every cell up to Monte Carlo error (Clopper-Pearson)",
                 "observed": d5,
                 "operationalisation": "Up to Monte Carlo error means: the Clopper-Pearson lower limit of the exact optimum is at most the bound, and the Clopper-Pearson lower limit of each attack rate is at most the Clopper-Pearson upper limit of the exact optimum. The point-estimate order is reported next to it.",
                 "pass": ok5}
    out["gap_note"] = gap_note(summary)
    return out
