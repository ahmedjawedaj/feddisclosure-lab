"""v0.15 WP8: does the evidence chain have teeth?

Known faults are injected one at a time into the privacy pipeline and every detector is asked
whether it notices. Real components only: the Opacus 1.6 PrivacyEngine step, the upstream Flower
SecAgg+ quantiser through `view_study.secagg_aggregate`, the repository verifiers in `scripts/`
run as programs on tampered COPIES of saved evidence, and the informed attacker of
`attacks/dp_informed.py` used as a black-box audit.

Fault types (exactly five)
  F1 DP noise scale wrong: real Opacus step with noise multiplier 0.25, 0.5 or 2 times the claimed sigma.
  F2 Per-sample clipping skipped: real Opacus step with a huge max grad norm (1e6) and the noise
     multiplier divided by the same factor, so the noise std stays at the claimed sigma * C and only
     the clipping is missing.
  F3 Plaintext fallback: the SecAgg+ aggregate silently replaced by the plaintext weighted mean.
  F4 Wrong threshold or dropout handling: (a) a dropout below the threshold is tolerated, (b) a dropout
     exactly at the threshold halts the run (off by one), (c) the aggregate silently includes the
     dropped client's cached update.
  F5 A client trains from a stale global model (the round 0 model in round 1).
No fault touches mask generation.

Detectors
  A  existing gates and verifiers run on tampered evidence copies (never on saved evidence)
  B  many-run white-box DP audit: chi-square test on standardised residuals of real Opacus outputs
     against the known clipped sum, plus a clipping-norm audit with single-record probes
  C  informed identification attack as a black-box audit against the Clopper-Pearson bound for the claimed sigma
  D1 quantisation error inside the proven bound (2cn/T plus the weight rounding term)
  D2 lower and upper tail test of the quantisation error against its exact distribution (an extension
     added so that the plaintext fallback, whose error is zero, can be seen at all)

Nothing here is tuned on evaluation seeds. Audit indices 0 and up are evaluation audits and indices
from 9000 up are tuning audits (see `feddisclosure.evaluation.intervals`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from feddisclosure.attacks import dp_informed as dpi
from feddisclosure.attacks.view_study import CLIP_RANGE, QUANT_RANGE, flatten, secagg_aggregate, unflatten
from feddisclosure.evaluation.intervals import (TUNING_SEED_BASE, clopper_pearson_interval, rate_summary)
from feddisclosure.training.task import DigitMLP, fedavg_weighted, train_local

ROOT = Path(__file__).resolve().parents[3]
SAVED_EVIDENCE = ROOT / "reports" / "results"
OUT_DIR = SAVED_EVIDENCE / "privacy" / "v0150" / "fault_matrix"

# ------------------------------------------------------------------ fixed design constants
CLAIMED_SIGMA = 1.0
CLAIMED_CLIP = 1.0
BATCH = dpi.BATCH            # 8 records per client, one full-batch step
LR = dpi.LR
ATTACK_U = 10                # any-label candidates, the setting where the attack tracks the bound most closely
ALPHA = 0.01                 # level of the white-box variance test (two-sided) and of D2, fixed before any run
BB_LEVEL = 0.90              # two-sided CP level, so the lower limit is a one-sided 95 percent limit
BLOCK_TRIALS = 60            # real Opacus steps per audit for the black-box attack audit
WB_RUNS = 10                 # real Opacus runs per audit for the white-box audit
CLIP_PROBES = 3              # single-record probes per audit for the clipping-norm audit
CLIP_TOL = 1e-4              # relative slack on the clip norm (Opacus adds 1e-6 to the norm)
NO_CLIP_NORM = 1e6
N_AUDITS = 200
V_MIN = 0.05                 # D2 uses coordinates whose rounding variance is at least this (stated, not tuned)
SIM_CLIENTS = 4
SIM_ROUNDS = 2
MAX_WEIGHT = 1000.0
SEED_TAG_DP, SEED_TAG_SIM = 815, 1815

# name -> (noise factor, clipping skipped, with the black-box attack and the full block of trials)
DP_CONDITIONS = OrderedDict([
    ("control", (1.0, False, True)),
    ("noise_x0.25", (0.25, False, True)),
    ("noise_x0.5", (0.5, False, True)),
    ("noise_x2", (2.0, False, True)),
    ("clip_skipped", (1.0, True, True)),
    ("noise_x0.9", (0.9, False, False)),     # sensitivity rows, white-box only
    ("noise_x1.1", (1.1, False, False)),
])
CORE_DP = ("control", "noise_x0.25", "noise_x0.5", "noise_x2", "clip_skipped")
SIM_CONDITIONS = ("control", "control_dropout", "plaintext_fallback", "stale_model", "dropout_cached_update")


def check_audit_index(audit: int, *, tuning: bool = False) -> None:
    if audit < 0:
        raise ValueError("audit index must be non-negative")
    if not tuning and audit >= TUNING_SEED_BASE:
        raise ValueError(f"audit {audit} is a tuning index (>= {TUNING_SEED_BASE})")
    if tuning and audit < TUNING_SEED_BASE:
        raise ValueError("tuning audits must use indices >= 9000")


# ================================================================== statistics shared by the audits
def chi2_variance_audit(sum_sq: float, dof: int, alpha: float = ALPHA) -> dict:
    """Two-sided chi-square test that standardised residuals have variance 1 (known mean 0).
    `sum_sq` is the sum of squared standardised residuals over `dof` independent values."""
    if dof <= 0:
        raise ValueError("dof must be positive")
    lo, hi = stats.chi2.ppf(alpha / 2, dof), stats.chi2.ppf(1 - alpha / 2, dof)
    p = float(2 * min(stats.chi2.cdf(sum_sq, dof), stats.chi2.sf(sum_sq, dof)))
    return {"std_ratio": float(math.sqrt(sum_sq / dof)), "p_two_sided": min(1.0, p),
            "flagged": bool(sum_sq < lo or sum_sq > hi), "alpha": alpha}


def bb_lower_limit(successes: int, trials: int, level: float = BB_LEVEL) -> float:
    """One-sided 95 percent lower Clopper-Pearson limit (the lower end of the two-sided 90 percent interval)."""
    return clopper_pearson_interval(int(successes), int(trials), level)[0]


def clip_audit(norms, clip: float = CLAIMED_CLIP, tol: float = CLIP_TOL) -> dict:
    m = float(max(norms))
    return {"max_norm_over_clip": m / clip, "flagged": bool(m > clip * (1 + tol))}


# ================================================================== DP faults on the real Opacus step
def dp_actual_params(cond: str) -> tuple[float, float]:
    """(noise multiplier, max grad norm) handed to the real Opacus engine for a condition."""
    factor, skip, _ = DP_CONDITIONS[cond]
    if skip:
        # Clipping is inert (1e6) and the noise std nm * max_norm stays at the claimed sigma * C.
        return CLAIMED_SIGMA * factor * CLAIMED_CLIP / NO_CLIP_NORM, NO_CLIP_NORM
    return CLAIMED_SIGMA * factor, CLAIMED_CLIP


def build_dp_trial(audit: int, t: int, y_all: np.ndarray, *, u: int = ATTACK_U, tuning: bool = False) -> dict:
    check_audit_index(audit, tuning=tuning)
    rng = np.random.default_rng(np.random.SeedSequence([audit, t, SEED_TAG_DP]))
    n = len(y_all)
    client = rng.choice(n, size=BATCH, replace=False)
    slot = int(rng.integers(BATCH))
    true_id = int(client[slot])
    mask = np.zeros(n, dtype=bool)
    mask[client] = True
    pool = np.flatnonzero(~mask)
    others = rng.permutation(pool)[:u - 1]
    ids = rng.permutation(np.concatenate([others, [true_id]]).astype(np.int64))
    return {"client": client.astype(np.int64), "slot": slot, "true_id": true_id, "cand_ids": ids,
            "true_idx": int(np.flatnonzero(ids == true_id)[0]), "init_seed": int(rng.integers(2 ** 31 - 1))}


def opacus_seed(audit: int, t: int, cond: str, tag: int = 909) -> int:
    code = list(DP_CONDITIONS).index(cond)
    return int(np.random.SeedSequence([audit, t, code, tag]).generate_state(1)[0] % (2 ** 31 - 1))


def run_dp_audit(audit: int, cond: str, x_all, y_all, *, wb_runs: int = WB_RUNS, block: int = BLOCK_TRIALS,
                 probes: int = CLIP_PROBES, tuning: bool = False) -> dict:
    """One audit of one condition. Everything comes from real Opacus steps whose parameters are
    those of the condition. The auditor works from the CLAIMED sigma and C."""
    torch.set_num_threads(1)
    nm, max_norm = dp_actual_params(cond)
    with_attack = DP_CONDITIONS[cond][2]
    n_trials = block if with_attack else wb_runs
    sumsq, zsum, ip_k, dist_k, d = [], [], 0, 0, 0
    for t in range(n_trials):
        tr = build_dp_trial(audit, t, y_all, tuning=tuning)
        init = dpi.init_state(tr["init_seed"])
        xb, yb = x_all[tr["client"]], y_all[tr["client"]]
        cg = dpi.clipped_gradients(init, xb, yb, CLAIMED_CLIP)          # hand computation with the claimed C
        out = dpi.opacus_noisy_sum(init, xb, yb, sigma=nm, clip=max_norm, lr=LR, seed=opacus_seed(audit, t, cond))
        if t < wb_runs:
            z = (out - cg.sum(axis=0)) / (CLAIMED_SIGMA * CLAIMED_CLIP)
            d = int(z.size)
            sumsq.append(float((z ** 2).sum()))
            zsum.append(float(z.sum()))
        if with_attack:
            known = cg.sum(axis=0) - cg[tr["slot"]]
            residual = out - known
            cand = dpi.clipped_gradients(init, x_all[tr["cand_ids"]], y_all[tr["cand_ids"]], CLAIMED_CLIP)
            ip_k += int(dpi.ip_choice(residual, cand) == tr["true_idx"])
            dist_k += int(dpi.nearest_choice(residual, cand) == tr["true_idx"])
    # clipping-norm audit: the same engine configuration, noise switched off, one record per step
    tr0 = build_dp_trial(audit, 0, y_all, tuning=tuning)
    init0 = dpi.init_state(tr0["init_seed"])
    norms = []
    for j in range(probes):
        rid = int(tr0["client"][j])
        o = dpi.opacus_noisy_sum(init0, x_all[[rid]], y_all[[rid]], sigma=0.0, clip=max_norm, lr=LR, seed=j)
        norms.append(float(np.linalg.norm(o)))
    return {"audit": audit, "cond": cond, "d": d, "wb_sumsq": sumsq, "wb_sum": zsum,
            "bb_trials": n_trials if with_attack else 0, "ip": ip_k, "dist": dist_k, "clip_norms": norms}


def dp_decisions(rec: dict, *, alpha: float = ALPHA) -> dict:
    """Detector decisions for one audit record. Uses only the claimed sigma and C."""
    out = {}
    wb = chi2_variance_audit(sum(rec["wb_sumsq"]), rec["d"] * len(rec["wb_sumsq"]), alpha)
    out["wb_noise"] = wb
    out["wb_clip"] = clip_audit(rec["clip_norms"])
    out["wb_any"] = bool(wb["flagged"] or out["wb_clip"]["flagged"])
    if rec["bb_trials"]:
        bound = dpi.rero_bound(ATTACK_U, CLAIMED_SIGMA)
        lo = bb_lower_limit(rec["ip"], rec["bb_trials"])
        out["bb"] = {"rate": rec["ip"] / rec["bb_trials"], "cp_lower": lo, "bound": bound, "flagged": bool(lo > bound)}
    return out


# ================================================================== SecAgg+ simulation with the real quantiser
def secagg_quant_moments(vectors, sizes, max_weight: float = MAX_WEIGHT) -> dict:
    """Exact mean, standard deviation and fourth moment of the SecAgg+ output under the upstream
    stochastic rounding. Client side arithmetic is repeated in the dtype of the vectors, as in
    `flwr.common.secure_aggregation.quantization.quantize`."""
    n = len(vectors)
    qs = [round(m / max_weight * QUANT_RANGE) for m in sizes]
    total = sum(qs)
    quantiser = QUANT_RANGE / (2 * CLIP_RANGE)
    sum_x = 0.0
    var = m4 = var_sq = 0.0
    for v, q in zip(vectors, qs):
        x = (np.clip(v * (q / QUANT_RANGE), -CLIP_RANGE, CLIP_RANGE) + CLIP_RANGE) * quantiser
        x64 = np.asarray(x, dtype=np.float64)
        phi = x64 - np.floor(x64)
        vi = phi * (1 - phi)
        sum_x = sum_x + x64
        var = var + vi
        var_sq = var_sq + vi ** 2
        m4 = m4 + vi * (phi ** 3 + (1 - phi) ** 3)
    e4 = m4 + 3 * (var ** 2 - var_sq)
    mean = (sum_x * (2 * CLIP_RANGE / QUANT_RANGE) - n * CLIP_RANGE) * (QUANT_RANGE / total)
    return {"mean": mean, "var": var, "e4": e4, "scale": 2 * CLIP_RANGE / total, "total_q": total, "n": n}


def d1_bound(vectors, sizes, max_weight: float = MAX_WEIGHT) -> float:
    """Proven worst case |A_j - exact weighted mean_j| (docs/proofs/one-round-secagg-error.md, theorem
    parts (a) and (c1)): 2cn/T + 2^-47 n Q / T for the quantiser and n (1 + 2^-30) R / (2T) for the
    rounding of the integer weights, where R is the largest spread of the clients' values."""
    n = len(vectors)
    qs = [round(m / max_weight * QUANT_RANGE) for m in sizes]
    total = sum(qs)
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    spread = float((stack.max(axis=0) - stack.min(axis=0)).max())
    return (2 * CLIP_RANGE * n / total + 2.0 ** -47 * n * QUANT_RANGE / total
            + n * (1 + 2.0 ** -30) * spread / (2 * total))


def d2_statistic(agg: np.ndarray, vectors, sizes, max_weight: float = MAX_WEIGHT) -> dict:
    """Standardised test that the aggregate has the error distribution of upstream SecAgg+.
    Z = (sum z_j^2 - k) / sqrt(sum Var(z_j^2)) over coordinates with rounding variance >= V_MIN.
    Honest: Z about N(0, 1). A plaintext aggregate has almost no error so Z is very negative.
    A wrong contribution gives a very large Z."""
    mom = secagg_quant_moments(vectors, sizes, max_weight)
    keep = mom["var"] >= V_MIN
    z = (np.asarray(agg, dtype=np.float64) - mom["mean"])[keep] / (mom["scale"] * np.sqrt(mom["var"][keep]))
    kurt = mom["e4"][keep] / mom["var"][keep] ** 2
    k = int(keep.sum())
    stat = float((z ** 2).sum() - k) / math.sqrt(float((kurt - 1).sum()))
    return {"coords": k, "z_stat": stat, "mean_z2": float((z ** 2).mean())}


def _flat32(state: dict) -> np.ndarray:
    return flatten(state).astype(np.float32)


def secagg_state_aggregator(rng_seed: int):
    """Fresh rounding draws on every call (upstream draws fresh randomness every round)."""
    calls = [0]

    def agg(states, sizes):
        like = states[0]
        vec = secagg_aggregate([_flat32(s) for s in states], list(sizes), rng_seed=rng_seed + calls[0])
        calls[0] += 1
        return OrderedDict((k, v.float()) for k, v in unflatten(vec, like).items())
    return agg


def plaintext_state_aggregator():
    """What a silent fallback computes: the plain weighted mean in float32, Flower FedAvg style."""
    def agg(states, sizes):
        total = np.float32(sum(sizes))
        vec = sum(_flat32(s) * np.float32(n) for s, n in zip(states, sizes)) / total
        return OrderedDict((k, v.float()) for k, v in unflatten(vec.astype(np.float64), states[0]).items())
    return agg


def exact_state_aggregator():
    return fedavg_weighted


def train_client(init_state: dict, x, y, *, seed: int, epochs: int = 1, batch_size: int = 32, lr: float = 0.15,
                 features: int = 64, classes: int = 10) -> OrderedDict:
    model = DigitMLP(features, classes)
    model.load_state_dict(init_state)
    train_local(model, x, y, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


def simulate_chain(parts, init_state, *, participation, aggregator, stale_client=None, cached=None,
                   cfg_seed: int = 0, lr: float = 0.15, batch_size: int = 32, hooks=None):
    """Multi-round FedAvg on fixed client data.

    parts: list of (x, y) per client. participation: list of client tuples per round.
    stale_client: from round 1 on this client trains from the round 0 global (F5).
    cached: if True the clients that dropped in a round are still aggregated with their update
    from round 0 (F4c).
    hooks: optional callable(r, submitted, honest, intended) called each round with state dicts.
    Returns (final_state, per_round_records)."""
    history = [OrderedDict((k, v.clone()) for k, v in init_state.items())]
    first_round: dict = {}
    records = []
    for r, subset in enumerate(participation):
        honest, submitted = {}, {}
        for c in subset:
            x, y = parts[c]
            honest[c] = train_client(history[-1], x, y, seed=cfg_seed + c, lr=lr, batch_size=batch_size)
            if stale_client is not None and c == stale_client and r >= 1:
                submitted[c] = train_client(history[0], x, y, seed=cfg_seed + c, lr=lr, batch_size=batch_size)
            else:
                submitted[c] = honest[c]
        if r == 0:
            first_round = dict(submitted)
        agg_ids = list(subset)
        if cached and r >= 1:
            agg_ids = sorted(set(agg_ids) | {c for c in first_round if c not in subset})
        used = {c: (submitted[c] if c in submitted else first_round[c]) for c in agg_ids}
        sizes = [len(parts[c][1]) for c in agg_ids]
        new = aggregator([used[c] for c in agg_ids], sizes)
        records.append({"round": r, "intended": list(subset), "aggregated": agg_ids,
                        "honest": honest, "submitted": submitted, "used": used, "sizes": sizes, "new": new})
        history.append(new)
    return history[-1], records


def sim_parts(audit: int, x_all, y_all, *, n_clients: int = SIM_CLIENTS, tuning: bool = False):
    check_audit_index(audit, tuning=tuning)
    rng = np.random.default_rng(np.random.SeedSequence([audit, SEED_TAG_SIM]))
    sizes = rng.integers(250, 441, size=n_clients)
    perm = rng.permutation(len(y_all))
    parts, pos = [], 0
    for s in sizes:
        idx = perm[pos:pos + int(s)]
        pos += int(s)
        parts.append((x_all[idx], y_all[idx]))
    init_seed = int(rng.integers(2 ** 31 - 1))
    torch.manual_seed(init_seed)
    init = OrderedDict((k, v.detach().clone()) for k, v in DigitMLP(64, 10).state_dict().items())
    return parts, init


def sim_participation(cond: str, n_clients: int = SIM_CLIENTS) -> list:
    full = tuple(range(n_clients))
    if cond in ("control_dropout", "dropout_cached_update"):
        return [full, full[:-1]]
    return [full] * SIM_ROUNDS


def run_sim_trial(audit: int, cond: str, x_all, y_all, *, tuning: bool = False, alpha: float = ALPHA) -> dict:
    """One simulated multi-round SecAgg+ experiment with the real upstream quantiser."""
    per_round_bound = _verifier("verify_secagg_native").per_round_bound
    torch.set_num_threads(1)
    parts, init = sim_parts(audit, x_all, y_all, tuning=tuning)
    part = sim_participation(cond)
    agg_seed = (audit % 1_000_000) * 100
    secagg = secagg_state_aggregator(agg_seed)
    aggregator = plaintext_state_aggregator() if cond == "plaintext_fallback" else secagg
    final, recs = simulate_chain(parts, init, participation=part, aggregator=aggregator,
                                 stale_client=1 if cond == "stale_model" else None,
                                 cached=(cond == "dropout_cached_update"), cfg_seed=audit % 1_000_000)
    # the independent honest reference (what the repository verifiers recompute with plain PyTorch)
    ref, _ = simulate_chain(parts, init, participation=part, aggregator=exact_state_aggregator(),
                            cfg_seed=audit % 1_000_000)
    z_crit = float(stats.norm.ppf(1 - (alpha / len(part)) / 2))
    rounds = []
    for rec in recs:
        ids = rec["intended"]
        honest_vecs = [_flat32(rec["honest"][c]) for c in ids]
        sizes = [len(parts[c][1]) for c in ids]
        agg_vec = flatten(rec["new"])
        total = float(sum(sizes))
        exact = sum(np.asarray(v, dtype=np.float64) * n for v, n in zip(honest_vecs, sizes)) / total
        err = float(np.abs(agg_vec - exact).max())
        bound = d1_bound(honest_vecs, sizes)
        d2 = d2_statistic(agg_vec, honest_vecs, sizes)
        rounds.append({"round": rec["round"], "n": len(ids), "max_err": err, "bound": bound,
                       "err_over_bound": err / bound, "d1": bool(err > bound), "z_stat": d2["z_stat"],
                       "mean_z2": d2["mean_z2"], "d2": bool(abs(d2["z_stat"]) > z_crit)})
    sizes_all = [len(p[1]) for p in parts]
    gap = float(np.abs(flatten(final) - flatten(ref)).max())
    tol = sum(per_round_bound(len(ids), sum(len(parts[c][1]) for c in ids), MAX_WEIGHT) for ids in part)
    return {"audit": audit, "cond": cond, "rounds": rounds, "d1": any(r["d1"] for r in rounds),
            "d2": any(r["d2"] for r in rounds), "final_gap": gap, "naive_tolerance": tol,
            "naive_gate": bool(gap > tol), "sizes": [int(s) for s in sizes_all]}


# ================================================================== evidence copies and gates
def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def safe_tamper_root(path: Path) -> Path:
    """Tampered evidence may only live under the system temporary directory or under the fault
    matrix output directory. Saved evidence directories are refused."""
    path = Path(path)
    if _under(path, SAVED_EVIDENCE) and not _under(path, OUT_DIR):
        raise PermissionError(f"refusing to write tampered evidence inside saved evidence: {path}")
    if _under(path, ROOT) and not _under(path, OUT_DIR):
        raise PermissionError(f"tampered evidence must live under /tmp or the fault matrix output dir: {path}")
    return path


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> None:
    safe_tamper_root(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def save_model(path: Path, state: dict) -> None:
    safe_tamper_root(path)
    torch.save(OrderedDict((k, v.detach().to(torch.float64)) for k, v in state.items()), path)


def run_script(script: str, args: list, *, cwd: Path = ROOT, timeout: int = 420) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _verifier(name: str):
    """Import a verifier from `scripts/` (they are scripts, not a package)."""
    import importlib
    p = str(ROOT / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module(name)


# ================================================================== evidence tampering for the repository gates
SECAGG_SAVED = SAVED_EVIDENCE / "secagg"
NATIVE_SAVED = SAVED_EVIDENCE / "native_flower" / "v014"
DATA4 = ROOT / "data" / "training" / "digits-docker-iid-4c-v014"
DATA3 = ROOT / "data" / "training" / "digits-docker-iid-v012"
GATE_CFG = {"seed": 20261118, "local_epochs": 1, "batch_size": 32, "learning_rate": 0.15}
FULL4, SURV3, SURV2 = (0, 1, 2, 3), (0, 1, 2), (0, 1)
# matrix tree run -> (participation per round, data dir)
MATRIX_RUNS = {
    "n4-t2-kill1-r2": ([FULL4, SURV3], DATA4), "n4-t3-kill1-r2": ([FULL4, SURV3], DATA4),
    "n4-t2-kill2-r2": ([FULL4, SURV2], DATA4), "n3-t2-kill1-r2": ([SURV3, SURV2], DATA3),
    "n4-1round-secagg": ([FULL4], DATA4),
}
TAMPER_VARIANTS = ("control", "plaintext_fallback", "plaintext_fallback_no_markers", "stale_model",
                   "wrong_threshold_low", "wrong_threshold_strict", "dropout_cached_update")


def load_parts(data_dir: Path):
    from feddisclosure.training.datasets import load_site_partition
    manifest = json.loads((data_dir / "manifest.json").read_text())
    parts = [load_site_partition(data_dir / p["file"]) for p in manifest["client_partitions"]]
    torch.manual_seed(GATE_CFG["seed"])
    init = OrderedDict((k, v.detach().clone()) for k, v in
                       DigitMLP(manifest["num_features"], manifest["num_classes"]).state_dict().items())
    return parts, init


def tampered_model(mode: str, data_dir: Path, participation: list):
    """Final model of a faulty run, computed with the real upstream quantiser (or the plain mean)."""
    parts, init = load_parts(data_dir)
    if mode == "plaintext":
        agg, kw = plaintext_state_aggregator(), {}
    elif mode == "stale":
        agg, kw = secagg_state_aggregator(4242), {"stale_client": 1}
    elif mode == "cached":
        agg, kw = secagg_state_aggregator(4242), {"cached": True}
    else:
        raise ValueError(mode)
    final, _ = simulate_chain(parts, init, participation=participation, aggregator=agg, cfg_seed=GATE_CFG["seed"], **kw)
    return final


def _log(entries: list, path: Path, before: str | None, what: str) -> None:
    entries.append({"file": str(path), "what": what, "sha256_before": before,
                    "sha256_after": sha256_file(path) if path.exists() else None})


def _rewrite_model(path: Path, mode: str, data_dir: Path, participation: list, entries: list, what: str) -> None:
    before = sha256_file(path) if path.exists() else None
    save_model(path, tampered_model(mode, data_dir, participation))
    _log(entries, path, before, what)


def _strip_markers(run: Path, entries: list) -> None:
    log = run / "flower_cli_stdout.log"
    if log.is_file():
        before = sha256_file(log)
        kept = [l for l in log.read_text().splitlines() if "Secure aggregation" not in l]
        log.write_text("\n".join(kept) + "\n")
        _log(entries, log, before, "removed the Secure aggregation lines, as a run that never entered the workflow")
    marker = run / "server" / "secagg_workflow_returned.json"
    if marker.is_file():
        before = sha256_file(marker)
        marker.unlink()
        entries.append({"file": str(marker), "what": "deleted, the workflow never ran", "sha256_before": before,
                        "sha256_after": None})


def _copy_run(src: Path, dst: Path) -> None:
    safe_tamper_root(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def build_matrix_tree(variant: str, work: Path) -> tuple[Path, list]:
    """Copy of the evidence that scripts/verify_secagg_dropout_matrix.py reads, with the fault applied.
    Layout: <root>/v014_dropout_matrix, <root>/v014_n4/n4-t2-baseline, <root>/v014_n4_replicates."""
    root = safe_tamper_root(Path(work) / variant / "matrix")
    if root.exists():
        shutil.rmtree(root)
    copy_tree(SECAGG_SAVED / "v014_dropout_matrix", root / "v014_dropout_matrix")
    copy_tree(SECAGG_SAVED / "v014_n4" / "n4-t2-baseline", root / "v014_n4" / "n4-t2-baseline")
    copy_tree(SECAGG_SAVED / "v014_n4_replicates", root / "v014_n4_replicates")
    m = root / "v014_dropout_matrix"
    entries: list = []
    two_round = [FULL4, FULL4]
    runs = {k: (m / k, v[0], v[1]) for k, v in MATRIX_RUNS.items()}
    runs["baseline"] = (root / "v014_n4" / "n4-t2-baseline", two_round, DATA4)
    for rep in sorted((root / "v014_n4_replicates").glob("rep*")):
        runs[rep.name] = (rep, two_round, DATA4)
    if variant in ("plaintext_fallback", "plaintext_fallback_no_markers", "stale_model", "dropout_cached_update"):
        mode = {"plaintext_fallback": "plaintext", "plaintext_fallback_no_markers": "plaintext",
                "stale_model": "stale", "dropout_cached_update": "cached"}[variant]
        for name, (run, part, data) in runs.items():
            if not (run / "server" / "final_model.pt").is_file():
                continue
            if mode == "stale" and len(part) < 2:
                continue            # a stale model needs a second round
            if mode == "cached" and (len(part) < 2 or len(part[1]) == len(part[0])):
                continue            # only runs that had a dropout
            _rewrite_model(run / "server" / "final_model.pt", mode, data, part, entries,
                           f"{name}: final model replaced by the {variant} result")
            if variant == "plaintext_fallback_no_markers":
                _strip_markers(run, entries)
    elif variant == "wrong_threshold_low":
        target, donor = m / "n4-t3-kill2-r2", m / "n4-t2-kill2-r2"
        for name in ("flower_cli_stdout.log",):
            before = sha256_file(target / name)
            shutil.copy(donor / name, target / name)
            _log(entries, target / name, before, "log of a run that tolerated the loss of 2 of 4 clients at threshold 3")
        for name in ("final_model.pt", "flower_summary.json", "secagg_workflow_returned.json"):
            shutil.copy(donor / "server" / name, target / "server" / name)
            _log(entries, target / "server" / name, None, "copied from a tolerated run")
        marker = target / "server" / "secagg_workflow_returned.json"
        mk = json.loads(marker.read_text())
        if "reconstruction_threshold" in mk:
            mk["reconstruction_threshold"] = 3          # the faulty system still reports the requested threshold
        marker.write_text(json.dumps(mk, indent=2) + "\n")
        fr = json.loads((target / "server" / "flower_run.json").read_text())
        fr.update(final_model_exists=True, summary_exists=True)
        (target / "server" / "flower_run.json").write_text(json.dumps(fr, indent=2) + "\n")
        _log(entries, target / "server" / "flower_run.json", None, "final_model_exists and summary_exists set")
    elif variant == "wrong_threshold_strict":
        donor = m / "n4-t3-kill2-r2"
        donor_log = (donor / "flower_cli_stdout.log").read_text()
        for name in ("n4-t2-kill2-r2", "n4-t3-kill1-r2"):      # both lose exactly threshold-many survivors
            target = m / name
            before = sha256_file(target / "flower_cli_stdout.log")
            (target / "flower_cli_stdout.log").write_text(donor_log)
            _log(entries, target / "flower_cli_stdout.log", before, "log of a halted run (survivors equal the threshold)")
            for f in ("final_model.pt", "flower_summary.json"):
                (target / "server" / f).unlink(missing_ok=True)
            fr = json.loads((target / "server" / "flower_run.json").read_text())
            fr.update(final_model_exists=False, summary_exists=False)
            (target / "server" / "flower_run.json").write_text(json.dumps(fr, indent=2) + "\n")
            _log(entries, target / "server" / "flower_run.json", None, "no final model, as after a halt")
    elif variant != "control":
        raise ValueError(variant)
    return root, entries


def matrix_channels(summary: dict) -> dict:
    """Pass or fail fields the script itself computes. True means the gate's own check passed."""
    ch = {}
    for name, v in summary["scenarios"].items():
        ch[f"rule:{name}"] = bool(v["prediction_matches_outcome"])
        if "vs_independent_pytorch" in v:
            ch[f"tolerance:{name}"] = bool(v["vs_independent_pytorch"]["within_tolerance"])
    if "n4_full_participation_equivalence" in summary:
        ch["n4_full_participation_equivalence"] = bool(summary["n4_full_participation_equivalence"]["within_tolerance"])
    if "n4_one_round_bound_check" in summary:
        ch["n4_one_round_bound"] = bool(summary["n4_one_round_bound_check"]["within_bound"])
    if "n4_two_round_replicates" in summary:
        for r in summary["n4_two_round_replicates"]["max_abs_diff_vs_independent_pytorch"]:
            tol = summary["n4_two_round_replicates"]["naive_two_round_tolerance_2x_bound"]
            ch[f"replicate:{r}"] = bool(r not in summary["n4_two_round_replicates"]["runs_above_naive_tolerance"])
            del tol
    return ch


def run_matrix_gate(variant: str, work: Path) -> dict:
    root, entries = build_matrix_tree(variant, work)
    out = root / "summary.json"
    cp = run_script("verify_secagg_dropout_matrix.py", ["--matrix", root / "v014_dropout_matrix", "--baseline",
                    root / "v014_n4" / "n4-t2-baseline", "--output", out])
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr[-2000:])
    summary = json.loads(out.read_text())
    scen = {k: {f: v.get(f) for f in ("protocol_outcome", "results_per_round", "prediction_matches_outcome",
                                       "secure_aggregation_completed_lines", "workflow_engine",
                                       "secure_aggregation_enabled")} for k, v in summary["scenarios"].items()}
    return {"script": "verify_secagg_dropout_matrix.py", "exit_code": cp.returncode,
            "channels": matrix_channels(summary), "scenario_report_fields": scen,
            "tamper_log": entries,
            "max_abs_diffs": {k: v["vs_independent_pytorch"]["max_abs_diff"]
                              for k, v in summary["scenarios"].items() if "vs_independent_pytorch" in v},
            "n4_full": summary.get("n4_full_participation_equivalence"),
            "one_round": summary.get("n4_one_round_bound_check")}


def native_channels(report: dict) -> dict:
    ch = {}
    if "equivalence_2_rounds" in report:
        ch["equivalence_2_rounds"] = bool(report["equivalence_2_rounds"]["within_tolerance"])
    for k, v in report.get("single_round_error_vs_bound", {}).items():
        ch[f"single_round:{k}"] = bool(v["within_bound"])
    for k, v in report["scenarios"].items():
        if "matches_a_participation_within_tolerance" in v:
            ch[f"dropout_match:{k}"] = bool(v["matches_a_participation_within_tolerance"])
    return ch


def run_native_gate(variant: str, work: Path) -> dict | None:
    """scripts/verify_secagg_native.py on a copy of reports/results/secagg/v014 (three clients)."""
    if variant not in ("control", "plaintext_fallback", "plaintext_fallback_no_markers", "stale_model"):
        return None
    root = safe_tamper_root(Path(work) / variant / "native")
    copy_tree(SECAGG_SAVED / "v014", root / "v014")
    entries: list = []
    base = root / "v014"
    if variant != "control":
        mode = "stale" if variant == "stale_model" else "plaintext"
        two = [SURV3, SURV3]
        targets = [("secagg-on-t2-n3", two)]
        if mode == "plaintext":
            targets += [("secagg-1round-collusion", [SURV3]), ("secagg-1round-maxweight200", [SURV3])]
        for name, part in targets:
            _rewrite_model(base / name / "server" / "final_model.pt", mode, DATA3, part, entries,
                           f"{name}: final model replaced by the {variant} result")
            if variant == "plaintext_fallback_no_markers":
                _strip_markers(base / name, entries)
    out = root / "report.json"
    cp = run_script("verify_secagg_native.py", ["--base", base, "--output", out])
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr[-2000:])
    rep = json.loads(out.read_text())
    flags = {k: {f: v.get(f) for f in ("secure_aggregation_enabled", "secure_aggregation_completed_lines",
                                        "workflow_engine")} for k, v in rep["scenarios"].items()
             if k == "secagg-on-t2-n3"}
    return {"script": "verify_secagg_native.py", "channels": native_channels(rep), "tamper_log": entries,
            "report_fields_for_secagg_on_t2_n3": flags,
            "equivalence_2_rounds": rep.get("equivalence_2_rounds")}


def run_parity_gate(variant: str, work: Path) -> dict | None:
    """verify_flower_parity.certify on a copy of the saved native plain FedAvg run (float64 reference, 1e-5)."""
    if variant not in ("control", "stale_model"):
        return None
    root = safe_tamper_root(Path(work) / variant / "parity")
    copy_tree(NATIVE_SAVED / "run-iid-d", root / "run-iid-d")
    run = root / "run-iid-d"
    entries: list = []
    if variant == "stale_model":
        parts, init = load_parts(DATA3)
        final, _ = simulate_chain(parts, init, participation=[SURV3, SURV3], aggregator=exact_state_aggregator(),
                                  stale_client=1, cfg_seed=GATE_CFG["seed"])
        before = sha256_file(run / "server" / "final_model.pt")
        torch.save(OrderedDict((k, v.detach().clone()) for k, v in final.items()), run / "server" / "final_model.pt")
        _log(entries, run / "server" / "final_model.pt", before, "final model of a run where client 1 trained from a stale global model in round 2")
    pv = _verifier("verify_flower_parity")
    try:
        rep = pv.certify(run / "server", run / "torch_reference.json",
                         run / "checkpoints" / "digits-iid-smoke-v1_final.pt", DATA3 / "manifest.json",
                         require_deployment=True)
        ok, worst, err = bool(rep["verified_numerical_parity"]), max(
            v["max_abs_diff"] for v in rep["layers"].values()), None
    except ValueError as exc:
        ok, worst, err = False, None, str(exc)
    return {"script": "verify_flower_parity.py", "channels": {"numerical_parity": ok}, "max_abs_diff": worst,
            "error": err, "tamper_log": entries}


def run_topology_gate() -> dict:
    """Static topology certification of the real compose files. No fault here changes a compose file."""
    tv = _verifier("verify_flower_topology")
    out = {}
    for name in ("compose.flower.yaml", "compose.flower.n4.yaml"):
        try:
            tv.static_certify(ROOT / name)
            out[name] = True
        except Exception as exc:  # noqa: BLE001
            out[name] = f"fails: {exc}"
    return out


def flagged_vs_control(channels: dict, control: dict) -> dict:
    """Channels that pass on the honest copy and fail on the tampered copy. A channel that already
    fails on honest evidence cannot discriminate and is listed separately."""
    newly = sorted(k for k, v in channels.items() if v is False and control.get(k) is True)
    uninformative = sorted(k for k, v in control.items() if v is False)
    return {"flagged": bool(newly), "newly_failing_channels": newly, "channels_failing_on_honest_evidence": uninformative}


def saved_evidence_digest() -> dict:
    return {"secagg": tree_digest(SECAGG_SAVED), "native_flower_v014": tree_digest(NATIVE_SAVED),
            "data_4c": tree_digest(DATA4), "data_3c": tree_digest(DATA3)}


def run_all_gates(work: Path) -> dict:
    before = saved_evidence_digest()
    res: dict = {"variants": {}}
    controls: dict = {}
    for v in TAMPER_VARIANTS:
        entry = {"matrix": run_matrix_gate(v, work), "native": run_native_gate(v, work), "parity": run_parity_gate(v, work)}
        if v == "control":
            controls = {k: (e["channels"] if e else None) for k, e in entry.items()}
        for k, e in entry.items():
            if e and controls.get(k) is not None:
                e["verdict"] = flagged_vs_control(e["channels"], controls[k])
        any_flag = any(e["verdict"]["flagged"] for e in entry.values() if e and "verdict" in e)
        entry["any_existing_gate_flags"] = bool(any_flag) if v != "control" else None
        res["variants"][v] = entry
    res["topology_static_gate"] = run_topology_gate()
    after = saved_evidence_digest()
    res["saved_evidence_unchanged"] = before == after
    res["saved_evidence_digest_before"], res["saved_evidence_digest_after"] = before, after
    return res


# ================================================================== summary, matrix and pre-registered checks
def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    by = {r["audit"]: r for r in out}
    return [by[k] for k in sorted(by)]


def _rate(flags) -> dict:
    return rate_summary([bool(f) for f in flags])


def summarize_dp(out_dir: Path, expected: int = N_AUDITS) -> dict:
    res = {}
    for cond in DP_CONDITIONS:
        recs = read_jsonl(Path(out_dir) / "dp" / f"{cond}.jsonl")
        if expected and len(recs) != expected:
            raise RuntimeError(f"{cond}: {len(recs)} audits found, {expected} expected")
        dec = [dp_decisions(r) for r in recs]
        row = {"audits": len(recs), "wb_noise": _rate([d["wb_noise"]["flagged"] for d in dec]),
               "wb_clip": _rate([d["wb_clip"]["flagged"] for d in dec]),
               "wb_any": _rate([d["wb_any"] for d in dec]),
               "noise_std_ratio_estimate": {
                   "mean": float(np.mean([d["wb_noise"]["std_ratio"] for d in dec])),
                   "min": float(np.min([d["wb_noise"]["std_ratio"] for d in dec])),
                   "max": float(np.max([d["wb_noise"]["std_ratio"] for d in dec]))},
               "clip_max_norm_over_clip": {"mean": float(np.mean([d["wb_clip"]["max_norm_over_clip"] for d in dec])),
                                           "min": float(np.min([d["wb_clip"]["max_norm_over_clip"] for d in dec]))}}
        if recs[0]["bb_trials"]:
            ks = [r["ip"] for r in recs]
            n = recs[0]["bb_trials"]
            row["bb"] = _rate([d["bb"]["flagged"] for d in dec])
            row["bb"]["attack_rate_pooled"] = rate_summary([True] * sum(ks) + [False] * (n * len(ks) - sum(ks)))
            row["bb"]["bound_for_claimed_sigma"] = dec[0]["bb"]["bound"]
            row["bb"]["trials_per_audit"] = n
        res[cond] = row
    return res


def summarize_sim(out_dir: Path, expected: int = N_AUDITS) -> dict:
    res = {}
    for cond in SIM_CONDITIONS:
        recs = read_jsonl(Path(out_dir) / "secagg_sim" / f"{cond}.jsonl")
        if expected and len(recs) != expected:
            raise RuntimeError(f"{cond}: {len(recs)} trials found, {expected} expected")
        last = [r["rounds"][-1] for r in recs]
        res[cond] = {"trials": len(recs), "d1": _rate([r["d1"] for r in recs]), "d2": _rate([r["d2"] for r in recs]),
                     "naive_two_round_gate": _rate([r["naive_gate"] for r in recs]),
                     "last_round_err_over_bound": {"median": float(np.median([x["err_over_bound"] for x in last])),
                                                   "min": float(np.min([x["err_over_bound"] for x in last])),
                                                   "max": float(np.max([x["err_over_bound"] for x in last]))},
                     "last_round_d2_z": {"median": float(np.median([x["z_stat"] for x in last])),
                                         "min": float(np.min([x["z_stat"] for x in last])),
                                         "max": float(np.max([x["z_stat"] for x in last]))}}
    return res


POWER_RULE = 0.9
FPR_RULE = 0.05


def _yes(rate: dict | None) -> bool | None:
    return None if rate is None else bool(rate["rate"] >= POWER_RULE)


def build_matrix(dp: dict, sim: dict, gates: dict) -> dict:
    """Rows are fault variants, columns are detectors. Each cell holds the detection rate with its
    Clopper-Pearson interval where randomness applies, or a deterministic yes or no."""
    gv = gates["variants"]

    def det(flag):          # deterministic evidence gate cell
        return {"detected": bool(flag), "deterministic": True}

    def rnd(rate):
        return {"detected": _yes(rate), "rate": rate["rate"], "trials": rate["trials"],
                "clopper_pearson": rate["clopper_pearson"]}

    rows = {}
    for name, cond in (("F1 noise x0.25", "noise_x0.25"), ("F1 noise x0.5", "noise_x0.5"), ("F1 noise x2", "noise_x2"),
                       ("F2 clipping skipped", "clip_skipped")):
        d = dp[cond]
        rows[name] = {"fault": name.split()[0], "A_existing_gates": None, "B_whitebox_noise": rnd(d["wb_noise"]),
                      "B_whitebox_clip": rnd(d["wb_clip"]), "B_whitebox_any": rnd(d["wb_any"]),
                      "C_informed_attack": rnd(d["bb"]), "D1_error_bound": None, "D2_error_distribution": None}
    for name, var, simc in (("F3 plaintext fallback, logs kept", "plaintext_fallback", "plaintext_fallback"),
                            ("F3 plaintext fallback, no workflow markers", "plaintext_fallback_no_markers",
                             "plaintext_fallback"),
                            ("F4 dropout below threshold tolerated", "wrong_threshold_low", None),
                            ("F4 dropout at threshold halts (off by one)", "wrong_threshold_strict", None),
                            ("F4 dropped client's cached update reused", "dropout_cached_update", "dropout_cached_update"),
                            ("F5 stale global model", "stale_model", "stale_model")):
        rows[name] = {"fault": name.split()[0], "A_existing_gates": det(gv[var]["any_existing_gate_flags"]),
                      "A_channels": sorted({c for e in gv[var].values() if isinstance(e, dict) and "verdict" in e
                                            for c in e["verdict"]["newly_failing_channels"]}),
                      "B_whitebox_noise": None, "B_whitebox_clip": None, "B_whitebox_any": None, "C_informed_attack": None,
                      "D1_error_bound": rnd(sim[simc]["d1"]) if simc else None,
                      "D2_error_distribution": rnd(sim[simc]["d2"]) if simc else None}
    return rows


def false_positive_rates(dp: dict, sim: dict, gates: dict) -> dict:
    c, s = dp["control"], sim["control"]
    ctl = gates["variants"]["control"]
    out = {"B_whitebox_noise": c["wb_noise"], "B_whitebox_clip": c["wb_clip"], "B_whitebox_any": c["wb_any"],
           "C_informed_attack": c["bb"], "D1_error_bound": s["d1"], "D2_error_distribution": s["d2"],
           "naive_two_round_tolerance_gate_logic": s["naive_two_round_gate"],
           "control_dropout_D1": sim["control_dropout"]["d1"], "control_dropout_D2": sim["control_dropout"]["d2"],
           "A_existing_gates_on_honest_saved_evidence": {
               "channels_failing": {k: e["verdict"]["channels_failing_on_honest_evidence"] if "verdict" in e else
                                    sorted(c_ for c_, v in e["channels"].items() if v is False)
                                    for k, e in ctl.items() if isinstance(e, dict) and "channels" in e},
               "note": "deterministic, one honest copy. The failing channel is the naive two-round tolerance, "
                       "which the verifier itself records as failed for the baseline run"}}
    return out


def fault_level(matrix: dict, fpr: dict, *, include_d2: bool) -> dict:
    """A fault type counts as detected when EVERY variant of it is detected by at least one detector whose
    false positive rate is within the rule. The evading variants are named."""
    usable = {k: (v["rate"] <= FPR_RULE) for k, v in fpr.items() if isinstance(v, dict) and "rate" in v}
    cols = [("A_existing_gates", True), ("B_whitebox_any", usable["B_whitebox_any"]),
            ("C_informed_attack", usable["C_informed_attack"]), ("D1_error_bound", usable["D1_error_bound"])]
    if include_d2:
        cols.append(("D2_error_distribution", usable["D2_error_distribution"]))
    by_fault: dict = {}
    for name, row in matrix.items():
        hit = [c for c, ok in cols if ok and row.get(c) and row[c].get("detected")]
        by_fault.setdefault(row["fault"], []).append({"variant": name, "detected_by": hit, "detected": bool(hit)})
    detected = {f: all(v["detected"] for v in vs) for f, vs in by_fault.items()}
    return {"by_fault": by_fault, "detected": detected, "n_detected": int(sum(detected.values())),
            "evading_faults": sorted(f for f, ok in detected.items() if not ok),
            "evading_variants": sorted(v["variant"] for vs in by_fault.values() for v in vs if not v["detected"])}


def prereg_check(matrix: dict, fpr: dict, dp: dict) -> dict:
    strict = fault_level(matrix, fpr, include_d2=False)
    ext = fault_level(matrix, fpr, include_d2=True)
    out = {"schema_version": 1, "preregistration": "docs/preregistration/v0.15.md",
           "note": "Thresholds are copied verbatim from the pre-registration and were not changed. The power rule "
                   "is read on the point estimate. The Clopper-Pearson interval is reported next to it."}
    out["F1"] = {"id": "F1", "prediction": "Every seeded fault is caught by at least one detector except possibly mask reuse",
                 "pass_rule": "At least 4 of 5 faults detected, and the evading faults are named in the results",
                 "observed": {"strict_detectors_A_B_C_D1": {k: strict[k] for k in ("detected", "n_detected", "evading_faults", "evading_variants")},
                              "with_D2_extension": {k: ext[k] for k in ("detected", "n_detected", "evading_faults", "evading_variants")},
                              "detail": ext["by_fault"]},
                 "operationalisation": "A fault type is detected only if all of its variants are detected by a detector whose "
                                       "false positive rate under the control is at most 0.05 (deterministic evidence gates count by "
                                       "their yes or no). Strict counts detectors A, B, C and D1 as specified. D2 is an extension "
                                       "added before the run so that a zero-error plaintext aggregate can be seen. Both counts are given. "
                                       "The pass verdict uses the strict count.",
                 "pass": strict["n_detected"] >= 4}
    f2 = {}
    for cond in ("noise_x0.25", "noise_x0.5", "noise_x2"):
        f2[cond] = {"power": dp[cond]["wb_noise"]["rate"], "clopper_pearson": dp[cond]["wb_noise"]["clopper_pearson"],
                    "audits": dp[cond]["audits"]}
    fp = dp["control"]["wb_noise"]
    out["F2"] = {"id": "F2", "prediction": "The many-run DP audit catches a noise scale off by a factor of 2 or more",
                 "pass_rule": "Detection power at least 0.9 with false positive rate at most 0.05",
                 "observed": {"power_by_noise_factor": f2,
                              "false_positive_rate": {"rate": fp["rate"], "clopper_pearson": fp["clopper_pearson"],
                                                      "audits": fp["trials"]},
                              "sensitivity_not_part_of_rule": {c: dp[c]["wb_noise"]["rate"] for c in ("noise_x0.9", "noise_x1.1")}},
                 "operationalisation": "Factors 0.25, 0.5 and 2 on the claimed sigma 1.0. The white-box noise test (chi-square at level 0.01, "
                                       "10 real Opacus runs of 2,410 coordinates per audit) must reach power 0.9 at every factor. "
                                       "The false positive rate is that of the same test under the no-fault control.",
                 "pass": bool(all(v["power"] >= 0.9 for v in f2.values()) and fp["rate"] <= 0.05)}
    out["fault_level_strict"], out["fault_level_with_D2"] = strict, ext
    return out
