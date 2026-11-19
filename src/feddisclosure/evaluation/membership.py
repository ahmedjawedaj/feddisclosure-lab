"""Privileged membership inference scoring. The attacker never imports this module."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def score_membership(scores: np.ndarray, membership: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=float)
    membership = np.asarray(membership, dtype=int)
    if scores.ndim != 1 or scores.shape != membership.shape or not np.isfinite(scores).all():
        raise ValueError("Membership scores must be finite and aligned with evaluator-only labels")
    if set(np.unique(membership)) != {0, 1}:
        raise ValueError("Need both members and nonmembers to calculate ROC")
    fpr, tpr, _ = roc_curve(membership, scores, drop_intermediate=False)
    # Conservative operating-point TPR. Never interpolate to an unobserved FPR
    # to claim actual low-FPR detections on tiny public-digit smoke samples.
    tpr_at_fpr = {str(k): float(max(tpr[fpr <= k], default=0)) for k in (0.01, 0.05, 0.1)}
    return {"roc_auc": float(roc_auc_score(membership, scores)),
            "attack_advantage": float(np.max(tpr - fpr)), "tpr_at_fpr": tpr_at_fpr,
            "num_members": int(membership.sum()), "num_nonmembers": int((1 - membership).sum()),
            "baseline_chance_auc": 0.5,
            "limitations": ["Finite-sample ROC can be unstable; report all results including near-chance attacks",
                            "TPR@FPR is measured at achievable empirical thresholds, not interpolated"]}


def evaluate_membership(scores_file: str | Path, truth_file: str | Path, output: str | Path) -> dict:
    with np.load(scores_file, allow_pickle=False) as pred, np.load(truth_file, allow_pickle=False) as truth:
        if set(pred.files) != {"scores"} or set(truth.files) != {"membership"}:
            raise ValueError("Unexpected attack or evaluator artifact")
        report = score_membership(pred["scores"], truth["membership"])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report
