"""External model user attack: confidence heuristic on approved model outputs.

No training data, test data or membership labels can be loaded here.
"""
import json
from pathlib import Path
import numpy as np


def confidence_scores(probabilities: np.ndarray, method: str = "max_confidence") -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected probability matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Invalid probability matrix")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Every probability row must sum to one")
    if method == "max_confidence":
        return np.max(probabilities, axis=1)
    if method == "negative_entropy":
        return np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    if method == "top2_margin":
        top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
        return np.abs(top_two[:, 1] - top_two[:, 0])
    raise ValueError(f"Unsupported output-only membership attack: {method}")


def attack_outputs(observation_file: str | Path, directory: str | Path, *, method: str = "max_confidence") -> dict:
    with np.load(observation_file, allow_pickle=False) as obs:
        if set(obs.files) != {"probabilities"}:
            raise ValueError("Attacker receives probabilities only, no member labels or ground truth")
        scores = confidence_scores(obs["probabilities"], method=method)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", scores=scores)
    metadata = {"attack": method + "_membership_heuristic", "observer": "external_model_user_confidence_scores_only",
                "queries_scored": len(scores), "requires_labels": False, "pre_registered_method": method,
                "limitation": "A heuristic rather than an optimal or calibrated attacker; score quality is evaluated separately"}
    (out / "attack_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
