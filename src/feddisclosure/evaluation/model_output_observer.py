"""Privileged query simulator for OUTPUT-ONLY membership inference evaluation.

Reads public train/test images and a trained classifier; gives the simulated
external model user only probability vectors for the same number of examples
from each source.  Membership truth remains evaluator-only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP


def create_output_observations(checkpoint: str | Path, site_files: list[str | Path], test_file: str | Path,
                               directory: str | Path, *, per_class: int = 6, seed: int = 1234) -> dict:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    train_xs, train_ys = zip(*(load_site_partition(f) for f in site_files))
    train_x = np.concatenate(train_xs)
    train_y = np.concatenate(train_ys)
    test_x, test_y = load_site_partition(test_file)
    if train_x.shape[1:] != test_x.shape[1:]:
        raise ValueError("Training and test feature dimensions differ")
    generator = np.random.default_rng(seed)
    train_indices, test_indices, selected_classes = [], [], []
    for label in sorted(set(train_y.tolist()).intersection(test_y.tolist())):
        members = np.flatnonzero(train_y == label)
        nonmembers = np.flatnonzero(test_y == label)
        n = min(per_class, len(members), len(nonmembers))
        if n:
            train_indices.extend(generator.choice(members, n, replace=False).tolist())
            test_indices.extend(generator.choice(nonmembers, n, replace=False).tolist())
            selected_classes.append({"class": int(label), "per_group": int(n)})
    if not train_indices:
        raise ValueError("No overlapping classes between member and non-member examples")
    batch_x = np.concatenate([train_x[train_indices], test_x[test_indices]])
    truth = np.concatenate([np.ones(len(train_indices), dtype=np.int64),
                            np.zeros(len(test_indices), dtype=np.int64)])
    order = generator.permutation(len(truth))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = int(state["network.2.bias"].numel())
    model = DigitMLP(features=batch_x.shape[1], classes=classes)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(batch_x[order]).float()), dim=1).numpy()
    root = Path(directory)
    (root / "attacker").mkdir(parents=True, exist_ok=True)
    (root / "evaluator_only").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "attacker" / "model_predictions.npz", probabilities=probabilities)
    np.savez_compressed(root / "evaluator_only" / "membership_labels.npz", membership=truth[order])
    contract = {"observer": "external_model_user_confidence_scores_only",
                "attacker_receives": ["probabilities"], "dataset": "approved_public_digits",
                "member_count": len(train_indices), "nonmember_count": len(test_indices),
                "query_count": len(truth), "class_matched": selected_classes,
                "known_labels": False, "observations": "full softmax confidence vectors",
                "limitations": ["Held-out test records are out-of-training distribution by definition, but class-matched",
                                "Balanced query sample is constructed by the evaluator, not a realistic membership prior",
                                "This is an offline replay of prediction responses, not a networked API measurement"]}
    (root / "observation_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract
