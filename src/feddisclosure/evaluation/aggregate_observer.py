"""Create isolated observations for an IDEALIZED secure-sum threat-model study.

This is NOT a cryptographic protocol or a Flower SecAgg+ run. The evaluator is
privileged to create exact sums from public digit clients; attacker folders
contain only observations allowed under each stated observer role.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from feddisclosure.training.task import DigitMLP
from feddisclosure.training.datasets import load_site_partition

CASES = ("individual", "aggregate_only", "all_but_one_collusion")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_aggregate_observations(site_files: list[str | Path], out_dir: str | Path, *,
                                  examples_per_site: int = 8, seed: int = 241,
                                  learning_rate: float = 0.15, joint_oracle: bool = False) -> dict:
    """One local full-batch SGD step/site; target site 0 and two other contributors.

    Each site must be a different file and use disjoint *source_idx* if present.
    Target labels are deliberately granted to all attack variants as an oracle.
    The colluding observer knows both other sites' individual updates, and can
    algebraically isolate the victim update from the published mean.
    """
    if len(site_files) != 3 or len({str(Path(p).resolve()) for p in site_files}) != 3:
        raise ValueError("Require exactly three distinct institutional files")
    if not 1 <= examples_per_site <= 32 or learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("Invalid experiment configuration")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite research evidence")
    out.mkdir(parents=True, exist_ok=True)

    private_data = []
    all_ids = []
    for i, site_file in enumerate(site_files):
        x, y = load_site_partition(site_file)
        if x.ndim != 2 or x.shape[1] != 64 or len(x) != len(y) or len(x) < examples_per_site:
            raise ValueError("Each institution needs sufficient public 8x8 digits")
        with np.load(site_file, allow_pickle=False) as source:
            if "source_idx" in source.files:
                ids = {int(j) for j in source["source_idx"]}
                if len(ids) != len(x):
                    raise ValueError("Duplicate source indices in a site")
                all_ids.append(ids)
        indices = np.random.default_rng(seed + i).choice(len(y), size=examples_per_site, replace=False)
        private_data.append((torch.tensor(x[indices], dtype=torch.float32),
                             torch.tensor(y[indices], dtype=torch.long), indices))
    if len(all_ids) == 3 and any(all_ids[i] & all_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Client datasets overlap")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DigitMLP()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    deltas = []
    for x, y, _ in private_data:
        local = DigitMLP()
        local.load_state_dict(initial)
        optimizer = torch.optim.SGD(local.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        nn.functional.cross_entropy(local(x), y).backward()
        optimizer.step()
        deltas.append({k: initial[k] - local.state_dict()[k].detach().clone() for k in initial})

    mean = {k: sum(delta[k] for delta in deltas) / 3 for k in initial}
    collusion = {k: 3 * mean[k] - deltas[1][k] - deltas[2][k] for k in initial}
    isolation_error = max(float((collusion[k] - deltas[0][k]).abs().max()) for k in initial)
    if isolation_error > 2e-6:
        raise RuntimeError("Collusion isolation failed tensor-level parity")

    target_x, target_y, target_indices = private_data[0]
    eval_dir = out / "evaluator_only"
    eval_dir.mkdir()
    np.savez_compressed(eval_dir / "target.npz", x=target_x.numpy(), y=target_y.numpy(),
                        source_idx=target_indices)

    # This extra observer receives the true labels of ALL 24 contributors,
    # not only victim labels. Images remain evaluator-only. The mean of three
    # equal-size one-step SGD updates equals the full pooled-batch gradient.
    if joint_oracle:
        pooled_x = torch.cat([entry[0] for entry in private_data])
        pooled_y = torch.cat([entry[1] for entry in private_data])
        # This is a stronger-than-realistic oracle: labels are in client blocks.
        # Label/site grouping is therefore additionally visible; record images
        # remain hidden. Do not call this a normal SecAgg+ observation.
        np.savez_compressed(eval_dir / "pooled.npz", x=pooled_x.numpy(), y=pooled_y.numpy())
        # Test mathematical identity before reporting any attack outcomes.
        parity_model = DigitMLP()
        parity_model.load_state_dict(initial)
        pooled_gradient = torch.autograd.grad(
            nn.functional.cross_entropy(parity_model(pooled_x), pooled_y),
            tuple(parity_model.parameters()),
        )
        max_error = max(float((mean[k] / learning_rate - grad).abs().max())
                        for k, grad in zip(initial, pooled_gradient))
        if max_error > 1e-5:
            raise RuntimeError("Pooled exact-gradient parity failed")
        joint_dir = out / "aggregate_joint_oracle" / "attacker"
        joint_dir.mkdir(parents=True)
        torch.save({"schema_version": 1,
                    "observer": "server_aggregate_model_delta_joint_oracle",
                    "features": 64, "classes": 10,
                    "batch_size": int(len(pooled_y)), "model_state": initial,
                    "model_delta": mean, "learning_rate": learning_rate,
                    "local_steps": 1, "known_labels": pooled_y.tolist()},
                   joint_dir / "observation.pt")

    descriptions = {
        "individual": "Unprotected server directly observes target's own update",
        "aggregate_only": "Server sees ONLY the three-client arithmetic mean; naive target-gradient proxy",
        "all_but_one_collusion": "Server plus two colluding clients can algebraically isolate target update",
    }
    tensors = {"individual": deltas[0], "aggregate_only": mean,
               "all_but_one_collusion": collusion}
    contracts = {}
    for case in CASES:
        attack_dir = out / case / "attacker"
        attack_dir.mkdir(parents=True)
        # The observer role distinguishes the assumptions. All cases use the
        # same target labels (privileged) and target-only gradient-matching attack.
        observer = {"individual": "server_individual_model_delta",
                    "aggregate_only": "server_aggregate_model_delta_target_probe",
                    "all_but_one_collusion": "server_collusion_isolated_delta"}[case]
        observation = {"schema_version": 1, "observer": observer,
                       "features": 64, "classes": 10,
                       "batch_size": examples_per_site,
                       "model_state": initial, "model_delta": tensors[case],
                       "learning_rate": learning_rate, "local_steps": 1,
                       "known_labels": target_y.tolist()}
        torch.save(observation, attack_dir / "observation.pt")
        contracts[case] = {"observer": observer, "assumptions": descriptions[case],
                           "observation_sha256": _sha(attack_dir / "observation.pt"),
                           "privileged_target_labels": True,
                           "target_images_visible": False,
                           "other_client_updates_visible_to_attack": case == "all_but_one_collusion"}
    if joint_oracle:
        contracts["aggregate_joint_oracle"] = {
            "observer": "server_aggregate_model_delta_joint_oracle",
            "assumptions": "All contributor labels known and grouped by client; images hidden; no exact record identities",
            "observation_sha256": _sha(joint_dir / "observation.pt"),
            "privileged_all_contributor_labels": True,
            "target_images_visible": False,
            "other_client_updates_visible_to_attack": False,
            "pooled_initial_gradient_max_error": max_error,
        }
    report = {"schema_version": 1, "experiment": "idealized_aggregate_observation_collusion",
              "seed": seed, "sites": 3, "examples_per_site": examples_per_site,
              "initial_model_shared": True, "local_steps": 1,
              "learning_rate": learning_rate, "all_updates_have_equal_sample_weight": True,
              "collusion_isolation_max_tensor_error": isolation_error,
              "target_evaluator_sha256": _sha(eval_dir / "target.npz"),
              "cases": contracts,
              "limitations": ["No cryptographic SecAgg+ protocol was executed",
                              "All three local model updates are produced by a privileged setup process",
                              "Aggregate-only attacker uses a deliberately misspecified target-only proxy",
                              "Target labels are granted to all attackers and are not typical server observations",
                              "Three equal-weight clients, same initial model, one full-batch local step",
                              "Code-level data separation; not OS-level sandboxing",
                              "Collusion isolates target when all other participants disclose their updates"]}
    (out / "observation_contract.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
