"""Joint, site-grouped-label-oracle attack on an arithmetic 3-client aggregate.

This is a high-privilege leakage *upper-bound experiment*, not SecAgg+.
Images remain evaluator-only; contributor labels are grouped by site.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.evaluation.aggregate_observer import create_aggregate_observations
from feddisclosure.evaluation.gradient import score_reconstruction


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score_joint(pooled_path: Path, target_path: Path, recovered_path: Path) -> dict:
    """Privileged label-matched matching, NEVER returned to the attacker process."""
    with np.load(pooled_path, allow_pickle=False) as pooled, np.load(target_path, allow_pickle=False) as target, np.load(recovered_path, allow_pickle=False) as recovered:
        all_x, all_y = pooled['x'], pooled['y']
        target_x, target_y = target['x'], target['y']
        estimated = recovered['x']
    if estimated.shape != all_x.shape or len(all_y) != len(all_x):
        raise ValueError('Invalid joint reconstruction dimensions')
    if not np.isfinite(estimated).all():
        raise ValueError('Nonfinite reconstructed images')
    # The attack receives labels in pooled-data order; the ground truth is
    # used only here to match recovered images without revealing a mapping.
    def match(x, y):
        cost = ((x[:, None, :] - estimated[None, :, :]) ** 2).mean(axis=2)
        cost[y[:, None] != all_y[None, :]] = 1e5
        row, col = linear_sum_assignment(cost)
        if np.any(cost[row, col] >= 1e5):
            raise ValueError('Label-matched assignment impossible')
        return float(cost[row, col].mean())
    return {
        'pooled_label_matched_mse': match(all_x, all_y),
        'target_oracle_label_matched_mse': match(target_x, target_y),
        'reconstruction_count': int(len(estimated)),
        'target_count': int(len(target_x)),
        'warning': 'Target-oracle metric gives an evaluator matching privilege; attacker cannot attribute recovered images to a specific site',
    }


def run(data_root: Path, output_dir: Path, seed: int = 241, iterations: int = 160):
    sites = [data_root / f'institution-{idx:02d}' / 'train.npz' for idx in range(3)]
    contract = create_aggregate_observations(sites, output_dir, seed=seed, joint_oracle=True)
    attack = output_dir / 'aggregate_joint_oracle' / 'attacker'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT), env.get('PYTHONPATH', '')])
    subprocess.run([sys.executable, '-m', 'feddisclosure.attacks.gradient_cli',
                    '--observation', str(attack / 'observation.pt'), '--output-dir', str(attack),
                    '--iterations', str(iterations), '--seed', str(seed + 1)],
                   check=True, timeout=240, env=env)
    # Strictly separate evaluator subprocess: only it receives ground truth.
    scorer = output_dir / 'joint_score.json'
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'evaluate',
                    '--pooled', str(output_dir / 'evaluator_only' / 'pooled.npz'),
                    '--target', str(output_dir / 'evaluator_only' / 'target.npz'),
                    '--recovered', str(attack / 'recovered.npz'), '--output', str(scorer)],
                   check=True, timeout=90, env=env)
    result = {'schema_version': 1, 'seed': seed, 'iterations': iterations,
              'threat_model': 'aggregate_with_all_contributor_labels_oracle',
              'observer_input_sha256': digest(attack / 'observation.pt'),
              'recovered_sha256': digest(attack / 'recovered.npz'),
              'scoring_sha256': digest(scorer), 'contract': contract['cases']['aggregate_joint_oracle'],
              'metrics': json.loads(scorer.read_text())}
    (output_dir / 'joint_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('run','evaluate'), nargs='?', default='run')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/training/digits-iid')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--seed', type=int, default=241)
    p.add_argument('--iterations', type=int, default=160)
    p.add_argument('--pooled', type=Path)
    p.add_argument('--target', type=Path)
    p.add_argument('--recovered', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.mode == 'evaluate':
        if any(x is None for x in (args.pooled, args.target, args.recovered, args.output)):
            p.error('Evaluation requires pooled, target, recovered and output paths')
        result = score_joint(args.pooled, args.target, args.recovered)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    else:
        if args.output_dir is None:
            p.error('--output-dir is required')
        if args.iterations < 1:
            p.error('iterations must be positive')
        result = run(args.data_root, args.output_dir, seed=args.seed, iterations=args.iterations)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
