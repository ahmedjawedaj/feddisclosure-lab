"""Recompute evaluator-only joint attack scores from saved data and hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

from scripts.run_joint_aggregate_attack import score_joint

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder: Path, seeds: tuple[int, ...] = (241,242,243)) -> dict:
    rows = []
    for seed in seeds:
        path = folder / f'seed{seed}'
        summary = json.loads((path / 'joint_summary.json').read_text())
        if summary['seed'] != seed or summary['iterations'] != 160:
            raise AssertionError('Incorrect attack seed or iterations')
        attack = path / 'aggregate_joint_oracle' / 'attacker'
        observation = attack / 'observation.pt'
        recovered = attack / 'recovered.npz'
        scorer = path / 'joint_score.json'
        assert sha(observation) == summary['observer_input_sha256'] == summary['contract']['observation_sha256']
        assert sha(recovered) == summary['recovered_sha256']
        assert sha(scorer) == summary['scoring_sha256']
        obs = torch.load(observation, map_location='cpu', weights_only=True)
        assert obs['observer'] == 'server_aggregate_model_delta_joint_oracle'
        assert len(obs['known_labels']) == 24
        assert not any(k in obs for k in ('x','images','source_idx','ground_truth','site_ids'))
        metrics = score_joint(path / 'evaluator_only' / 'pooled.npz',
                              path / 'evaluator_only' / 'target.npz', recovered)
        saved = json.loads(scorer.read_text())
        for key in ('pooled_label_matched_mse','target_oracle_label_matched_mse'):
            if not math.isclose(metrics[key], summary['metrics'][key],abs_tol=1e-9):
                raise AssertionError(f'Incorrect summary metric: {key}')
            if not math.isclose(metrics[key], saved[key],abs_tol=1e-9):
                raise AssertionError(f'Incorrect evaluator score: {key}')
        rows.append({'seed': seed,
                     'pooled_mse': metrics['pooled_label_matched_mse'],
                     'target_oracle_mse': metrics['target_oracle_label_matched_mse']})
    return {'integrity': 'passed', 'n_seeds': len(rows), 'results': rows,
            'limitation': 'All-label site-grouped oracle; NOT secure aggregation or an attacker site-attribution result'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, default=ROOT / 'reports/results/sprint4/joint_aggregate_oracle')
    a=p.parse_args()
    result=verify(a.folder)
    output=a.folder / 'verification_summary.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
