"""Read-only scalar evidence adapter. Never opens training images or attacker tensors."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    'heterogeneity': 'reports/results/sprint5/balanced_heterogeneity/summary.json',
    'membership': 'reports/results/sprint3/membership_training_seeds/membership_training_seeds_summary.json',
    'collusion': 'reports/results/sprint4/aggregate_collusion/verification_summary.json',
}


def _load(root: Path, rel: str):
    path = root / rel
    data = json.loads(path.read_text(encoding='utf-8'))
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def _number(x):
    if not isinstance(x, (float, int)) or isinstance(x, bool) or not math.isfinite(x):
        raise ValueError('Evidence contains invalid numerical metric')
    return float(x)


def make_snapshot(root: Path = ROOT) -> dict:
    """Strictly whitelist public aggregate results, never raw/evaluator files."""
    root = Path(root)
    data = {}
    hashes = {}
    for name, rel in SOURCES.items():
        data[name], hashes[name] = _load(root, rel)
    if data['collusion']['integrity'] != 'passed' or data['collusion']['n_attack_executions'] != 15:
        raise ValueError('Collusion summary not verified')
    if sorted(data['heterogeneity']['aggregate']['completed_seeds']) != list(range(241,246)):
        raise ValueError('Heterogeneity study incomplete')
    partitions = []
    for label, item in data['heterogeneity']['aggregate']['by_partition'].items():
        if item['n'] != 5:
            raise ValueError('Missing five-seed replication')
        partitions.append({'name': label,
                           'global_accuracy': _number(item['accuracy']['mean']) * 100,
                           'worst_client_accuracy': _number(item['worst_client_accuracy']['mean']) * 100,
                           'global_sd': _number(item['accuracy']['sample_sd']) * 100,
                           'seeds': item['n']})
    membership = [{'name': k, 'auc_mean': _number(v['auc_mean']),
                   'auc_sd': _number(v['auc_sample_sd']),
                   'trained_models': len(v['observed_aucs'])}
                  for k,v in data['membership']['aggregate'].items()]
    collusion = [{'name': k, 'mean_mse': _number(v), 'seeds': 5}
                 for k,v in data['collusion']['mean_mse'].items()]
    joint = []
    for seed in range(241,246):
        rel = f'reports/results/sprint4/joint_aggregate_oracle/seed{seed}/joint_summary.json'
        if (root / rel).exists():
            summary, sha = _load(root, rel)
            if summary['seed'] != seed or summary['threat_model'] != 'aggregate_with_all_contributor_labels_oracle':
                raise ValueError('Incorrect joint-oracle threat model')
            joint.append({'seed': seed,
                          'pooled_mse': _number(summary['metrics']['pooled_label_matched_mse']),
                          'target_oracle_mse': _number(summary['metrics']['target_oracle_label_matched_mse'])})
            hashes[f'joint_{seed}'] = sha
    return {'schema_version': 1, 'project': 'FedDisclosure Lab',
            'data_scope': 'reviewed_scalar_evidence_only',
            'native_protocol_verified': {'docker': False, 'flower': False,
                                         'opendp': False, 'opacus': False, 'secagg_plus': False},
            'heterogeneity': partitions, 'membership': membership,
            'collusion': collusion, 'joint_oracle': joint,
            'source_sha256': hashes,
            'limitations': [
                'OpenDP, Opacus, Flower and SecAgg+ native execution is not verified.',
                'All attack data are approved public digits; no institutional patient records.',
                'Joint aggregate observer knows all contributor labels; site attribution is not established.',
                'Observed reconstruction error is not a formal privacy guarantee.',
                'Five training seeds and three membership models are exploratory, not a publication-grade power analysis.',
            ]}
