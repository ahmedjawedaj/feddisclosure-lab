"""Independent, read-only replay of output-only MIA scores and artifact hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import math
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from feddisclosure.evaluation.membership import score_membership
from scripts.run_balanced_membership import METHODS, CONDITIONS, confidence_interval


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root:Path)->dict:
    summary=json.loads((root/'summary.json').read_text())
    seeds=summary['training_seeds']
    assert len(seeds)==5 and len(set(seeds))==5
    assert set(summary['conditions'])==set(CONDITIONS)
    assert set(summary['methods'])==set(METHODS)
    assert summary['trained_models']==10 and len(summary['runs'])==10
    for seed in seeds:
        pair=[r for r in summary['runs'] if r['training_seed']==seed]
        assert {r['partition'] for r in pair}==set(CONDITIONS)
        assert pair[0]['matched_query_cohort_sha256']==pair[1]['matched_query_cohort_sha256']
        assert pair[0]['evaluator_truth_sha256']==pair[1]['evaluator_truth_sha256']
        assert pair[0]['query_count']==pair[1]['query_count']
    count=0
    for run in summary['runs']:
        cond,seed=run['partition'],run['training_seed']
        folder=root/cond/f'seed-{seed}'
        with np.load(folder/'attacker'/'model_predictions.npz', allow_pickle=False) as data:
            assert set(data.files)=={'probabilities'}, 'Attacker receives data beyond output confidences'
        assert sha(folder/'attacker'/'model_predictions.npz')==run['observation_sha256']
        assert sha(folder/'evaluator_only'/'membership_labels.npz')==run['evaluator_truth_sha256']
        contract=json.loads((folder/'observation_contract.json').read_text())
        assert contract['matched_query_cohort_sha256']==run['matched_query_cohort_sha256']
        assert contract['attacker_known_labels'] is False
        assert contract['attacker_receives']==['probabilities']
        for method, met in run['methods'].items():
            scored_file=folder/'methods'/method/'attacker'/'scores.npz'
            assert sha(scored_file)==met['scores_sha256']
            with np.load(scored_file,allow_pickle=False) as scores, np.load(folder/'evaluator_only'/'membership_labels.npz',allow_pickle=False) as truth:
                assert set(scores.files)=={'scores'}
                assert set(truth.files)=={'membership'}
                calculated=score_membership(scores['scores'],truth['membership'])
            assert abs(calculated['roc_auc']-met['roc_auc'])<1e-12
            assert abs(calculated['attack_advantage']-met['attack_advantage'])<1e-12
            assert calculated['tpr_at_fpr']==met['tpr_at_fpr']
            count+=1
    for method, item in summary['aggregate'].items():
        for condition in CONDITIONS:
            source=[r['methods'][method]['roc_auc'] for r in summary['runs'] if r['partition']==condition]
            actual=item['by_partition'][condition]
            assert len(source)==5 and source==actual['aucs']
            assert math.isclose(statistics.mean(source),actual['mean_roc_auc'],abs_tol=1e-12)
            assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(confidence_interval(source),actual['t95_interval_auc']))
    return {'verified_trained_models':len(summary['runs']), 'verified_attack_evaluations':count,
            'seeds':seeds, 'confidence_methods':list(METHODS)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,default=Path('reports/results/sprint3/membership_balanced_comparison'))
    args=p.parse_args()
    root=args.input_dir if args.input_dir.is_absolute() else ROOT/args.input_dir
    print(json.dumps(verify(root),indent=2))


if __name__=='__main__':
    main()
