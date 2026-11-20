"""Five matched FedAvg model seeds, IID vs strong label heterogeneity, output-only MIA.

The orchestrator is an authorized evaluator; attack implementations receive
only exported probability tensors. Every checkpoint comes from a previously
executed PyTorch multi-client training experiment, NOT Flower.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from feddisclosure.evaluation.matched_membership_observer import paired_predictions
from feddisclosure.attacks.membership import attack_outputs
from feddisclosure.evaluation.membership import evaluate_membership

CONDITIONS = ('iid', 'dirichlet_balanced_0.1')
METHODS = ('max_confidence', 'negative_entropy', 'top2_margin')
SOURCE = ROOT/'reports/results/sprint5/balanced_heterogeneity'


def sha(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def bundle_for(seed: int, condition: str):
    experiment = f'balanced-{condition}-seed{seed}-r4'
    run_file = SOURCE/'runs'/f'{experiment}.json'
    payload = json.loads(run_file.read_text())
    checkpoint = SOURCE/'runs'/'checkpoints'/f'{experiment}_final.pt'
    data_root = SOURCE/'data'/f'seed-{seed}'/condition
    manifest_file = data_root/'manifest.json'
    assert payload['engine'] == 'torch_independent_process_reference_NOT_Flower'
    assert payload['config']['seed'] == seed and payload['config']['partition'] == condition
    assert payload['config']['rounds'] == 4 and payload['config']['clients'] == 3
    assert payload['final_checkpoint']['sha256'] == sha(checkpoint)
    assert payload['manifest_sha256'] == sha(manifest_file)
    manifest = json.loads(manifest_file.read_text())
    assert manifest['schema_version'] == 2
    site_files = [data_root/item['file'] for item in manifest['client_partitions']]
    for file, meta in zip(site_files, manifest['client_partitions']):
        assert sha(file) == meta['sha256']
    test_file = data_root/manifest['evaluator_test']['file']
    assert sha(test_file) == manifest['evaluator_test']['sha256']
    return experiment, checkpoint, site_files, test_file, manifest


def matched_cohorts(seed: int):
    cohorts=[]
    for condition in CONDITIONS:
        _, _, sites, test, _ = bundle_for(seed, condition)
        indices=[]
        for file in sites:
            with np.load(file, allow_pickle=False) as data:
                assert 'source_idx' in data.files
                indices.extend(data['source_idx'].tolist())
        assert len(indices)==len(set(indices))
        with np.load(test,allow_pickle=False) as data:
            assert 'source_idx' in data.files
            held_out=set(data['source_idx'].tolist())
        assert not (set(indices) & held_out)
        cohorts.append((set(indices), held_out))
    # Avoid describing the comparison as paired if public cohorts differ.
    if cohorts[0][1] != cohorts[1][1]:
        raise RuntimeError('Public test cohorts do not match')
    common=cohorts[0][0] & cohorts[1][0]
    return {'train_records_per_condition':[len(c[0]) for c in cohorts],
            'common_member_pool_size':len(common),
            'public_test_records':len(cohorts[0][1]),
            'full_training_cohorts_identical':cohorts[0][0]==cohorts[1][0]}


def confidence_interval(values):
    mean = statistics.mean(values)
    if len(values)<2:
        return [mean,mean]
    margin = float(t.ppf(.975,df=len(values)-1))*statistics.stdev(values)/(len(values)**.5)
    return [mean-margin,mean+margin]


def run(output: Path, seeds: list[int], per_class: int=6):
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or per_class<1:
        raise ValueError('Need >=2 unique model training seeds and positive per-class query count')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Research evidence exists, refusing overwrite')
    completed=[]
    for seed in seeds:
        cohort=matched_cohorts(seed)
        bundles={condition:bundle_for(seed,condition) for condition in CONDITIONS}
        contract=paired_predictions(
            {condition:bundles[condition][1] for condition in CONDITIONS},
            {condition:bundles[condition][2] for condition in CONDITIONS},
            {condition:bundles[condition][3] for condition in CONDITIONS},
            output, per_class=per_class, seed=seed+1000,
        )
        for condition in CONDITIONS:
            experiment, checkpoint, sites, test, manifest = bundles[condition]
            folder=output/condition/f'seed-{seed}'
            methods={}
            for method in METHODS:
                scores_dir=folder/'methods'/method
                # This attack does not import any evaluator membership truth.
                attack_outputs(folder/'attacker'/'model_predictions.npz',scores_dir/'attacker',method=method)
                scored=evaluate_membership(scores_dir/'attacker'/'scores.npz',
                                           folder/'evaluator_only'/'membership_labels.npz',
                                           scores_dir/'evaluation.json')
                methods[method] = scored
            completed.append({
                'training_seed':seed,'partition':condition,'training_engine':'independent_process_PyTorch_not_Flower',
                'checkpoint_sha256':sha(checkpoint),'manifest_sha256':sha(SOURCE/'data'/f'seed-{seed}'/condition/'manifest.json'),
                'cohort':cohort,'matched_query_cohort_sha256':contract['matched_query_cohort_sha256'],
                'query_count':contract['query_count'],
                'member_count':contract['member_count'],'nonmember_count':contract['nonmember_count'],
                'observation_sha256':sha(folder/'attacker'/'model_predictions.npz'),
                'evaluator_truth_sha256':sha(folder/'evaluator_only'/'membership_labels.npz'),
                'methods':{name:{'roc_auc':result['roc_auc'],'attack_advantage':result['attack_advantage'],
                                 'tpr_at_fpr':result['tpr_at_fpr'],
                                 'scores_sha256':sha(folder/'methods'/name/'attacker'/'scores.npz')}
                           for name,result in methods.items()},
            })
            print(f'seed={seed} partition={condition}: '+', '.join(
                f'{name}={methods[name]["roc_auc"]:.4f}' for name in METHODS),flush=True)
    agg={}
    for method in METHODS:
        by_condition={}
        for cond in CONDITIONS:
            scores=[c['methods'][method]['roc_auc'] for c in completed if c['partition']==cond]
            by_condition[cond]={'mean_roc_auc':statistics.mean(scores),
                                'sample_sd_auc':statistics.stdev(scores),
                                't95_interval_auc':confidence_interval(scores),
                                'aucs':scores}
        pairs=[]
        for seed in seeds:
            x=next(c for c in completed if c['training_seed']==seed and c['partition']=='iid')
            y=next(c for c in completed if c['training_seed']==seed and c['partition']=='dirichlet_balanced_0.1')
            pairs.append(y['methods'][method]['roc_auc']-x['methods'][method]['roc_auc'])
        agg[method]={'by_partition':by_condition,
                     'skew_minus_iid_paired_difference':{'mean':statistics.mean(pairs),
                                                          't95_interval':confidence_interval(pairs),
                                                          'per_seed':pairs}}
    report={'schema_version':1,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'experiment':'five_independent_model_seed_membership_iid_vs_strong_skew',
            'trained_models':len(completed),'training_seeds':seeds,'per_class_limit':per_class,
            'conditions':list(CONDITIONS),'methods':list(METHODS),
            'observer':'external_model_user_softmax_only_unknown_true_labels',
            'runs':completed,'aggregate':agg,
            'limitations':['Reused previously trained four-round public-digit PyTorch models, no Flower',
                           'Training record sets are NOT identical between IID and skew partitions; comparisons are descriptive rather than causal',
                           'Each seed does use the exact same subset of members drawn from training-set intersection and same held-out nonmembers in both arms',
                           'Queries are balanced, class-matched and sampled once per checkpoint; not a real network prediction API',
                           'The 95% t intervals across five seeds are exploratory, not power-qualified',
                           'FPR=0.01 may be below the empirical resolution of the small query cohort',
                           'AUC near 0.5 does not establish differential privacy',
                           'Some samples may appear in different training subsets across different training seeds',
                           'Attacker receives output probabilities, not labels or membership truth']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('reports/results/sprint3/membership_balanced_comparison'))
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(241,246)))
    p.add_argument('--per-class',type=int,default=6)
    args=p.parse_args()
    output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    report=run(output,args.seeds,args.per_class)
    print(json.dumps({k:{cond:v['mean_roc_auc'] for cond,v in met['by_partition'].items()}
                      for k,met in report['aggregate'].items()},indent=2))


if __name__=='__main__':
    main()
