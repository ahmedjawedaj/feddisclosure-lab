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
    'analytics_history': 'reports/results/sprint1/history_matrix_summary.json',
    'membership_balanced': 'reports/results/sprint3/membership_balanced_comparison/summary.json',
}


def _load(root: Path, rel: str):
    path = root / rel
    data = json.loads(path.read_text(encoding='utf-8'))
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def _number(x):
    if not isinstance(x, (float, int)) or isinstance(x, bool) or not math.isfinite(x):
        raise ValueError('Evidence contains invalid numerical metric')
    return float(x)


def _v014(root: Path, hashes: dict):
    """Native execution, SecAgg+, OpenDP and DP-FedAvg scalars. Returns None if not present.

    A protocol flag is True only when the saved evidence itself says it passed.
    """
    paths = {
        'scenarios': 'reports/results/native_flower/v014/scenario_summary.json',
        'mnist_parity': 'reports/results/native_flower/v014/run-mnist-full-iid/flower_parity.json',
        'secagg': 'reports/results/secagg/v014/secagg_verification.json',
        'opendp': 'reports/results/privacy/v014/opendp/opendp_study.json',
        'digits_dp': 'reports/results/privacy/v014/digits_full_dpfedavg/summary.json',
        'mnist_dp': 'reports/results/privacy/v014/mnist_full_dpfedavg/summary.json',
    }
    if not all((root / rel).exists() for rel in paths.values()):
        return None
    d = {}
    for key, rel in paths.items():
        d[key], hashes['v014_' + key] = _load(root, rel)
    secagg = d['secagg']['scenarios']
    secagg_ok = (secagg['secagg-on-t2-n3']['runner_native_flower_verified'] is True
                 and secagg['plaintext-control']['runner_native_flower_verified'] is True
                 and d['secagg']['equivalence_2_rounds']['within_tolerance'] is True)
    flags = {
        'docker': bool(d['mnist_parity']['require_deployment']),
        'flower': d['mnist_parity']['verified_numerical_parity'] is True,
        'opendp': d['opendp']['ledger_with_real_mechanism']['all_passed'] is True,
        'opacus': any(g['mode'] == 'opacus_dp' for g in d['digits_dp']['groups']),
        'secagg_plus': secagg_ok,
    }
    def dp_rows(summary):
        rows = []
        for g in summary['groups']:
            rows.append({
                'partition': g['partition'], 'mode': g['mode'], 'target_epsilon': g['target_epsilon'],
                'achieved_epsilon': (_number(g['achieved_epsilon_max_client']['mean'])
                                     if 'achieved_epsilon_max_client' in g else None),
                'test_accuracy': _number(g['test_accuracy']['mean']),
                'worst_client_accuracy': _number(g['worst_client_val_accuracy']['mean']),
                'membership_auc': _number(g['membership_auc']['mean']),
                'seeds': g['test_accuracy']['n']})
        return rows
    sc = d['secagg']['collusion_single_round']
    return {'flags': flags,
            'fault_scenarios': [{'name': x['scenario'], 'verdict': x['verdict']}
                                for x in d['scenarios']['scenarios']],
            'digits_dp': dp_rows(d['digits_dp']), 'mnist_dp': dp_rows(d['mnist_dp']),
            'secagg': {'plaintext_diff': _number(d['secagg']['equivalence_2_rounds']['secagg_vs_plaintext_control_max_abs_diff']),
                       'tolerance': _number(d['secagg']['equivalence_2_rounds']['tolerance']),
                       'victim_error_server_only': _number(sc['baseline_error_if_victim_assumed_equal_to_aggregate']),
                       'victim_error_1_colluder': _number(sc['error_with_server_plus_1_colluder']),
                       'victim_error_n_minus_1': _number(sc['error_with_server_plus_2_colluders_n_minus_1'])},
            'opendp_ledger_passed': d['opendp']['ledger_with_real_mechanism']['all_passed'],
            'completion': _completion(root, hashes)}


def _completion(root: Path, hashes: dict):
    """v0.14 completion evidence (4-client SecAgg+, dropout matrix, MNIST matrix, fail-fast). None if absent."""
    dm_rel = 'reports/results/secagg/v014_dropout_matrix/dropout_matrix_summary.json'
    if not (root / dm_rel).exists():
        return None
    dm, hashes['v014_dropout_matrix'] = _load(root, dm_rel)
    scen = []
    for name, x in dm['scenarios'].items():
        scen.append({'name': name, 'outcome': x['protocol_outcome'],
                     'prediction_matches': x['prediction_matches_outcome'] is True,
                     'within_quantisation_bound': (x.get('vs_independent_pytorch') or {}).get('within_tolerance')})
    rep = dm.get('n4_two_round_replicates', {})
    one = dm.get('n4_one_round_bound_check', {})
    mnist = []
    for run in ('run-mnist-full-iid', 'run-mnist-full-iid-seed2', 'run-mnist-full-iid-seed3', 'run-mnist-full-labelskew'):
        par, hashes['v014_' + run + '_parity'] = _load(root, f'reports/results/native_flower/v014/{run}/flower_parity.json')
        bw, hashes['v014_' + run + '_bitwise'] = _load(root, f'reports/results/native_flower/v014/{run}/bitwise_replay.json')
        mnist.append({'name': run, 'parity_1e-5': par['verified_numerical_parity'] is True,
                      'bitwise_flower_replay': bw['bitwise_reproduction_of_flower_run'] is True})
    ff = []
    for name in ('clientexec-before-run', 'clientexec-after-round1', 'supernode-after-round1'):
        run, hashes['v014_ff_' + name] = _load(root, f'reports/results/native_flower/v014/fail_fast/{name}/server/flower_run.json')
        ff.append({'name': name, 'exit_code': run['exit_code'], 'seconds': _number(run['seconds'])})
    lira = {}
    for key, rel in (('overfit', 'large_overfit'), ('moderate', 'large_moderate'), ('mnist_overfit', 'large_mnist_overfit')):
        summ, hashes['v014_lira_' + key] = _load(root, f'reports/results/privacy/v014/lira/{rel}/summary.json')
        lira[key] = {'shadows': summ['shadows'], 'targets': summ['targets'], 'rows': [
            {'mode': m['mode'], 'target_epsilon': m['target_epsilon'],
             'achieved_epsilon': _number(m['achieved_epsilon_max']) if m['achieved_epsilon_max'] is not None else None,
             'heldout_accuracy': _number(m['heldout_accuracy'][0]), 'lira_auc': _number(m['lira_auc'][0]),
             'lira_auc_ci_low': _number(m['lira_auc'][1]),
             'lira_tpr_at_fpr_0.01': _number(m['lira_tpr_at_fpr_0.01'][0]),
             'loss_threshold_auc': _number(m['loss_threshold_auc'][0])} for m in summ['modes']]}
    vc, hashes['v014_view_comparison'] = _load(root, 'reports/results/privacy/v014/view_comparison/summary.json')
    views = {k: {'victim_mse': _number(v['victim_mse'][0]), 'recognisable_rate': _number(v['rate_mse_below_0_05'][0])}
             for k, v in vc['views'].items()}
    lf, hashes['v0143_label_floor'] = _load(root, 'reports/results/privacy/v0143/label_floor.json')
    iv, hashes['v0143_view_intervals'] = _load(root, 'reports/results/privacy/v0143/view_intervals.json')
    for k, v in views.items():
        d = iv['views'][k]['paired_rate_minus_label_floor_bootstrap95']
        v['rate_minus_label_floor'] = {'mean': _number(d[0]), 'low': _number(d[1]), 'high': _number(d[2])}
    label_floor = {'mse': _number(lf['pooled']['class_average_mse']),
                   'recognisable_rate': _number(lf['pooled']['class_average_rate_mse_below_0_05']),
                   'public_pool_size': int(lf['per_seed'][0]['public_pool_size']),
                   'client_records': int(vc['clients'] * vc['records_per_client']),
                   'pool_size_sensitivity': [{'pool_size': int(lf['per_seed'][0]['public_pool_size']) if r['pool_size'] == 'full' else int(r['pool_size']),
                                              'mse': _number(r['mse']),
                                              'recognisable_rate': _number(r['rate_mse_below_0_05'])}
                                             for r in lf['pool_size_sensitivity']['rows']]}
    mc, hashes['v014_mc_rule'] = _load(root, 'reports/results/secagg/v014_dropout_matrix/multiround_acceptance_rule.json')
    mcc = mc['cases']['n4_2_rounds']
    gate, hashes['v014_replay_gate'] = _load(root, 'reports/results/native_flower/v014/replay_gate_summary.json')
    return {'dropout': scen, 'lira': lira,
            'view_comparison': {'views': views, 'label_floor': label_floor,
                                'mean_image_floor_ignores_labels_mse': _number(vc['floor']['mean_image_mse'][0]), 'seeds': vc['seeds']},
            'multiround_rule': {'q99': _number(mcc['mc_quantiles']['0.99']), 'median': _number(mcc['mc_quantiles']['0.5']),
                                'fraction_above_naive_2x': _number(mcc['fraction_above_naive_2x_bound']), 'trials': mc['trials'],
                                'docker_runs_accepted': all(r['accepted'] for r in mcc['docker_runs'].values())},
            'replay_gate': {'passes': gate['replay_gate_passes'], 'float64_passes': gate['float64_gate_passes'], 'runs': gate['runs_total']},
            'n4_one_round': {'max_abs_diff': _number(one.get('max_abs_diff')), 'bound': _number(one.get('bound')),
                             'within_bound': one.get('within_bound') is True},
            'n4_two_round': {'diffs': {k: _number(v) for k, v in rep.get('max_abs_diff_vs_independent_pytorch', {}).items()},
                             'naive_tolerance': _number(rep.get('naive_two_round_tolerance_2x_bound')),
                             'runs_above_naive_tolerance': rep.get('runs_above_naive_tolerance', [])},
            'mnist_native': mnist, 'fail_fast': ff}


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
    control = data['analytics_history']
    if control['number_of_matched_populations'] != 90 or control['queries_evaluated'] != 1080:
        raise ValueError('History control matrix missing required cases')
    history = [{'name': mode, 'exact_inference_rate': _number(v['exact_inference_rate']),
                'second_query_release_rate': _number(v['second_query_release_rate']),
                'case_count': v['cases']}
               for mode, v in sorted(control['condition_summary'].items())]
    paired = data['membership_balanced']
    if paired['trained_models'] != 10 or len(paired['training_seeds']) != 5:
        raise ValueError('New five-seed membership study incomplete')
    membership_balanced = [
        {'method': method, 'iid_mean_auc': _number(res['by_partition']['iid']['mean_roc_auc']),
         'skew_mean_auc': _number(res['by_partition']['dirichlet_balanced_0.1']['mean_roc_auc']),
         'models_per_condition': 5}
        for method, res in paired['aggregate'].items()
    ]
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
    v014 = _v014(root, hashes)
    flags = v014['flags'] if v014 else {'docker': False, 'flower': False,
                                        'opendp': False, 'opacus': False, 'secagg_plus': False}
    return {'schema_version': 1, 'project': 'FedDisclosure Lab',
            'data_scope': 'reviewed_scalar_evidence_only',
            'native_protocol_verified': flags, 'v014': v014,
            'heterogeneity': partitions, 'membership': membership,
            'collusion': collusion, 'joint_oracle': joint, 'analytics_history': history,
            'membership_balanced': membership_balanced,
            'source_sha256': hashes,
            'limitations': [
                'Native flags mean the saved v0.14 evidence passed its own checks. They are not privacy certifications.',
                'SecAgg+ gives no protection against a server that colludes with n-1 clients. A client that is absent at round start halts SecAgg+. A loss during a round is tolerated when surviving share holders are at least the threshold (5 Docker scenarios).',
                'Two of four full-MNIST native runs fail the 1e-5 float64 parity gate (float32 reply-order aggregation). The 4-client original 2-round baseline fails the naive 2-round tolerance.',
                'Opacus and OpenDP guarantees are formal for their stated mechanisms only. Measured membership AUC is near chance with or without DP.',
                'All attack data are approved public digits; no institutional patient records.',
                'Joint aggregate observer knows all contributor labels; site attribution is not established.',
                'Observed reconstruction error is not a formal privacy guarantee.',
                'The principal-wide query history control prevents one ID-reset attack, not arbitrary adaptive inference.',
                'Five training seeds and three historical membership models are exploratory, not a publication-grade power analysis.',
                'Five-model IID/skew membership queries are paired, but training membership sets differ across conditions.',
            ]}
