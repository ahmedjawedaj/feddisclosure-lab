"""Controlled repeated-query disclosure study using the ACTUAL SQLite policy engine.

This is a local site-engine benchmark, not a Docker or HTTP deployment test.
The HTTP gateway bypass/mitigation is separately run by run_history_scope_benchmark.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from feddisclosure.data.synthetic import InstitutionSpec, initialize_database
from feddisclosure.common.models import AggregateRequest
from feddisclosure.analytics.policy import evaluate_release
from feddisclosure.attacks.aggregate_difference import ObservedCount, infer_restricted_count

CONDITIONS = {
    'naive': ('naive', False),
    'history_same_id': ('history', False),
    'history_reset_id': ('history', True),
    'principal_history_reset_id': ('principal_history', True),
}


def run_case(seed: int, min_cell: int, distribution: str, *, scratch: Path) -> dict:
    rng = random.Random(f'{seed}:{min_cell}:{distribution}:v1')
    max_cases = min_cell - 1 if distribution == 'small_subgroups' else 2 * min_cell
    site_specs = [InstitutionSpec(name, rows, rng.randint(0, max_cases)) for name, rows in
                  [('institution-a', 40), ('institution-b', 35), ('institution-c', 35)]]
    databases = []
    try:
        for spec in site_specs:
            file = scratch / f'{spec.name}.sqlite'
            initialize_database(file, spec)
            databases.append(sqlite3.connect(file))
        outcomes = {}
        for mode, (policy, reset_id) in CONDITIONS.items():
            # New query history for each independent matched condition. The
            # synthetic source records remain identical across conditions.
            for conn in databases:
                conn.execute('DELETE FROM release_history')
                conn.execute('DELETE FROM query_audit')
                conn.commit()
            observed = []
            release_status = []
            for query_id, eid in [('all_records', 'first'),
                                  ('negative_records', 'second' if reset_id else 'first'),
                                  ('all_records', 'third')]:
                request = AggregateRequest(query_id=query_id, experiment_id=eid, policy=policy)
                decisions = [evaluate_release(conn, request, min_cell) for conn in databases]
                for conn in databases:
                    conn.commit()
                all_ok = all(dec.approved for dec in decisions)
                count = sum(dec.count for dec in decisions) if all_ok else None
                if count is not None:
                    observed.append(ObservedCount(query_id, count))
                release_status.append({'query_id': query_id, 'released': all_ok, 'aggregate_count': count})
            inference = infer_restricted_count(observed)
            outcomes[mode] = {
                'released': release_status,
                'inferred_restricted_count': inference,
                'inference_exact': inference == sum(s.positive for s in site_specs)
                                 if inference is not None else False,
            }
        return {'seed': seed, 'min_cell': min_cell, 'population': distribution,
                'evaluator_only': {'per_site_positive_counts': [s.positive for s in site_specs],
                                   'total_sensitive_count': sum(s.positive for s in site_specs)},
                'conditions': outcomes}
    finally:
        for db in databases:
            db.close()


def make_matrix(*, seeds: list[int], thresholds: list[int], output: Path) -> dict:
    if len(set(seeds)) != len(seeds) or len(set(thresholds)) != len(thresholds):
        raise ValueError('Seed and threshold values must be unique')
    if not seeds or not thresholds or any(n < 2 or n > 12 for n in thresholds):
        raise ValueError('Need nonempty seeds and thresholds in range 2..12')
    cases = []
    with tempfile.TemporaryDirectory(prefix='fdl-history-matrix-') as folder:
        for seed in seeds:
            for threshold in thresholds:
                for distribution in ('small_subgroups', 'mixed_subgroups'):
                    scratch = Path(folder) / str(seed) / str(threshold) / distribution
                    scratch.mkdir(parents=True)
                    cases.append(run_case(seed, threshold, distribution, scratch=scratch))
    n = len(cases)
    summary = {}
    for mode in CONDITIONS:
        inference_count = sum(c['conditions'][mode]['inference_exact'] for c in cases)
        second_releases = sum(c['conditions'][mode]['released'][1]['released'] for c in cases)
        repeated_releases = sum(c['conditions'][mode]['released'][2]['released'] for c in cases)
        summary[mode] = {
            'cases': n, 'exact_count_inference_cases': inference_count,
            'exact_inference_rate': inference_count / n,
            'second_query_released_cases': second_releases,
            'second_query_release_rate': second_releases / n,
            'harmless_repeat_released_cases': repeated_releases,
        }
    # Hash covers all actual observations and evaluator truth, not timestamps.
    case_hash = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = {
        'schema_version': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'matched_statistical_history_scope_matrix',
        'method': 'actual_local_SQLite_release_policy_engine',
        'seeds': seeds, 'min_cell_thresholds': thresholds,
        'populations': ['small_subgroups', 'mixed_subgroups'],
        'cases_sha256': case_hash,
        'observer': 'approved_cross_site_aggregate_counts_only',
        'summary': summary, 'cases': cases,
        'limitations': [
            'Synthetic counts with generated low/mixed prevalence, not a population estimate',
            'One fixed two-query attack only; adaptive SQL attacks outside the allowlist not tested',
            'No user identity isolation beyond the single research principal',
            'Query rejection is not equivalent to a false-positive rate for legitimate analysis',
            'The principal-history rule is a heuristic, NOT differential privacy',
            'This matrix executes SQLite policy logic directly, not the HTTP gateway or Docker',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    # Whitelisted public scalar rollup for the offline explorer. No site
    # membership sets or evaluator-per-case populations are included.
    summary_out = output.with_name('history_matrix_summary.json')
    summary_out.write_text(json.dumps({
        'schema_version': 1,
        'experiment': result['experiment'],
        'condition_summary': summary,
        'number_of_matched_populations': n,
        'queries_evaluated': n * len(CONDITIONS) * 3,
        'full_evidence_cases_sha256': case_hash,
        'limitations': result['limitations'],
    }, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('reports/results/sprint1/history_matrix.json'))
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(241, 256)))
    p.add_argument('--thresholds', type=int, nargs='+', default=[2, 5, 10])
    a = p.parse_args()
    result = make_matrix(seeds=a.seeds, thresholds=a.thresholds,
                         output=a.output if a.output.is_absolute() else ROOT / a.output)
    print(json.dumps(result['summary'], indent=2))
    print('case hash:', result['cases_sha256'])


if __name__ == '__main__':
    main()
