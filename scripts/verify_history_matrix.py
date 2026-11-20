"""Read-only independent verification of matched statistical disclosure matrix."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def verify(path: Path) -> dict:
    data = json.loads(path.read_text())
    assert data['schema_version'] == 1
    cases = data['cases']
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == data['cases_sha256'], 'Tampered evidence cases'
    assert len(cases) == len(data['seeds']) * len(data['min_cell_thresholds']) * 2
    expected_keys = {'naive', 'history_same_id', 'history_reset_id', 'principal_history_reset_id'}
    for case in cases:
        threshold = case['min_cell']
        counts = case['evaluator_only']['per_site_positive_counts']
        assert len(counts) == 3 and all(0 <= x <= 2 * threshold for x in counts)
        total = sum(counts)
        assert total == case['evaluator_only']['total_sensitive_count']
        assert set(case['conditions']) == expected_keys
        must_deny = any(0 < x < threshold for x in counts)
        for mode, result in case['conditions'].items():
            seq = result['released']
            assert [q['query_id'] for q in seq] == ['all_records', 'negative_records', 'all_records']
            assert seq[0]['released'] is True and seq[0]['aggregate_count'] == 110
            assert seq[2]['released'] is True and seq[2]['aggregate_count'] == 110
            should_release = (not must_deny) if mode in ('history_same_id', 'principal_history_reset_id') else True
            assert seq[1]['released'] is should_release, f'Unexpected rule {mode} in {case}'
            assert seq[1]['aggregate_count'] == 110 - total if should_release else seq[1]['aggregate_count'] is None
            assert result['inferred_restricted_count'] == total if should_release else result['inferred_restricted_count'] is None
            assert result['inference_exact'] is should_release
    for mode in expected_keys:
        successes = sum(int(c['conditions'][mode]['inference_exact']) for c in cases)
        summary = data['summary'][mode]
        assert successes == summary['exact_count_inference_cases']
        assert abs(successes / len(cases) - summary['exact_inference_rate']) < 1e-12
        assert successes == summary['second_query_released_cases']
        assert summary['harmless_repeat_released_cases'] == len(cases)
    return {'verified_cases': len(cases), 'conditions': len(expected_keys),
            'verified_queries': len(cases) * len(expected_keys) * 3, 'cases_sha256': digest}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('reports/results/sprint1/history_matrix.json'))
    args = p.parse_args()
    print(json.dumps(verify(args.input), indent=2))


if __name__ == '__main__':
    main()
