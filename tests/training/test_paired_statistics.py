from pathlib import Path
import json
import pytest
from scripts.analyze_balanced_study import analyze

ROOT = Path(__file__).resolve().parents[2]


def test_real_matched_study_analysis():
    out = analyze(ROOT / 'reports/results/sprint5/balanced_heterogeneity/summary.json')
    assert out['contrast_order'] == 'alpha_0.1_minus_iid'
    for metric in ('test_accuracy','worst_client_accuracy'):
        item=out['comparisons'][metric]
        assert len(item['paired_difference_pp_by_seed']) == 5
        assert 0 <= item['exact_sign_flip_two_sided_p'] <= 1
        assert item['t_interval_95_pp'][0] <= item['mean_difference_pp'] <= item['t_interval_95_pp'][1]


def test_missing_seed_fails_closed(tmp_path):
    original = json.loads((ROOT / 'reports/results/sprint5/balanced_heterogeneity/summary.json').read_text())
    original['aggregate']['completed_seeds'] = [241]
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='Five matched seeds'):
        analyze(path)
