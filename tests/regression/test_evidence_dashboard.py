import json
from pathlib import Path
import pytest
from dashboard.evidence import make_snapshot
from dashboard.build_dashboard import build


def test_scalar_snapshot_and_offline_html(tmp_path):
    snapshot = make_snapshot()
    assert snapshot['data_scope'] == 'reviewed_scalar_evidence_only'
    assert len(snapshot['heterogeneity']) == 4
    assert len(snapshot['collusion']) == 3
    flags = snapshot['native_protocol_verified']
    if snapshot['v014'] is None:
        assert not any(flags.values())
    else:
        assert all(flags.values())
        assert snapshot['v014']['secagg']['victim_error_n_minus_1'] < snapshot['v014']['secagg']['victim_error_server_only']
        assert snapshot['v014']['opendp_ledger_passed'] is True
    output = build(destination=tmp_path / 'index.html')
    html = output.read_text()
    assert '<!doctype html>' in html
    assert 'Aggregate disclosure' in html
    assert 'Defenses (v0.14)' in html
    assert 'type="application/json"' in html
    assert 'localhost:' not in html
    assert '<script src=' not in html
    assert 'source_idx' not in html
    saved = json.loads((tmp_path / 'evidence_snapshot.json').read_text())
    assert saved == snapshot
    assert not 'evaluator_only' in str(saved)


def test_fail_closed_missing_summary(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_snapshot(tmp_path)


def test_completion_evidence_keeps_failures_visible():
    v014 = make_snapshot()['v014']
    if v014 is None or v014.get('completion') is None:
        pytest.skip('completion evidence not present')
    c = v014['completion']
    assert len(c['dropout']) == 5 and all(x['prediction_matches'] for x in c['dropout'])
    assert {x['outcome'] for x in c['dropout']} == {'TOLERATED', 'HALTED'}
    parity = {x['name']: x['parity_1e-5'] for x in c['mnist_native']}
    assert parity['run-mnist-full-iid'] is True
    assert parity['run-mnist-full-labelskew'] is False and parity['run-mnist-full-iid-seed3'] is False
    assert all(x['bitwise_flower_replay'] for x in c['mnist_native'])
    assert c['n4_two_round']['runs_above_naive_tolerance'] == ['baseline']
    assert all(x['exit_code'] == 125 for x in c['fail_fast'])


def test_inversion_rows_show_the_label_floor_and_the_renamed_mean_image_floor(tmp_path):
    vc = make_snapshot()['v014']['completion']['view_comparison'] if make_snapshot()['v014'] else None
    if vc is None:
        pytest.skip('completion evidence not present')
    lf = vc['label_floor']
    assert 0.0 < lf['mse'] < vc['mean_image_floor_ignores_labels_mse']
    assert lf['public_pool_size'] == 1765 and lf['client_records'] == 32
    assert [r['pool_size'] for r in lf['pool_size_sensitivity']] == [20, 50, 100, 300, 1765]
    assert vc['views']['plaintext_individual']['rate_minus_label_floor']['low'] > 0
    assert vc['views']['secagg_aggregate_maxw1000']['rate_minus_label_floor']['high'] < 0
    html = build(destination=tmp_path / 'index.html').read_text()
    assert 'Label-only floor' in html and 'ignores labels' in html and 'prior-knowledge reference' in html
    assert 'no-information floor' not in html
    assert 'naive attacker only, not a protection claim' in html
