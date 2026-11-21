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
