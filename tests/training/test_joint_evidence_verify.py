import json
import shutil
import pytest
from scripts.verify_joint_aggregate import verify
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_joint_saved_evidence():
    raw = ROOT/'reports/results/sprint4/joint_aggregate_oracle/seed241/aggregate_joint_oracle/attacker/recovered.npz'
    if not raw.exists():
        pytest.skip('Full researcher evidence bundle not present in Git checkout')
    report=verify(ROOT/'reports/results/sprint4/joint_aggregate_oracle')
    assert report['integrity']=='passed'
    assert report['n_seeds']==3


def test_joint_hash_corruption_is_rejected(tmp_path):
    src=ROOT/'reports/results/sprint4/joint_aggregate_oracle/seed241'
    if not (src/'aggregate_joint_oracle/attacker/recovered.npz').exists():
        pytest.skip('Full researcher evidence bundle not present in Git checkout')
    shutil.copytree(src,tmp_path/'seed241')
    evidence=tmp_path/'seed241/joint_summary.json'
    s=json.loads(evidence.read_text())
    s['recovered_sha256']='0'*64
    evidence.write_text(json.dumps(s))
    with pytest.raises(AssertionError):
        verify(tmp_path,seeds=(241,))
