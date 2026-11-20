from pathlib import Path
import json
import numpy as np
import torch

from feddisclosure.training.task import DigitMLP
from feddisclosure.evaluation.matched_membership_observer import paired_predictions
from scripts.run_balanced_membership import confidence_interval


def test_paired_membership_observation_identical_cohort_and_hidden_truth(tmp_path):
    rng=np.random.default_rng(42)
    states={}
    sites={}
    tests={}
    for j,condition in enumerate(('iid','dirichlet_balanced_0.1')):
        indices=np.arange(30) if j==0 else np.arange(5,35)
        # Same source ID has identical content in both partitions.
        x=np.stack([np.full(64,int(i)/100,dtype=np.float32) for i in indices])
        y=(indices % 2).astype(np.int64)
        loc=tmp_path/condition
        loc.mkdir()
        site=loc/'train.npz'
        np.savez_compressed(site,x=x,y=y,source_idx=indices)
        sites[condition]=[site]
        test=loc/'test.npz'
        test_idx=np.arange(60,80)
        tx=np.stack([np.full(64,int(i)/100,dtype=np.float32) for i in test_idx])
        np.savez_compressed(test,x=tx,y=(test_idx%2).astype(np.int64),source_idx=test_idx)
        tests[condition]=test
        torch.manual_seed(j+1)
        model=DigitMLP()
        ck=loc/'checkpoint.pt'
        torch.save(model.state_dict(),ck)
        states[condition]=ck
    contract=paired_predictions(states,sites,tests,tmp_path/'output',seed=1241,per_class=3)
    assert contract['member_count']==6 and contract['nonmember_count']==6
    assert contract['full_train_cohorts_identical'] is False
    for condition in states:
        base=tmp_path/'output'/condition/'seed-241'
        with np.load(base/'attacker'/'model_predictions.npz',allow_pickle=False) as data:
            assert data.files==['probabilities']
            assert data['probabilities'].shape==(12,10)
        with np.load(base/'evaluator_only'/'membership_labels.npz',allow_pickle=False) as data:
            assert data.files==['membership']
        meta=json.loads((base/'observation_contract.json').read_text())
        assert meta['attacker_known_labels'] is False
        assert 'source_idx' not in str(meta)


def test_interval_has_nonzero_width_on_variable_samples():
    lower,upper=confidence_interval([0.45,0.50,0.55,0.52,0.48])
    assert lower<0.5<upper
