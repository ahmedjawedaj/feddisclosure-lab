"""Evaluator-only paired membership queries for independently trained public models.

Both classifiers are queried on exactly the same MEMBER records present in
both training sets, and exactly the same held-out NONMEMBER records. This does
NOT make the training cohorts identical, nor identify a causal privacy effect.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from feddisclosure.training.task import DigitMLP


def _read(site_files: list[Path]) -> dict[int, tuple[np.ndarray, int]]:
    records = {}
    for file in site_files:
        with np.load(file, allow_pickle=False) as data:
            assert {'x','y','source_idx'}.issubset(data.files)
            x, y, ids = data['x'], data['y'], data['source_idx']
        for key, row, label in zip(ids, x, y):
            idx = int(key)
            if idx in records:
                raise ValueError('Duplicate source ID in client data')
            records[idx] = (row, int(label))
    return records


def paired_predictions(checkpoints: dict[str, Path], site_files: dict[str, list[Path]],
                       test_files: dict[str, Path], output: Path,
                       *, seed: int, per_class: int = 6) -> dict:
    if set(checkpoints) != {'iid','dirichlet_balanced_0.1'} or per_class < 1:
        raise ValueError('Two configured matched conditions and positive per-class required')
    first, second = list(checkpoints)
    train = {name: _read(site_files[name]) for name in checkpoints}
    tests = {name: _read([test_files[name]]) for name in checkpoints}
    common_members = set(train[first]) & set(train[second])
    if set(tests[first]) != set(tests[second]):
        raise ValueError('Test source cohort mismatch')
    if common_members & set(tests[first]):
        raise ValueError('Member and nonmember sources overlap')
    for idx in common_members:
        a,b = train[first][idx],train[second][idx]
        if a[1] != b[1] or not np.array_equal(a[0], b[0]):
            raise ValueError('Shared member source has different content')
    for idx in tests[first]:
        a,b = tests[first][idx],tests[second][idx]
        if a[1] != b[1] or not np.array_equal(a[0], b[0]):
            raise ValueError('Shared public nonmember source has different content')
    rng = np.random.default_rng(seed)
    member_ids, nonmember_ids = [], []
    for cls in range(10):
        eligible_members = sorted(i for i in common_members if train[first][i][1] == cls)
        eligible_nonmembers = sorted(i for i,(_,y) in tests[first].items() if y == cls)
        amount = min(per_class, len(eligible_members),len(eligible_nonmembers))
        if amount:
            member_ids.extend(rng.choice(eligible_members,amount,replace=False).tolist())
            nonmember_ids.extend(rng.choice(eligible_nonmembers,amount,replace=False).tolist())
    if not member_ids or len(member_ids) != len(nonmember_ids):
        raise ValueError('Cannot create class-matched membership queries')
    X=np.stack([train[first][idx][0] for idx in member_ids]+[tests[first][idx][0] for idx in nonmember_ids])
    truth=np.concatenate((np.ones(len(member_ids),dtype=np.int64),np.zeros(len(nonmember_ids),dtype=np.int64)))
    order=rng.permutation(len(truth))
    ids=[*member_ids,*nonmember_ids]
    query_hash=hashlib.sha256(json.dumps([ids[i] for i in order]).encode()).hexdigest()
    torch.set_num_threads(1)
    for name, checkpoint in checkpoints.items():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=DigitMLP(features=X.shape[1],classes=int(state['network.2.bias'].numel()))
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probabilities=torch.softmax(model(torch.tensor(X[order],dtype=torch.float32)),dim=1).numpy()
        root=output/name/f'seed-{seed-1000}'
        (root/'attacker').mkdir(parents=True,exist_ok=True)
        (root/'evaluator_only').mkdir(parents=True,exist_ok=True)
        np.savez_compressed(root/'attacker'/'model_predictions.npz',probabilities=probabilities)
        np.savez_compressed(root/'evaluator_only'/'membership_labels.npz',membership=truth[order])
        (root/'observation_contract.json').write_text(json.dumps({
            'observer':'external_model_user_softmax_only',
            'attacker_receives':['probabilities'],
            'attacker_known_labels':False,
            'matched_query_cohort_sha256':query_hash,
            'query_count':len(X),
            'member_count':len(member_ids),
            'nonmember_count':len(nonmember_ids),
            'matched_overlapping_train_records':len(common_members),
            'limitation':'Training membership sets differ across partitions; paired queries do not create identical training cohorts',
        },indent=2)+'\n')
    return {'query_count':len(X),'member_count':len(member_ids),
            'nonmember_count':len(nonmember_ids),
            'matched_query_cohort_sha256':query_hash,
            'common_member_pool_size':len(common_members),
            'full_train_cohorts_identical':set(train[first])==set(train[second])}
