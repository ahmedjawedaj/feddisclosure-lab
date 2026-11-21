import numpy as np
import pytest

from feddisclosure.training.lira_study import (lira_scores, lira_scores_global_var, logit_confidence,
                                               roc_summary)


def test_roc_summary_perfect_and_random():
    scores = np.array([3.0, 4.0, 0.0, 1.0])
    membership = np.array([1, 1, 0, 0])
    r = roc_summary(scores, membership)
    assert r["auc"] == 1.0 and r["tpr_at_fpr_0.05"] == 1.0
    rng = np.random.default_rng(0)
    s = rng.normal(size=4000)
    m = rng.integers(0, 2, size=4000)
    assert abs(roc_summary(s, m)["auc"] - 0.5) < 0.03


def test_lira_separates_shifted_gaussians_and_is_chance_when_identical():
    rng = np.random.default_rng(1)
    n, k = 300, 16
    member = rng.integers(0, 2, size=n)
    ins = [rng.normal(2.0, 1.0, size=k) for _ in range(n)]
    outs = [rng.normal(0.0, 1.0, size=k) for _ in range(n)]
    target = np.where(member == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n))
    assert roc_summary(lira_scores(target, ins, outs), member)["auc"] > 0.85
    same_in = [rng.normal(0.0, 1.0, size=k) for _ in range(n)]
    same_out = [rng.normal(0.0, 1.0, size=k) for _ in range(n)]
    target = rng.normal(0.0, 1.0, n)
    assert abs(roc_summary(lira_scores(target, same_in, same_out), member)["auc"] - 0.5) < 0.08


def test_logit_confidence_matches_definition():
    import torch
    from torch import nn

    class Fixed(nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 0.0, -1.0]]).repeat(len(x), 1)
    x = np.zeros((1, 2), dtype=np.float32)
    phi = logit_confidence(Fixed(), x, np.array([0]))
    p = np.exp(2) / (np.exp(2) + 1 + np.exp(-1))
    assert phi[0] == pytest.approx(np.log(p / (1 - p)), rel=1e-9)


def test_requires_both_classes():
    with pytest.raises(ValueError):
        roc_summary(np.array([1.0, 2.0]), np.array([1, 1]))


def test_global_variance_lira_separates_and_handles_missing_shadows():
    rng = np.random.default_rng(2)
    n, k = 200, 6
    member = rng.integers(0, 2, size=n)
    ins = [rng.normal(1.5, 1.0, size=k) for _ in range(n)]
    outs = [rng.normal(0.0, 1.0, size=k) for _ in range(n)]
    target = np.where(member == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n))
    assert roc_summary(lira_scores_global_var(target, ins, outs), member)["auc"] > 0.75
    empty = [np.array([]) for _ in range(n)]
    assert (lira_scores_global_var(target, empty, empty) == 0).all()
