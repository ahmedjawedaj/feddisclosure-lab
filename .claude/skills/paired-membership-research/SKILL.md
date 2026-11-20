---
name: paired-membership-research
description: Compare output-only membership attacks on independently trained federated checkpoints with matched query records and honest training-cohort limitations.
---

# Paired-query membership evaluation

1. Read `docs/v0.10-results.md`, `src/feddisclosure/evaluation/matched_membership_observer.py`, `scripts/run_balanced_membership.py` and `scripts/verify_balanced_membership.py`.
2. Confirm checkpoints, source metadata manifests, per-institution data hashes, independent training seeds, and public held-out test cohort.
3. **Critical:** IID and severe non-IID training membership sets differ. Evaluate on shared member examples and identical nonmember examples for each seed, but never claim the training cohorts are identical or the observed AUC difference is causal.
4. Attacker receives only `model_predictions.npz` with the `probabilities` array; no training images, class labels, model weights or membership truth. Save evaluator-only membership labels separately.
5. Score all three predeclared confidence heuristics, ROC-AUC, attack advantage and empirical low-FPR TPR; report negative results and exploratory five-seed confidence intervals.
6. Verify full evidence using `python scripts/verify_balanced_membership.py`, preserve hashes and do not rerun over immutable completed experiments. No result here is a formal privacy guarantee or a native Flower execution.
