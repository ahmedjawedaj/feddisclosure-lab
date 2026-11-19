# FedDisclosure model-output leakage research

## Purpose

Run membership-inference baselines under **external output-only user** observations.
Read `docs/threat-models.md`, `docs/disclosure-protocol.md`, and `CURRENT_STATE.md` first.

## Invariants

- Attacker input is an NPZ containing ONLY `probabilities`; no query labels, input images, member flags, weights or training indices.
- The evaluator alone creates balanced class-matched queries and knows membership truth.
- Separate independent *model training seeds* from query-subsample seeds. Their variances are not interchangeable.
- Pre-register output-only heuristics and report **all** ROC-AUC results, including values below chance, TPR at achievable empirical FPR, and sample sizes.
- ROC-AUC around 0.5 is a weak result, not evidence that the model is formally private.
- Preserve checkpoint hashes, data manifest versions and which public samples were used to train. Never use real confidential records.

## Reproduction

Full extracted evidence bundle: `python scripts/verify_v060_evidence.py`.
To generate independent-model results, first create the three paired FedAvg public-digit checkpoints in `reports/results/sprint2/ablations/paired/checkpoints/` using `scripts/run_fedprox_ablation.py` if not present.
Then: `python scripts/run_membership_training_seeds.py --seeds 241 242 243 --output-dir reports/results/sprint3/membership_training_seeds`.
Use a **fresh output directory**; the runner refuses overwrite.

## Deliverables

Attacker-only score files, evaluator-only member labels, per-run method metrics and the aggregate JSON. Keep raw NPZ files out of Git. Include limitations, failed cases and independent test evidence in `CURRENT_STATE.md` and the research report.
