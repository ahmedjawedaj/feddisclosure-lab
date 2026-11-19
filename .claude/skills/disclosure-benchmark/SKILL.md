# Disclosure Benchmark Skill

Use this when designing or extending controlled gradient inversion, output-only
membership inference, and matched noise/clipping ablations.

1. Read `docs/threat-models.md`, `docs/methodology.md`, `docs/disclosure-protocol.md` and `CURRENT_STATE.md`.
2. Name the observer and enumerate exactly what the attacker receives. Distinguish exact single-step gradient from a multi-step FL model update and from aggregate secure-aggregation output.
3. Prepare only approved synthetic/public records in a privileged setup process. Write attacker observations under `attacker/`; store originals, membership labels, and matching permutations under `evaluator_only/`.
4. Run attack using only observation files; run evaluator separately. Add negative-case tests that reject oracle fields and invalid probabilities.
5. Record MSE, PSNR, SSIM, matching budget, label privileges, runtime, and membership ROC-AUC/TPR@FPR as appropriate. Preserve failures.
6. For perturbation experiments, match public image/model/seed. DO NOT call Gaussian noise or clipping DP without proper sensitivity, sampling, adjacency, composition and accounting.
7. Save JSON and figures from real runs, execute `pytest -q -m 'not slow'`, update handoff documents, and commit measured changes.

See `scripts/run_gradient_inversion.py`, `scripts/run_disclosure_matrix.py`, `scripts/run_membership_inference.py`, and `scripts/run_membership_matrix.py`.
