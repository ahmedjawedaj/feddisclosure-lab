# Changing-minibatch client-update privacy experiment

## Scope

The unprotected honest-but-curious coordinator receives one **individual-client model delta**, initial weights, learning rate and step count. In these privileged experiments it receives **all contributing class labels**. It does not receive images, sample identities, ground truth or the local SGD minibatch order. It cannot reconstruct from individual updates when Flower SecAgg+ actually prevents their exposure.

## Research loop

1. Define batch size, local steps, target unique records, label-knowledge budget and evaluator metric.
2. Run `scripts/run_minibatch_update_inversion.py` on an approved public `train.npz` into a fresh folder.
3. Verify only `attacker/observation.pt` is passed to `feddisclosure.attacks.gradient_cli`; `evaluator_only/original.npz` goes solely to `evaluation.gradient_cli`.
4. Record reconstruction MSE, SSIM, PSNR and gradient-proxy relative L2 error. Negative outcomes are evidence.
5. Run `pytest -q tests/training/test_minibatch_update.py` and `python scripts/verify_v060_evidence.py` against the complete release.
6. Append findings and caveats to `docs/disclosure-protocol.md`, `CURRENT_STATE.md` and `NEXT_SPRINT.md`.

**Known limitation:** this is local SGD across disjoint batches, not a native Flower client. Do not compare it as a compute-matched substitute for fixed full-batch repeated-SGD experiments or claim differential privacy.
