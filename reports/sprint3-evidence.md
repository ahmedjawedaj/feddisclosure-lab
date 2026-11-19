# Sprint 3 first disclosure benchmark evidence — 2026-11-19

**Status:** Executed controlled public-digits disclosure attacks with separately invoked attacker/evaluator CLIs. This is **not a model-update inversion, a native Flower attack, a privacy guarantee or a publication claim**. Full threat model: `docs/disclosure-protocol.md`.

## Gradient inversion

Adam input optimization to minimize squared discrepancy from an exact first-step gradient in a known MLP. Single-example label inferred (not supplied); batch 8 uses explicitly revealed labels. Original images are held in evaluator-only artifacts. The multi-image evaluator uses optimal MSE assignment, an evaluation privilege unavailable to the attacker.

| Setting | Public image seeds | Mean image MSE | Notes |
|---|---|---:|---|
| Single example, exact gradient | 231–235 | 0.000000002427 | 450 optimization steps, inferred labels |
| Single example, exact gradient plus Gaussian noise 0.01 | 231–235 (matched) | 0.00121135 | empirical noise ablation, **not DP** |
| Eight images, exact batch gradient | 231 | 0.0106761 | 400 steps, **known labels** |

The exact-gradient case reconstructs these simple public digits nearly perfectly. This cannot be transferred to a model-update-only or secure-aggregation observer. A learned model with several local steps can produce different information and dynamics. Noise magnitude is not calibrated to sensitivity or accounted; no epsilon/delta claim exists. Source: `results/sprint3/matrix/matrix_summary.json` and individual `gradient_b*/evaluation.json` files.

## Output-only membership inference

One frozen public-digit FedAvg reference checkpoint (two rounds) was queried on class-matched balanced public member/non-member records. External user receives softmax scores only, not data or membership labels; attack ranks by max confidence. The evaluator alone computes ROC metrics.

| Query setting | ROC-AUC |
|---|---:|
| Initial pilot | 0.5475 |
| Five query-sampling seeds (same fixed model) | Mean 0.47811, sample SD 0.01832 |

These are **not five training repeats** and do not establish that the model provides membership privacy. With uncalibrated max-confidence inference on a poorly converged model, the signal is essentially absent. Stronger attacks, multiple trained models and stable high-utility models are needed before scientific interpretation. Source: `results/sprint3/membership_matrix/membership_matrix_summary.json`.

## Reproducibility and boundary assertions

- Attack scripts accept **observation** files, never original-image or membership-label paths; privileged setup/evaluator process handles ground truth. This is a software convention, not a hardened container/OS boundary.
- Attack-input validation rejects oracle-bearing fields; corrupted client partitions fail before producing any model update.
- All artifacts and figures derive from executed configurations, not illustration placeholders. Plots regenerate using `python scripts/plot_disclosure.py`.
- Scoring records unsuccessful attacks, uncertain low-FPR results and matched noise scenarios separately; it never invents a combined privacy score.
- Flower `flwr` package and Docker Engine were unavailable: `results/runtime_gates/status.json` documents failed native verification.


## Additional model-update observation experiment (v0.4.1)

Rather than giving the attacker an exact initial gradient, the simulated coordinator provides an actual individual client's model **delta** after 1, 4 or 16 SGD steps, plus known optimizer details. The attacker attempts input reconstruction from `delta / (learning_rate * steps)`. No exact intermediate gradient is observable to the attacker. Exactly one same public image is used repeatedly for these controlled local updates.

| Local SGD steps | Proxy-gradient relative L2 error | Image reconstruction MSE |
|---:|---:|---:|
| 1 | 1.55e-7 | 2.02613e-9 |
| 4 | 0.4757 | 0.0271682 |
| 16 | 0.7569 | 0.140351 |

The diminishing quality under multi-step local training is a *single-example pilot*, not evidence of privacy from FedAvg, especially when other attack methods or repeated examples are available. Artifacts: `results/sprint3/model_update_inversion/update_attack_summary.json` and figure `figures/sprint3_model_update_inversion.png`.

The gradient-input attack remains a distinct observer contract; no exact-gradient attack result is silently reclassified as a client-update attack.
