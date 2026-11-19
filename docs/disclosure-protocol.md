# Sprint 3 disclosure protocol (v0.4) — executed pilot

## Observers and exact permissions

**Gradient observer:** A deliberately privileged *simulated coordinator* receives exact full gradients of a single public-data batch, model architecture and initial model weights. Batch 1 labels are **not** revealed; the attack infers them from the final-layer bias gradient (iDLG-style). Batch 8 labels **are** revealed, as an explicit stronger attacker privilege. Nothing here implies that a multi-epoch Flower/FedAvg server observes these gradients. The authentic Flower coordinator would see per-client **model updates** under the unprotected protocol. Neither SecAgg+ nor DP has been executed.

**External model observer:** Receives only softmax confidence vectors for an equal, class-matched number of training and held-out public-digit records. No member/nonmember tags, true labels, input pixels, logits, model weights or gradients are in the attack artifact. The model is a previously trained two-round public-digit FedAvg reference; output responses are offline replay, not a deployed prediction API. Attack heuristic is maximum confidence, without privileged calibration.

**Evaluator:** Holds original images or membership labels in separate `evaluator_only/` artifacts, optionally uses best-MSE Hungarian matching for batch reconstruction scoring, and writes scored outcomes. This is privileged scoring, not attacker information or guaranteed filesystem isolation. An OS/container-level access-control test remains pending.

## Hypotheses and measured configurations

- G1: Exact gradients on one 8×8 digit can be matched by input optimization under a known network and full gradient access. Adam minimizes squared gradient difference for 450 steps. Primary metrics: MSE, PSNR (unit data range), SSIM, runtime and gradient residual.
- G2: Adding independent Gaussian noise of standard deviation 0.01 to each gradient entry reduces empirical reconstruction fidelity versus the *same* exact gradient. Five paired public examples/seeds 231–235, no epsilon claim, no calibrated sensitivity, no accountant.
- G3: Batch 8 exact gradients and **known labels** permit some reconstruction; evaluator uses optimal assignment. This is a stronger-label-observation case, not directly comparable with unknown labels in batch 1.
- M1: Maximum confidence can distinguish model members from nonmembers. ROC-AUC, advantage, and empirical TPR at attainable FPR <= 0.01, 0.05 and 0.1; five query seeds test subsampling stability for **one frozen checkpoint**, not five independent training seeds.

## Reproduction (offline Python with `.[research,test]` installed)

```bash
python scripts/prepare_fl_data.py --output data/training/disclosure --seed 231 --limit 600
python scripts/run_gradient_inversion.py --site-file data/training/disclosure/institution-00/train.npz --output-dir reports/results/sprint3/gradient_b1 --batch-size 1 --seed 231 --steps 450
python scripts/run_gradient_inversion.py --site-file data/training/disclosure/institution-00/train.npz --output-dir reports/results/sprint3/gradient_b1_noisy --batch-size 1 --seed 231 --steps 450 --noise-std 0.01
python scripts/run_gradient_inversion.py --site-file data/training/disclosure/institution-00/train.npz --output-dir reports/results/sprint3/gradient_b8_known_labels --batch-size 8 --known-labels --seed 231 --steps 400
python scripts/run_disclosure_matrix.py --site-file data/training/disclosure/institution-00/train.npz --output-dir reports/results/sprint3/matrix --seeds 231 232 233 234 235 --steps 450
```

A reference FedAvg checkpoint is provided by `scripts/run_torch_reference.py` using the same partition parameters as in the archived v0.3 result. Do **not** test an unrelated checkpoint and present it as a trained-on-that-dataset membership attack.

```bash
python scripts/prepare_fl_data.py --output data/training/membership-reference --seed 20261118 --limit 600
python scripts/run_membership_inference.py --checkpoint reports/results/sprint2/checkpoints/digits-iid-smoke-v1_final.pt --data-dir data/training/membership-reference --output-dir reports/results/sprint3/membership_smoke
python scripts/run_membership_matrix.py --checkpoint reports/results/sprint2/checkpoints/digits-iid-smoke-v1_final.pt --data-dir data/training/membership-reference --output-dir reports/results/sprint3/membership_matrix
python scripts/plot_disclosure.py
```

The checkpoint used by membership tests was trained in v0.3 against the `20261118` public-digits split, and its recorded SHA is in `reports/results/sprint2/torch_iid_reference.json`.

## Interpretation and limits

- Exact-gradient recovery is not evidence of leakage through model updates, model releases or SecAgg+. Results here are on a tiny public-digit classifier. Generalization to realistic model architectures and batch schedules is unknown.
- The Gaussian perturbation is **not** DP. A proper DP-SGD baseline needs Opacus and an accountant; a query-statistics baseline needs OpenDP, explicit adjacency and sensitivity.
- No attack receives evaluator ground truth directly. The shared development filesystem does **not** enforce genuine attacker/evaluator access separation.
- Perfect gradient inversion of a small sample should not be generalized without trials, harder architectures and realistic observation assumptions.
- An AUC near 0.5 is a weak or negative attack result, not proof of membership privacy; additional attacks and repeated training seeds are needed.

## v0.4.1: actual per-client SGD model-delta observations (executed)

A second, more realistic *type of observation* is `server_individual_model_delta`: an unprotected coordinator sees initial parameters and the difference between initial and locally trained client parameters, plus a known SGD learning rate and local step count. An attacker computes `delta / (learning_rate * local_steps)` as a proxy gradient and runs the same reconstruction algorithm. That proxy is exactly equal to the first-step gradient only up to floating-point effects for **one step**. For multiple steps it is an *approximation* that can fail. The privileged setup records the exact first-step gradient only for a separate evaluator-only proxy-error metric, never in the attack observation.

Executed a same-image, same-model 1/4/16-step ablation using **one public digit repeatedly** in SGD. The label was inferred from the permitted delta-derived bias proxy in each case. Result: MSE 2.02613e-9, 0.0271682, 0.140351, respectively; average-gradient proxy relative L2 error 1.55e-7, 0.4757, 0.7569. These observations are real local SGD parameter changes, **not Flower execution** and **not typical multi-record client training**. This is deliberately an intermediate experiment between exact-gradient inversion and actual federated multi-minibatch updates.

Reproduce:

```bash
python scripts/run_update_inversion.py --site-file data/training/disclosure/institution-00/train.npz --output-dir reports/results/sprint3/model_update_inversion --local-steps 1 4 16 --attack-iterations 350 --seed 231
python scripts/plot_disclosure.py
```

Next: use actual per-client model updates from multi-example Flower local training and compare attack baselines under honest labels/steps and realistic, resource-matched scenarios.

## v0.5.0: distinct-record client-update inversion (executed)

To avoid the one-image-only limitation, the privileged setup trained on fixed
batches containing **1, 8 or 32 distinct approved public digit images**.
Each case produced a real initial-model-minus-final-model parameter delta
from either 1 or 4 full-batch SGD steps, and an attacker-observable initial
model and learning rate. Attacker code inferred a label only for batch=1;
**all true labels for batches 8/32 were explicitly disclosed** as an enhanced
attacker privilege. The public image truth and source row indices were saved
separately under `evaluator_only/`. The evaluator uses privileged optimal
assignment to score MSE and SSIM. No image truth is passed to the attack CLI.

| Distinct records | Local SGD steps | Mean reconstruction MSE | SSIM | Fraction MSE < 0.05 |
|---|---:|---:|---:|---:|
| 1 | 1 | 9.59e-9 | 1.000 | 100% |
| 8 | 1 | 0.00740 | 0.970 | 100% |
| 8 | 4 | 0.04949 | 0.725 | 75% |
| 32 | 1 | 0.06339 | 0.690 | 56.25% |
| 32 | 4 | 0.07000 | 0.672 | 43.75% |

**Status:** Single-seed pilot (seed 241), 160 reconstruction iterations,
initialization held fixed within each case. Larger batches and multiple
optimizer steps were harder *under these attack conditions*. These are still
fixed batches reused over the SGD steps, **not a changing mini-batch schedule
or actual Flower client**, and there is no calibrated DP or SecAgg+.
Known-label batch attacks are not directly comparable with batch=1 inferred
labels. The apparent recovery threshold is a predeclared illustrative MSE
cutoff, not a universal privacy metric or face-recognition benchmark.

Reproduction:

```bash
python scripts/prepare_fl_data.py --output data/training/digits-iid --seed 20261118 --limit 900
python scripts/run_multirecord_update_inversion.py --site-file data/training/digits-iid/institution-00/train.npz --output-dir reports/results/sprint3/multirecord_update --attack-iterations 160 --seed 241
python scripts/plot_new_benchmarks.py
```

Output includes each attack observation, separate ground truth, independent
evaluator score and `multirecord_update_summary.json`. The next methodological
step is inversion on genuine changing-minibatch FedAvg updates, with a
registered exact attacker observation and multiple seeds.
