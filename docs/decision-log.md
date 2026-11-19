# Engineering decision and evidence log

## 2026-11-18 — Sprint 0 and early Sprint 1

Implemented typed count APIs, per-institution persistent SQLite data, minimum-cell policy, limited membership-based query history, three-institution gateway, attacker/evaluator separation, YAML config, local process integration harness, pytest tests, CI and agent handoff. Verified non-containerized four-service run and saved machine-readable benchmark. Docker Engine unavailable in this execution environment; Compose is prepared but unverified here.

## 2026-11-18 — Verified reproducibility closure

Expanded tests to 18, including gateway fail-closed behavior, permission checks, exact expected artifact contents and agreement across two fully independent four-process runs. Installed the package in editable mode and verified CLI. Created `reports/figures/sprint1_analytics.png` from measured JSON, and added machine-readable tracked baseline artifacts. Docker Compose remains unverified because Docker Engine is unavailable here.

## 2026-11-18 — Sprint 1 guardrail hardening (commit after tests)

Added metadata-only per-site query audit (including denied releases), serializable SQLite transaction concurrency test with twenty parallel clients, gateway unreachable-site fail-closed regression, native Compose verification script for volume/network/port inspection. No Docker daemon installed in this environment, so Compose execution is still explicitly pending. Analytics HTTP integration rerun maintained 110/107/3 and history rejection.

## 2026-11-18 — Sprint 2 measured PyTorch reference and native Flower implementation (commit after tests)

Implemented versioned public digits IID and severe label-skew partition manifests, immutable site-owned NPZ files, PyTorch model/training and weighted FedAvg math, three concurrent independent client processes, evaluator-only public test and pooled/local-only references, model-update byte estimates, and final per-client validation of aggregated weights. Fixed a role boundary issue during review by moving public test scoring into a separate evaluator process and restricting coordinator to metadata/model arrays. Exact IID repeat results matched. Figure and JSON artifacts persisted under `reports/`. Native Flower 1.27 ServerApp/ClientApp and a CI integration runner are implemented against official Message API documentation, but **the Flower package could not be fetched in this network-restricted environment**. Neither native Flower nor Docker executed; runtime gates remain pending.

## 2026-11-19 — Sprint 2 multiple-seed research and metadata boundary hardening

Ran **ten** three-client two-round offline PyTorch FedAvg references: five matched source/test seeds each for IID and severe label-skew data. Global public accuracy means 0.2000 (IID, sd 0.06095) and 0.17833 (skew, sd 0.05420). Worst-client global validation means 0.11875 and 0.0000. Client-count ablation also ran one seed for 1 IID client and for 5 IID/non-IID clients. This is pilot evidence; models were not trained to convergence and per-client optimizer steps differ by client count. Fixed a data-governance flaw: coordinator-readable schema-v1 manifests exposed site label histograms. Schema-v2 moves histograms into evaluator-only ground-truth metadata and refuses legacy manifests. Flower client checks per-partition hashes; worker tampering test fails closed. Native Docker and Flower runtime gates remain unavailable and **unverified**.

## 2026-11-19 — Sprint 3 observer-controlled model-disclosure experiments

Implemented DLG-style Adam input optimization, iDLG-style single-label recovery, exact-gradient observation capture on approved public digits, batch-8 known-label case, paired noise ablation, privileged optimal-matching MSE/PSNR/SSIM evaluation and strictly separated attack/evaluator CLIs. Five exact/noisy paired image seeds executed: mean reconstruction MSE 2.4268e-9 exact and 0.00121135 after uncalibrated Gaussian noise standard deviation 0.01. **Noise is not formally private.** Independently implemented confidence-score model-output membership inference and evaluated the frozen v0.3 two-round reference model. Five query seeds yielded mean AUC 0.47811 (sd 0.01832), a weak/negative result. These are logically separated artifacts and subprocesses, not OS-level isolation. Created actual figures from the recorded result files, documented threat privileges and reproducibility steps, and added regression tests and a not-yet-executed CI workflow.


## 2026-11-19 — v0.4.1 local-model-update leakage bridge

Added a distinct unprotected model-delta observer with initial model parameters, actual local SGD parameter delta, known learning rate and step count. Attack code converts the delta to a proxy gradient; no original public image or exact intermediate gradients are available to the attack. Measured one public digit repeatedly optimized for 1, 4, and 16 steps: MSE 2.02613e-9, 0.0271682, 0.140351; approximation L2 errors 1.55e-7, 0.4757 and 0.7569. This bridges the exact-gradient demonstration toward server-visible FL updates but is **not yet multi-example/multi-minibatch Flower training**. All three attack/evaluator artifact pairs and generated figure are preserved. Native Flower/Docker gates remain pending.

## 2026-11-19 — v0.5.0 distinct-record update inversion and FedProx comparison

Expanded privileged observed client model delta to fixed local batches of 1,
8 and 32 *different* public digits, with one/four SGD steps. The attacker sees
only model state/delta and declared hyperparameters plus explicit known labels
for batches >1. Evaluator-only truth/assignment remain separate. Five
experiments completed and were scored; larger batch and four-step cases show
higher reconstruction MSE, but fixed-batch and one-seed limitations remain.

Implemented FedProx loss in the shared PyTorch training path (and prepared the
native Flower ClientApp configuration), tested `mu=0` equivalence and positive
penalty reduction of parameter drift. Three matched-seed non-IID PyTorch
FedAvg/FedProx studies executed with identical data, model and optimizer budget:
mean global accuracy 55.19% versus 49.07%, paired change -6.11 pp, with zero
worst-site validation accuracy for either method. Saved six checkpoints,
hashes, per-round metrics and a new figure. Retained the negative result.

Portability issue found on clean runtime: tests spawn worker modules that
require an installed package; used `pip install -e . --no-build-isolation
--no-deps` since build isolation tries network DNS here. Fast tests and
integration tests were executed separately. Docker binary absent; native
Flower dependency unavailable; these gates remain explicitly unverified.

## 2026-11-19 — v0.6.0 practical experiments and unresolved native gates

Added disjoint changing-minibatch client-model-delta observer with known-label capability explicitly logged, three executed attack/evaluator trials and distinct attacker/evaluator artifacts. Expanded the output-only attack to three **predeclared** scores and ran all nine combinations on three independently trained 4-round FedAvg models, preserving weak and below-chance outcomes. Extended the two methods' severe-non-IID training to eight rounds and added class-conditional Dirichlet(alpha=0.5) partitions; recorded both observed improvement and the unequal-sample/optimizer-steps confound. Added a fail-closed native Flower checkpoint parity checker/CI requirement but **did not** run Flower or Docker in this environment. Formal DP and SecAgg+ remain future stages. The research methodology prioritizes a matched multiple-seed partition study before any generalization or thesis claim.

**v0.6 closure:** 55 fast tests and two separately invoked slow worker tests passed; live HTTP analytics disclosure remained 110/107/3. Baseline and new research artifact hashes/metrics were checked. Attempting the actual Docker verifier returned 2 (no Docker binary), and the Flower runner returned 1 (missing `flwr` module); both were retained as explicit unsuccessful runtime gates. Do not reinterpret the Flower parity fixture unit tests as real Flower execution.
