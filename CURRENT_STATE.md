# CURRENT STATE — 2026-11-19

**Version:** 0.4.1. **Working baseline:** Sprint 0, Sprint 1 analytics, offline Sprint 2 FedAvg reference and first Sprint 3 model-disclosure experiments. **NOT a completed native Flower/Docker/DP release.**

## Executed here: analytical disclosure (Sprint 1)

Three separate FastAPI institution processes, three private SQLite databases and one HTTP gateway execute the allowlisted query benchmark. On 2026-11-19 the live local stack repeated the naive `110 - 107 = 3` restricted-group inference; the history-aware policy denied the second query. Result: `reports/results/sprint1_local_v0.4_validation.json`. Static Compose safety contract also passes. The native Compose runtime is **NOT VERIFIED**: `docker` binary is absent; actual verifier returned exit 2. See `reports/results/runtime_gates/status.json` and logged stderr.

## Executed here: PyTorch federated-training reference (Sprint 2)

- The CPU reference uses separate, concurrent OS processes for institutional clients and an independent evaluator, private public-digit NPZ partitions, immutable SHA-256 checks, FedAvg weighted model aggregation, and final global per-client validation. This is **NOT Flower** and its processes do not enforce filesystem isolation.
- Ten paired reference runs: 5 seeds each for IID and severe label-skew partitions, 3 clients, 600 selected public 8×8 digit images, 2 rounds. Mean public test accuracy **0.2000 ± 0.06095 SD** IID and **0.17833 ± 0.05420 SD** non-IID. Mean worst-client global-model validation **0.11875** IID and **0.0000** severe non-IID. Short, weak convergence smoke results, not a general effect or statistical significance claim.
- One-seed client-count pilot: 1 client IID public accuracy 0.3833; 5 clients IID 0.1750, severe non-IID 0.1833. **The different client counts do not have matched optimizer steps; do not compare these as fair efficiency or privacy effects.** Two initial timed-out five-client attempts were resolved by limiting simultaneous worker processes to 3. One institution cannot provide secure aggregation concealment.
- Schema-v2 manifest now removes coordinator-readable `label_histogram` fields, saving distribution statistics in evaluator-only files. New training refuses legacy v1 manifests. Flower ClientApp additionally validates client partition hashes. A tampered client-partition test passes (fails closed). `scripts/run_training_sweep.py` can resume previously saved run JSONs without losing matching-condition summaries.
- Evidence: `reports/results/sprint2/sweep/sweep_summary.json`, ablations under `reports/results/sprint2/`, figure `reports/figures/sprint2_five_seed_accuracy.png`; full public-digit checkpoints and hashes in the release archive.
- Native Flower ServerApp/ClientApp source and CI exist, but `flwr` is absent and the Python package network is unavailable in this environment. `scripts/run_flower_experiment.py` exited 1 with a documented dependency error; **NO native Flower run or one-round Flower aggregation cross-check has occurred**.

## Executed here: controlled model disclosure (Sprint 3)

**Gradient inversion.** Implemented exact-gradient observation construction from approved public data in a privileged setup role, an independent DLG-style Adam optimization attack and an independent image evaluator. Batch 1 uses inferred label (iDLG-style output-layer bias rule), 450 steps; batch 8 uses explicitly known labels, 400 steps. Five matched single-example exact-vs-Gaussian-noise seed pairs executed. Exact mean MSE **2.42678e-9**; manually added noise 0.01 mean MSE **0.00121135**. The single-seed batch-8 reconstruction MSE was 0.010676. **No privacy guarantee or DP claim applies to the noise.** This measures highly privileged exact first-step gradients, not multi-step FL updates or SecAgg+ observations.

**Output-only membership inference.** On the frozen two-round public-digit reference checkpoint, the attacker saw only class confidence vectors, scoring maximum confidence; the evaluator alone saw balanced class-matched membership truth. The first pilot ROC-AUC was 0.5475. Across five **query-subset** seeds (one fixed model), mean ROC-AUC **0.47811 ± 0.01832 SD**. This is near-chance/negative evidence, not proof of model privacy, and not five independently trained models.

**Actual local-update inversion (additional measured loop):** The simulated unprotected coordinator receives an initial model and an actual SGD client parameter delta, *not* an exact gradient. When the attacker divides that delta by the known learning rate and number of local steps, one-step inversion MSE was **2.02613e-9**, four-step MSE **0.0271682**, sixteen-step MSE **0.140351**; the proxy-gradient relative L2 error increased from **1.55e-7** to **0.4757** and **0.7569**. Same public digit and initialization across cases, one record reused for local SGD (a deliberately idealized tiny client). The label was inferred in all three cases. This is not a realistic multi-record Flower update, and does not involve secure aggregation. Full `reports/results/sprint3/model_update_inversion/` evidence and a generated figure are included.

- Evidence: `reports/results/sprint3/gradient_b1/`, `gradient_b1_noisy/`, `gradient_b8_known_labels/`, `matrix/matrix_summary.json`, `membership_smoke/`, `membership_matrix/membership_matrix_summary.json`. `scripts/plot_disclosure.py` regenerated three figures from original measured artifacts.
- Attack CLI arguments cannot include original public images or membership truth. These live under `evaluator_only/` and are passed only to the evaluation subprocess. This is a code/interface boundary, NOT an OS security boundary.

## Test and release evidence

- Executed `pytest -q -m 'not slow'`: **40 passed, 1 optional Flower import skipped, 2 slow deselected**.
- Ran each of the two slow three-client subprocess integration tests independently: **both passed**, each with successful exit code. An unsplit full-suite command timed out during subprocess tests; do not claim that invocation passed.
- Ran the live four-process analytics harness and research evidence checker successfully. Recorded structured failed attempts for unavailable Docker/Flower gates. Three new figures generated from real saved results.
- Git history includes baseline commits, sprint hardening, disclosure work, tests and agent-skill handoffs. Files/data are approved public or synthetic only. No GitHub remote/Actions execution is claimed.

## Known limitations and next steps

Docker, Flower, MNIST network download, five independently trained membership models, realistic multi-record/multi-minibatch FedAvg update inversion, OpenDP, Opacus, SecAgg+, dashboard and digital-twin validation remain **unimplemented, blocked, or unverified**. The short training baseline has poor quality, and attacker processes share a development filesystem. Revisit after native runtime gates. `NEXT_SPRINT.md`, `docs/disclosure-protocol.md`, `docs/fl-experiment-protocol.md`, and `AGENTS.md` are the next agent's first reads.
