# v0.11.0 — Verified retry safety, native certification preparation (2026-11-20)

- **Measured:** Three-site fault-injection HTTP benchmark, persistent idempotent DP query retry and conflict rejection. `reports/results/sprint4/retry_benchmark.json` uses a deterministic TEST mechanism. It offers NO differential-privacy guarantee.
- **Implemented but NOT executed natively:** Upstream Flower SecAgg+ app, strict native CI gate, hardened Flower evidence runner and parity checker. Docker, Flower, OpenDP and Opacus are absent here.
- **Verified:** 115 passing fast tests after the final changes, two independent-worker training integration passes, live analytics benchmark, historical evidence verification. See `docs/v0.11-runtime-and-retry.md` and the latest saved verification logs for final counts.
- **Action required to run hosted native CI:** Configure a dedicated GitHub remote for this repository (none is present) and run `.github/workflows/native-v1-certification.yml` on GitHub's Ubuntu runner. Never claim 100% or formal DP/SecAgg+ before actual native results.

## v0.10.0 handoff priority (2026-11-20)

Read `docs/v0.10-results.md`, `CURRENT_STATE.md` and `NEXT_SPRINT.md` first. The actual three-site HTTP test demonstrates the experiment-ID reset vulnerability (110 − 107 = 3), and validates persistent `principal_history` enforcement (second query 409, policy switching 403). A 90-case × 4-policy × 3-query SQLite study and 10-checkpoint/30-evaluation output-only MIA comparison have executed with independent verifiers. The matched MIA *query* cohorts are identical, but the underlying training data membership differs across IID and severe non-IID. Never claim causal privacy benefit, formal DP, SecAgg+, native Docker or Flower verification. Next native gates remain P0. New skills: `.agents/skills/principal-history-control/`, `.agents/skills/paired-membership-research/`, mirrored for Claude.

## v0.8.0 handoff priority

Read `CURRENT_STATE.md`, `NEXT_SPRINT.md`, `docs/v0.8-research-protocol.md` and `docs/v0.8-results.md`. Fifteen executed idealized 3-client aggregate-observer/collusion reconstructions and 5-seed integrity hashes are available. These are **NOT Flower SecAgg+**, and the aggregate-only attack is a misspecified target-only probe with known victim labels. Native Opacus DP-SGD adapter/CI/tests are coded but **NOT EXECUTED** in this runtime. Never label the new benchmark formally private, and keep package/runtime gates explicit. Skills: `.agents/skills/aggregate-collusion-research/` and `.agents/skills/native-opacus-privacy/` mirrored under `.claude/skills/`. Keep model observations and original images out of Git. Run `scripts/verify_aggregate_collusion.py` before interpreting results.

## v0.7.0 agent handoff priority

Read `CURRENT_STATE.md`, `NEXT_SPRINT.md`, `docs/v0.7-pre-registered-protocol.md`, `docs/v0.7-research-report.md` and `docs/opendp-runtime.md`. **Twenty native PyTorch, not Flower, matched study runs** are complete. Run `python scripts/verify_v070_evidence.py` to check hashes, data disjointness, identical optimizer budgets and reported metrics. The new OpenDP adapter is **not executed natively**: its test-double transport and ledger tests cannot certify ε-DP. Do not claim native Docker, Flower, Opacus or SecAgg+ execution. Skills: `.agents/skills/balanced-heterogeneity/` and `.agents/skills/private-query-budget/`, mirrored under `.claude/skills/`. Keep public raw data and model observations out of Git. Preserve all 20 seeds/conditions and record negative outcomes.

## v0.6.0 handoff priority

Read `CURRENT_STATE.md` and `NEXT_SPRINT.md` first. Reproduce `docs/v0.6-experiment-protocol.md` against `docs/v0.6-research-report.md` before editing experiments. **Do not conflate** 8 vs 32 distinct-record changing-minibatch inversion, independently seeded membership models vs query seeds, or imbalanced Dirichlet vs equal-sized severe label skew. Added `.agents/skills/changing-minibatch-update/SKILL.md` and `.agents/skills/model-output-research/SKILL.md` (mirrored under `.claude/skills/`). The Flower parity *checker* has unit fixtures, but a native Flower run remains unverified. Use `scripts/verify_v060_evidence.py` on the full evidence archive.

# FedDisclosure Lab agent instructions (Codex and Claude)

## v0.9.0 research handoff

Read `docs/v0.9-results.md` for the three-seed joint aggregate oracle, paired confidence intervals, and offline evidence explorer. Use `scripts/verify_joint_aggregate.py` and `python -m dashboard.build_dashboard` for new verification. The attack oracle receives site-grouped labels of ALL contributors; do not claim anonymous site identity or SecAgg+. Native privacy libraries remain unexecuted.

## Mandatory first read

Read, in order: `PROJECT_CONTEXT.md`, `CURRENT_STATE.md`, `NEXT_SPRINT.md`, `docs/threat-models.md`, `docs/methodology.md`, `docs/decisions.md`, `docs/research-questions.md`, `docs/project-spec.md`, `docs/execution-loops.md`.
Do not assume an unimplemented directory implies functionality. Check evidence in `reports/results/` and run tests.

## Goal

Build a defensible, reproducible educational and applied-research benchmark of privacy leakage under clearly separated federated analytics and ML attacker observations. Prioritize scientific correctness over features or interface polish.

## Non-negotiable safety and methodological rules

- Use only synthetic or explicitly approved public data. No real medical, confidential, or obtained-without-consent data.
- Never send raw institution rows to gateway, attack code or other institutions.
- Attack implementations receive only their threat-model-approved observations. Ground truth belongs only to evaluator code.
- Do not call clipping, suppression, DP, secure aggregation and encryption equivalent defenses.
- Make formal epsilon/delta claims only with valid accountant, adjacency definition, sensitivity and composition assumptions. No invented privacy scores.
- Restrict public query definitions to typed allowlists. No arbitrary SQL execution endpoints.
- Never reimplement cryptographic secure aggregation. Use established libraries.
- Do not claim a result without an executed command, observed artifact and documented limitations. Record failures.
- Protect .env, secrets, raw databases and research artifacts from accidental commits.
- Any added training or privacy method requires a baseline, a matched scenario, and a regression test.

## Working loop

1. Locate the next open acceptance criterion in `NEXT_SPRINT.md`.
2. Write the threat model, hypothesis, config and metric before the attack/defense code.
3. Implement narrowly, with tests for attacker/evaluator separation and fail-closed behavior.
4. Test: `pytest -q` then `python scripts/run_local_stack.py --once` for networking changes.
5. For Docker-dependent changes additionally run `docker compose config -q`, `docker compose up --build -d --wait`, and a host benchmark on a Docker-enabled machine.
6. Write machine-readable results under `reports/results/` and a short evidence summary under `reports/`.
7. Update `CURRENT_STATE.md`, `NEXT_SPRINT.md`, and `docs/decision-log.md`; commit a coherent unit. Do not silently broaden the scope.

## Skills

For threat-model or disclosure work use `.agents/skills/threat-model-review/SKILL.md` (Codex) or `.claude/skills/threat-model-review/SKILL.md` (Claude). For running or changing benchmarks read `experiment-reproduction/SKILL.md`. For claims involving defenses read `privacy-claims/SKILL.md`.

## Native Flower Docker handoff (v0.12)

Read `docs/v0.12-docker-flower-certification.md` and `.agents/skills/native-flower-docker/SKILL.md` before touching deployment. The v0.12 Compose topology and staging are tested statically, but native Docker execution is **not** verified here. Explicitly select `fdl-docker` rather than local simulation; never claim success from a missing Docker/Flower environment.

## Current verified evidence

- Three independent HTTP site processes, one HTTP gateway, and private per-process SQLite files were run via `scripts/run_local_stack.py --once`.
- Synthetic naive query responses: 110 and 107; inferred restricted subgroup: 3.
- History-aware second query: denied; no inferred value.
- Sprint 2: executed separate-process PyTorch 3-client FedAvg reference on public 8x8 digits (IID and label-skew non-IID). Evaluator runs in a separate process; IID repeat metrics matched exactly. This is **not** a Flower run.
- Docker and Flower native runtime gates are still **NOT VERIFIED** because Docker Engine and the Flower package were unavailable. Do not mark them complete from generated scripts or static code.

## Next priority

Execute the pending Sprint 1 Docker gate and Sprint 2 native Flower ServerApp/ClientApp runtime gate; extend experiments only after collecting observed evidence. See `NEXT_SPRINT.md` and `docs/fl-experiment-protocol.md`.

## v0.4 executed disclosure evidence

- Five matched exact/noisy public-digit gradient-inversion seeds and batch-8 known-label case, scored separately by evaluator. See `docs/disclosure-protocol.md` and `.agents/skills/disclosure-benchmark/SKILL.md`.
- Five query-seed output-only membership experiments on one frozen public-digit model (AUC near 0.5). Do not describe these as five training repeats.
- Five matched IID/non-IID training seeds on three separate PyTorch worker processes. Still NOT a native Flower run.
- Schema-v2 coordinator manifest excludes label histograms and rejects v1 input. Guard against oracle leakage if evolving the data pipeline.
- Continue preserving negative outcomes and unmet Docker/Flower gates. `NEXT_SPRINT.md` is current.

- v0.4.1 also executes local SGD **model-delta**, not exact-gradient, inversion across 1/4/16 steps. See `docs/disclosure-protocol.md`; multi-example Flower update inversion still pending.

## v0.5 handoff (newly executed)

- Distinct-record client update inversion now supports batches 1/8/32 and multiple local SGD steps. `scripts/run_multirecord_update_inversion.py` produces five actual attack/evaluator results. Attackers do not receive images; batches >1 explicitly assume known labels. This still reuses a fixed batch and does NOT demonstrate Flower training or secure-aggregation leakage.
- FedProx uses a `mu/2 * ||w-w_global||^2` penalty in the common training task; mu=0 numerically equals FedAvg. Three matched severe label-skew paired-seed runs on public 8x8 digits were executed by `scripts/run_fedprox_ablation.py`, mean FedAvg 55.19% vs FedProx 49.07% and all worst-client accuracies zero. Native Flower code is wired but not verified. See `docs/fedprox-ablation.md`.
- New evidence and plots under `reports/results/sprint2/ablations/paired/`, `reports/results/sprint3/multirecord_update/`, and `reports/figures/`. Do not claim DP or SecAgg+, or infer generalizability from one attack seed/three training seeds.
- Runtime here lacks Docker and Flower. `pip install -e . --no-build-isolation --no-deps` works when requirements already exist. Execute each marked-slow test individually if the combined suite hits a watchdog. Update `CURRENT_STATE.md` and `NEXT_SPRINT.md` after every next milestone.
