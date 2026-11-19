# FedDisclosure Lab agent instructions (Codex and Claude)

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
