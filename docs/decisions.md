## 2026-11-20 — v0.10: experiment-ID reset and paired-query membership study

- Old `history` release decisions depended on an attacker-controlled experiment ID, so it remains in the code only as a deliberately vulnerable reference. New `principal_history` checks all site-wide prior releases, plus immutable enforced mode and fresh-volume upgrade checks. This is a restricted heuristic for *one research principal*, not a formal DP mechanism or complete protection against adaptive queries.
- We observed different **training membership** cohorts across IID and severe non-IID FedAvg runs, even with equal sizes. We stopped an invalid matched-training-cohort claim, instead evaluated both models on exactly the same shared member records and nonmembers, and explicitly downgraded the interpretation to descriptive. Every observed weak attack is retained.
- Native runtime proof is a hard acceptance gate. Static Compose overlay/CI definitions or a local subprocess stack cannot be counted as native Docker/Flower/OpenDP/Opacus/SecAgg+ execution.

## 2026-11-19 — Aggregate observation versus collusion (v0.8)

Compare attacker observer privileges, not "privacy scores." An arithmetic mean is an *idealized visibility model*, not a secure-aggregation protocol. Grant target labels consistently, preserve weakness of aggregate-only target-proxy attack, and explicitly record all-but-one collusion. Add native Opacus DP-SGD only with a real accountant and no manual noise stand-ins; defer repeated FL accounting and formal claim until executed.

# Architecture decisions

- **ADR-001: Single machine first.** Docker Compose for three isolated institutional containers; local process harness used for testing where Docker daemon unavailable.
- **ADR-002: Typed query allowlist.** Only `all_records` and `negative_records` available in first benchmark; no remote SQL endpoint.
- **ADR-003: Institution release control.** Each site owns its threshold and private query history; gateway rejects whole federated query if one site rejects.
- **ADR-004: Separate observer/evaluator.** Attack sees aggregate observations, evaluator alone receives synthetic truth.
- **ADR-005: No invented privacy claims.** Membership-set history rule is a heuristic, never differential privacy; separate future OpenDP/Opacus/SecAgg+ evaluations.
- **ADR-006: Framework selection.** Flower current ServerApp/ClientApp for training (not older `start_simulation`). Official APIs before integration. Preserve a reference FedAvg implementation only if tests require it.
- **ADR-007: No active digital twin in v1.** Only consider validated simulator in future based on research findings and faculty guidance.
- **ADR-008: Public dataset reproducibility.** Store only partition metadata/hash in tracked results; generate immutable private NPZ partitions in ignored per-site directories. Keep public evaluator ground truth in separate role.
- **ADR-009: Dual training engines.** A transparent, executed independent-process PyTorch reference establishes numerical baseline; a genuine Flower ServerApp/ClientApp uses the officially documented Message API. Never relabel the former as native Flower.
- **ADR-010: Honest Docker gate.** Compose inspection and full integration benchmark are required for a verified Docker claim; running subprocesses or parsing YAML does not clear the gate.

- **ADR-015: Matched heterogeneity and honest DP runtime.** Enforce equal client volumes and per-round optimizer budgets for Dirichlet comparisons; label distributions are evaluator-only. A fake-tested transactional budget is not an Opacus/OpenDP measurement; formal claims require native OpenDP privacy-map and noise execution in a DP-only deployment.

## 2026-11-19 — v0.9 research execution and evidence explorer

- Added a high-privilege joint aggregate-gradient reconstruction with all 24 labels grouped by institution. Retain the grouping as an explicit additional attacker capability. This is neither Flower SecAgg+ nor normal aggregate-only privacy.
- Kept original scored artifacts immutable. Historical saved contract descriptions need a v0.9 correction: the labels' order exposes site grouping, even though raw images are never in the attacker files.
- Added post hoc paired exploratory confidence intervals and sign-flip p-values for an existing five-seed equal-work study. Both intervals include zero; no confirmatory significance claim.
- Added dependency-free offline evidence explorer with whitelisted scalar JSON; do not embed raw data or model updates. Streamlit remains an unexecuted optional interface.
- Native Docker, Flower, OpenDP, Opacus, SecAgg+ continue to block the full project gate; adding source and workflows does not clear them.

## 2026-11-20 — v0.11: idempotent release retries and fail-closed native gate

- An explicit `request_id` is only a replay key bound to a principal, query and epsilon. A replay returns the *previous* randomized answer without spending epsilon; a new key is a new release. Use a transactional cache, not a new random draw on network retry. Reject request-ID mismatch at the gateway. The current evaluation uses a fake nonprivate measurement only and cannot support epsilon-DP claims.
- Maintain the pinned main Flower ServerApp/ClientApp parity benchmark. Add a separate Flower *legacy workflow* research app for upstream SecAgg+ rather than implementing cryptography. Its 3-client threshold and deployment assumptions must be tested on the native runtime; simulation is not an OS isolation proof.
- Add a native CI workflow with a final multi-component evidence consistency gate. It must fail when any mandatory artifact is missing. A local preflight or GitHub workflow file is not itself executed evidence.

## 2026-11-20 — v0.12: true native Flower Docker deployment

- Use Flower 1.27's Deployment Runtime explicitly, with separate SuperLink,
  SuperNodes and app executors. Do not count `flwr run` using the default
  `local-simulation` federation as independent-container training.
- Build a strict source-only image. Each ClientApp executor mounts exactly one
  institution's SHA-verified public NPZ partition read-only; coordinator and
  CLI mount no client or evaluator raw data. Per-client attestation is a
  reviewable operational receipt, not a cryptographic proof.
- Accept the Docker/Flower milestone only with live Docker inspect, three
  client participation receipts, a ServerApp checkpoint and strict checkpoint
  parity against the independent-worker reference.
- This controlled benchmark uses insecure internal Flower transport and only
  approved public digit data. TLS, real private data, and production security
  are out of scope. Current tool runtime lacks Docker/Flower: gate remains open.
