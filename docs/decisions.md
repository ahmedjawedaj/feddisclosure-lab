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
