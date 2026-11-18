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
