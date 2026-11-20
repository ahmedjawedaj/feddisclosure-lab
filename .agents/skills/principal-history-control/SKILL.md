---
name: principal-history-control
description: Investigate caller-controlled experiment IDs, persistent analytics release controls, and disclosure benchmarks without mislabeling heuristic suppression as differential privacy.
---

# Principal-wide query history and threat-model testing

1. Read `docs/v0.10-results.md`, `src/feddisclosure/analytics/policy.py` and `src/feddisclosure/analytics/enforcement.py`.
2. Distinguish the **intentionally vulnerable** `history` reference from `principal_history`. The former scopes to an attacker-controlled experiment ID; the latter covers a single shared research principal across all experiment IDs and pins the site release policy.
3. A release must pass the local allowlist, cell threshold, history check and durable site-mode lock before crossing the gateway. Keep site rows and membership IDs in their local SQLite database.
4. Reproduce by running `PYTHONPATH=src:. python scripts/run_history_scope_benchmark.py`, then `python scripts/verify_history_matrix.py`. When Docker is available run `bash scripts/verify_compose_history.sh` on **fresh volumes**; its unexecuted CI workflow is not proof of Docker.
5. Document query budget, observer privileges, policy downgrade paths and false denials under a predeclared legitimate workload. Do not interpret the 15.6% residual as a general data-leakage probability.
6. Do not call this differential privacy or secure aggregation. Only the actual established OpenDP mechanism and accountant can justify formal count privacy claims.
