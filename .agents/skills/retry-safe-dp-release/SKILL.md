---
name: retry-safe-dp-release
description: Maintain and test fail-closed, idempotent count-release retries without confusing test doubles with differential privacy.
---
# Retry-safe DP release research

Read `docs/v0.11-runtime-and-retry.md`, `src/feddisclosure/defenses/dp_release.py`, `services/client/app.py`, and `services/gateway/app.py` first.

- A `request_id` identifies one previous randomized release for one principal, query, and epsilon. A new ID is a new release and consumes fresh epsilon; `experiment_id` is NEVER a budget key.
- The SQLite budget increment and cache insertion occur inside one BEGIN IMMEDIATE transaction. On an exact retry, return the prior noisy count without redrawing or charging epsilon.
- Fail closed on request ID reused with a different query or epsilon. The gateway checks the echoed request ID from every site and must not expose partial answers on any error.
- Use `pytest -q tests/privacy/test_dp_ledger.py tests/privacy/test_dp_http.py tests/privacy/test_retry_benchmark_contract.py` and `python scripts/benchmark_retry_safety.py`.
- These tests inject a deterministic FAKE measurement; do not call the output DP, a privacy proof, or native OpenDP evidence.
- To make DP claims run `scripts/run_dp_analytics.py` with a native OpenDP installation, inspect the privacy map/adjacency and account for repeated and adaptive releases.
