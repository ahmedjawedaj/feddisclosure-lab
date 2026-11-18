---
name: experiment-reproduction
description: Run, audit, reproduce, troubleshoot or extend the FedDisclosure benchmarks with configs, actual observed metrics, deterministic seeds, tests and explicit failure recording.
---

# Benchmark reproduction

Read `CURRENT_STATE.md`, `docs/methodology.md` and config under `configs/`.

1. Freeze config and record experiment ID, seed, environment/software versions, observation class and metric definitions.
2. Check all used data is synthetic or approved public data. Assert independent site partitions.
3. Test before running. Execute `pytest -q`.
4. For federated analytics, execute `python scripts/run_local_stack.py --once` and inspect JSON under `reports/results/`.
5. For Docker validation, **actually** run Compose commands on a host with Docker. Never infer success from YAML existence.
6. Compare to matched naive/reference baseline. Record accuracy, disclosure outcome, rejection/utility and elapsed time.
7. Write a short research-results summary and limitations. Record failed attempts as distinct artifact rows.
8. Update `CURRENT_STATE.md`, `NEXT_SPRINT.md` with exact passing/failing checks and command paths.

Do not invent benchmarks, claim fresh data is validated without a run, or discard negative results.
