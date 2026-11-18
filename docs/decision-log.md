# Engineering decision and evidence log

## 2026-11-18 — Sprint 0 and early Sprint 1

Implemented typed count APIs, per-institution persistent SQLite data, minimum-cell policy, limited membership-based query history, three-institution gateway, attacker/evaluator separation, YAML config, local process integration harness, pytest tests, CI and agent handoff. Verified non-containerized four-service run and saved machine-readable benchmark. Docker Engine unavailable in this execution environment; Compose is prepared but unverified here.

## 2026-11-18 — Verified reproducibility closure

Expanded tests to 18, including gateway fail-closed behavior, permission checks, exact expected artifact contents and agreement across two fully independent four-process runs. Installed the package in editable mode and verified CLI. Created `reports/figures/sprint1_analytics.png` from measured JSON, and added machine-readable tracked baseline artifacts. Docker Compose remains unverified because Docker Engine is unavailable here.
