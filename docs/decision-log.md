# Engineering decision and evidence log

## 2026-11-18 — Sprint 0 and early Sprint 1

Implemented typed count APIs, per-institution persistent SQLite data, minimum-cell policy, limited membership-based query history, three-institution gateway, attacker/evaluator separation, YAML config, local process integration harness, pytest tests, CI and agent handoff. Verified non-containerized four-service run and saved machine-readable benchmark. Docker Engine unavailable in this execution environment; Compose is prepared but unverified here.

## 2026-11-18 — Verified reproducibility closure

Expanded tests to 18, including gateway fail-closed behavior, permission checks, exact expected artifact contents and agreement across two fully independent four-process runs. Installed the package in editable mode and verified CLI. Created `reports/figures/sprint1_analytics.png` from measured JSON, and added machine-readable tracked baseline artifacts. Docker Compose remains unverified because Docker Engine is unavailable here.

## 2026-11-18 — Sprint 1 guardrail hardening (commit after tests)

Added metadata-only per-site query audit (including denied releases), serializable SQLite transaction concurrency test with twenty parallel clients, gateway unreachable-site fail-closed regression, native Compose verification script for volume/network/port inspection. No Docker daemon installed in this environment, so Compose execution is still explicitly pending. Analytics HTTP integration rerun maintained 110/107/3 and history rejection.

## 2026-11-18 — Sprint 2 measured PyTorch reference and native Flower implementation (commit after tests)

Implemented versioned public digits IID and severe label-skew partition manifests, immutable site-owned NPZ files, PyTorch model/training and weighted FedAvg math, three concurrent independent client processes, evaluator-only public test and pooled/local-only references, model-update byte estimates, and final per-client validation of aggregated weights. Fixed a role boundary issue during review by moving public test scoring into a separate evaluator process and restricting coordinator to metadata/model arrays. Exact IID repeat results matched. Figure and JSON artifacts persisted under `reports/`. Native Flower 1.27 ServerApp/ClientApp and a CI integration runner are implemented against official Message API documentation, but **the Flower package could not be fetched in this network-restricted environment**. Neither native Flower nor Docker executed; runtime gates remain pending.
