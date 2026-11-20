---
name: native-v1-certification
description: Execute and verify the actual Docker, Flower, OpenDP, Opacus, and upstream SecAgg+ acceptance gates, retaining evidence and failed runs.
---
# Native v1 certification

Read `docs/v0.11-runtime-and-retry.md`, `NEXT_SPRINT.md`, and `.github/workflows/native-v1-certification.yml` before claiming progress.

1. Use a network-enabled Python 3.12 Ubuntu host with Docker Engine/Compose v2. Run `python scripts/native_preflight.py --strict`.
2. Generate approved public-data partitions from a clean source checkout: `python scripts/prepare_fl_data.py --output data/training/digits-iid --clients 3 --seed 20261118 --limit 600 --partition iid`.
3. Run native Docker isolation and protected-history verification, native Flower training plus exact PyTorch checkpoint parity, native OpenDP and Opacus, then upstream Flower SecAgg+. Preserve command logs, configs, privacy accounting, hashes, errors and GitHub Actions run IDs.
4. Run `python scripts/verify_native_release.py`. A missing or inconsistent artifact MUST leave the gate red. A green consistency report is not a cryptographic audit or a privacy guarantee for arbitrary deployments.
5. Never replace native test results with mocks, arithmetic aggregation or a prepared CI YAML. Do not modify acceptance criteria to make a failed run appear successful.
6. Update `CURRENT_STATE.md`, `NEXT_SPRINT.md`, `docs/decisions.md` and release evidence only after observed execution.
