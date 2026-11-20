---
name: native-flower-docker
description: Execute and audit three independent Docker client containers using Flower's Deployment Runtime.
---

# Native Flower Docker skill

1. Read `docs/v0.12-docker-flower-certification.md`, `CURRENT_STATE.md`, `NEXT_SPRINT.md`.
2. Run `python scripts/verify_flower_topology.py` and the contract tests.
3. On a Docker host run `python scripts/run_flower_docker.py` using a fresh run directory.
4. Inspect observed `docker_topology.json`, `client_attestations.json`, Flower CLI logs, checkpoints, and `flower_run.json`.
5. Compare against independent-worker PyTorch with `scripts/verify_flower_parity.py --require-deployment`.
6. Treat missing Docker, failed native runtime, empty receipts or parity mismatches as FAILED/UNVERIFIED. Never substitute static YAML checks or Flower local simulation.
7. Update the handoff, record exact commands, CI run, versions and failures. Do not alter ground-truth/attacker boundaries.
