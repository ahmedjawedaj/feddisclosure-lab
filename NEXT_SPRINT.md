# NEXT SPRINT — Complete the native runtime gates, then extend Sprint 2

**Current work:** Sprint 1 hardened and the standalone three-client PyTorch FedAvg reference executed. Sprint 1 Docker and Sprint 2 *native Flower* gates remain open.

## P0 — Native Compose acceptance (NOT VERIFIED)

1. In a Docker Engine/Compose v2 environment, run `bash scripts/verify_compose.sh`.
2. Verify `docker compose config -q`, four healthy containers, distinct SQLite volumes, internal-only client network, localhost-only published gateway; run both analytics policies from host. Save `docker_isolation.json`, benchmark JSON and logs.
3. If Compose fails, fix in the tested environment and rerun. Do not mark Sprint 1 closed from static YAML checks.

## P0 — Native Flower three-client acceptance (NOT VERIFIED)

1. Install supported Python 3.12 and `pip install -e '.[ml,test]'` in a network-enabled environment.
2. Run `python scripts/run_flower_experiment.py` after checking `flwr --version` and Flower 1.27 compatibility. Save real Flower logs, `flower_run.json`, `flower_summary.json` and checkpoint.
3. Assert three participating ClientApps, both rounds, correct client partition paths, aggregated validation metrics, no raw record transfer. Compare a one-round aggregated model to the transparent weighted reference under **matching** initialization, sample weights and client shuffle schedule.
4. Make the Flower CI gate run and collect artifact; diagnose and fix actual runtime issues instead of reporting its presence as success.
5. Evaluate separate SuperNodes mounted to per-site data volumes. A Flower local simulation is not evidence of OS-level client dataset isolation.

## P1 — Improve experiment quality

1. Run MNIST from an approved public download, save checksum/license/source and deterministic IID/label-skew splits. Never add raw datasets to Git.
2. Run 5 seeds and more meaningful training rounds; record distributions and worst-client *global model* validation, not only local-update validation. Add a public-digits 1/3/5-client ablation.
3. Keep matched local-only, pooled public reference and FedAvg scenarios with clear compute and optimizer differences. Later introduce FedProx with explicit reference paper.
4. Record numerical model communication, wall-time and actual Flower transport bytes separately. Add a second OS-isolation test and intentional worker failure test.
5. Add per-client Flower validation logging without exposing raw examples, and ensure negative cases survive regression.

## Later, not this sprint

- Sprint 3: controlled gradient inversion and output-only membership inference; separate exact gradient observations from multi-step model updates.
- Sprint 4: OpenDP/Opacus/Flower SecAgg+ with proper accounting and assumptions.
- No dashboard-first work, synthetic real-world privacy claims, or unvalidated digital twin.

## Exit criteria

Sprint 1 Docker gate: **pending actual Docker execution**. Sprint 2 native Flower gate: **pending actual 3-client Flower run and one-round numerical cross-check**. Executed PyTorch reference results are valuable but cannot substitute for these tests.

Whenever a gate succeeds, write machine-readable evidence, update `CURRENT_STATE.md`, `docs/decision-log.md` and README, then commit the tested change. Preserve failed logs and configs.
