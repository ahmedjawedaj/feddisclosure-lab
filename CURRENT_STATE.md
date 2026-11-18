# CURRENT STATE — 2026-11-18

**Project:** FedDisclosure Lab 2.0. **Version:** 0.3.0 — Sprint 1 hardening + partial Sprint 2.

## VERIFIED HERE — SPRINT 0/1

- Three separate institution FastAPI processes and one gateway ran over HTTP with different private SQLite files (`scripts/run_local_stack.py --once`). 110 total and 107 negatives disclosed 3 restricted synthetic cases under the naive policy; the limited history rule refused the second query. Saved `reports/results/analytics_local_sprint2_validation.json`.
- New concurrency test: 20 parallel site releases completed atomically with SQLite transactions. New `query_audit` stores *metadata only*, never raw member identifiers; internal `release_history` still requires simulated membership IDs for heuristic comparisons.
- New unreachable-site test asserts gateway fails closed (503, no partial count). Other gateway/role boundary tests continue passing.
- Docker verification script (`scripts/verify_compose.sh`) and CI job provide a native path to inspect distinct volumes, private network, host loopback and a real benchmark. **Docker is not installed here: gate remains UNVERIFIED.**

## VERIFIED HERE — SPRINT 2 REFERENCE ENGINE

- PyTorch 2.10 CPU + NumPy/scikit-learn public 8x8 digit dataset split into three institution-owned train and held-out validation files, plus separate evaluator-only public test. Deterministic IID and label-skew partitions, on-disk hashes, manifest and duplicate overwrite prevention.
- Real **independent OS process** client workers trained locally; three workers run concurrently per round. Coordinator aggregates individual model arrays using a sample-count-weighted FedAvg reference and never imports raw data arrays. A separate evaluator process scores the public test and computes privileged pooled/local-only references. The final global model is separately evaluated on each client's private held-out validation through worker processes.
- **IID smoke (600 selected public examples, 3 clients, 2 rounds, seed 20261118):** public test accuracy **0.2083333333**, final global worst-client validation accuracy **0.1875**, evaluator-only pooled reference accuracy **0.3666666667**, estimated 57,840 raw model-tensor bytes exchanged each round.
- **Severe label-skew non-IID smoke:** public test accuracy **0.2**, final global worst-client validation accuracy **0.0**, evaluator-only pooled reference accuracy **0.25**. Zero worst-client accuracy on this tiny/short split is a reported negative result, not a deployed failure claim.
- Independent IID rerun exactly matched each round's public test metrics, final global worst-client metric, reference baseline metrics and final model checkpoint SHA-256. Exact machine-readable files in `reports/results/sprint2/` and figure `reports/figures/sprint2_training_reference.png`.
- FedAvg tensor arithmetic cross-checked against independent NumPy weighted averaging. **This is not a native Flower aggregation test.**
- Test evidence: `pytest -q -m 'not slow'` **24 passed, 1 skipped, 2 deselected**. Two heavy integration tests each passed independently, hence **26 passed / 1 Flower import skipped** overall. Full-suite pytest displayed these results but its outer execution timed out after the summary; explicit split-suite commands exited successfully. Real analytics HTTP harness passed after the changes.

## IMPLEMENTED BUT NOT EXECUTED / VERIFIED

- Native Flower 1.27 `ServerApp` + `ClientApp` Message API (`src/feddisclosure/training/flower_server.py` and `flower_client.py`) using PyTorch task, `FedAvg`, `ArrayRecord`, `MetricRecord`. Script `scripts/run_flower_experiment.py` launches actual `flwr run` and captures checkpoint, metrics and logs **on a dependency-enabled host**.
- GitHub Actions `flower-smoke.yml` prepares an official Flower environment and runs the real three-client simulation; requires push to a repository with actions enabled. There is no observed Actions result or CI attestation yet.
- MNIST download/partition path exists but was not executed, due to unavailable network and Flower dependency in this runtime.
- `docker compose config`, container execution, network/volume inspection and remote independent Flower SuperNodes all remain unverified. There is no Docker executable/daemon available here.

## FURTHER LIMITATIONS

- Offline PyTorch clients are separate OS processes but not file-system-isolated containers. The evaluator process can access public original data by design; attacker roles do not. This testbed is **not** a production security boundary.
- Public-digit hold-out is only a smoke test, 1 seed/2 rounds, not an effect size, significance or convergence study. Numerical pooled/local-only references use different optimization schedules from federated training.
- Per-site membership history updates are not atomically rolled back across a failed federated query. Gateway operators have stronger visibility than external analytics users. Rejection may leak information.
- No formal DP, secure aggregation, gradient reconstruction, model-output membership inference, dashboard, digital twin or publication claim.

## NEXT VERIFIED COMMANDS

```bash
python -m pip install -e '.[test]'  # on a network-enabled host; core deps
pytest -q -m 'not slow'
pytest -q tests/training/test_runner_integration.py  # when torch+sklearn available
python scripts/run_local_stack.py --once
python scripts/run_torch_reference.py --config configs/training/fedavg_digits_iid.yaml --data-dir data/training/digits-iid --output reports/results/sprint2/local_trial.json --references
# Flower + Docker acceptance gates need matching native runtimes:
# python -m pip install -e '.[ml,test]'
# python scripts/run_flower_experiment.py
# bash scripts/verify_compose.sh
```

See `NEXT_SPRINT.md` and `docs/fl-experiment-protocol.md`. No GitHub remote is configured. Source and evidence persist in the repository and its next ZIP handoff.
