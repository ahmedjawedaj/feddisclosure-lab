# FedDisclosure Lab v0.10.0 release acceptance ledger

Date: November 20, 2026. This remains an **interim reproducible research release**, not a formal DP claim, production deployment or paper-ready contribution.

## Newly completed and verified

| Milestone | Actual evidence | Status |
|---|---|---|
| Caller-controlled experiment-ID reset and persistent history control | 3 independent HTTP site services + gateway; 110 − 107 = 3 before; second query blocked after; policy switch denied | Executed |
| Matched query-release experiment | 90 populations × four modes × three queries = 1,080 SQLite release-policy decisions; hash and independent replay | Executed |
| Paired-query output-only membership baseline | 10 independently trained model checkpoints, 30 attack evaluations, 5 seeds and exact query cohort matching; verifier re-scores all | Executed |
| Full fast test suite | 100 passed, 2 skipped (Flower/OpenDP), 3 slow deselected | Passed |
| Slow independent-worker training | IID and label-skew tests passed separately | Passed |
| Prior training, model-update and collusion evidence | v0.7 20 checkpoints, v0.8 15 scores, v0.9 joint reconstruction all re-verified | Passed |
| Live original analytics baseline | HTTP 110 − 107 = 3 and history rejection reverified | Passed |
| Offline evidence explorer | Updated with new query history and 5-seed membership tabs; regenerated | Passed |
| Interim faculty briefing | 12-slide PowerPoint created and reopened, generated from published research artifacts | Created and opened |
| Native Flower / Docker / OpenDP / Opacus / Flower SecAgg+ | No native execution; Docker missing and pip DNS unavailable | **BLOCKED** |
| Full MNIST 28×28, formal defenses, publication novelty | Not completed | **OPEN** |

## What the progress estimate means

The project is approximately **75–80% implemented as a research engineering portfolio**, judged by code, recorded public/synthetic-data experiments, basic reporting and testing. This is an **informal estimate**, not a measured acceptance score. The fraction of the original specification **natively verified** is lower, because genuine Flower, OpenDP, Opacus, SecAgg+, Docker isolation and full MNIST remain unexecuted or unavailable. We do not assert that all Sprint 4 or full v1.0 acceptance requirements have passed.

## Known scientific limits

- The principal-wide history rule covers one shared research principal and one typed two-query differencing pattern; its observed residual inference rate is not a real-world attack probability and the control is not DP.
- IID/skew membership queries match on exact queried records but not on complete underlying training-membership sets, so paired AUC differences are descriptive, not causal.
- Membership attack heuristics are uncalibrated and the 1% FPR threshold is poorly resolved with only 60 nonmembers/model.
- Idealized arithmetic means do not implement or validate Flower SecAgg+.
- Real Python workers do not establish isolated Docker network or host-OS user boundaries.

## Reproduce the release

```bash
PYTHONPATH=src:. pytest -q -m 'not slow'
PYTHONPATH=src:. python scripts/run_history_scope_benchmark.py --output reports/results/sprint1/repeat_http.json
python scripts/verify_history_matrix.py
python scripts/verify_balanced_membership.py  # requires full evidence archive
python scripts/verify_v070_evidence.py
python scripts/verify_aggregate_collusion.py
python scripts/verify_joint_aggregate.py
python -m dashboard.build_dashboard
python scripts/plot_v010.py
python scripts/build_faculty_briefing.py  # requires optional presentation extra
```

Next: native runtime/defense acceptance, real Flower SecAgg+, full MNIST and statistical external replication, research novelty selection with faculty.
