# CURRENT STATE — 2026-11-18

**Project:** FedDisclosure Lab 2.0. **Version:** 0.2.0 initial implementation.

## VERIFIED IN THIS ENVIRONMENT

- Built Python package, tests, synthetic deterministic datasets, YAML experiment config, agent instructions/skills, CI, and Docker Compose definitions.
- Three independent institutional HTTP processes, each writing a separate private SQLite DB, and a fourth gateway HTTP process ran successfully using the local-process harness.
- Naive release mode: `all_records=110`, `negative_records=107`, attacker inferred a restricted group of `3` exactly. The scorer marked `successful_small_group_disclosure=true`.
- History mode: first count released, second rejected. Attacker received insufficient observations; scorer marked disclosure false and legitimate-query rejection 0.5 in the two-request sequence.
- `pytest -q` passed 18 tests after gateway failure-behavior and two-run reproducibility checks.
- Artifacts: `reports/results/sprint1_analytics_verified.json`, `reports/results/sprint1_analytics_repeat_verified.json`, and `reports/figures/sprint1_analytics.png` (explicitly retained as canonical baseline evidence).

## UNVERIFIED / NOT CLAIMED

- Docker/Compose deployment (Docker binary/daemon is not available in this execution environment).
- Flower/PyTorch federated training, MNIST downloads, reconstruction, membership inference.
- OpenDP, Opacus, Flower SecAgg+, any formal privacy budget/accountant.
- Streamlit dashboard, simulation/digital twin, a publication or any novel research contribution.

## KNOWN LIMITATIONS

- Membership-based history rule is a heuristic that can miss adaptive query combinations, and query rejections may leak information.
- Local process harness has separate paths/processes, not Docker or OS-enforced network/filesystem isolation.
- Research API key is not production-grade tenant authorization; never expose gateway publicly.
- Two-query demonstration is a proof of possibility and mitigation for **this** attack, not a general privacy effectiveness estimate.

## HANDOFF COMMANDS

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/run_local_stack.py --once
```

Read `NEXT_SPRINT.md`, `docs/threat-models.md`, `docs/methodology.md` before further implementation.

## Historical starter availability

Prior starter benchmark behavior was recovered from the documented research plan. The original starter ZIP/source was not available for byte-level migration in this environment; this is a clean, tested reconstruction rather than a code-for-code carryover.
