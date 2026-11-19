# FedDisclosure Lab v0.9.0 — verified release note

**Scope:** Executed joint three-client aggregate inversion (all-contributor-label oracle), exploratory paired statistical analysis, offline scalar-evidence explorer, updated Claude/Codex handoff, and CI test-splitting improvements.

**Evidence:** `reports/results/sprint4/joint_aggregate_oracle/seed241..243`, `reports/results/v090_verification`, `dashboard/dist`. The researcher evidence bundle contains original public digit pixels in evaluator-only directories; these are excluded from Git but included in ZIP. Do not treat folder separation as a security boundary.

**New attack observations:** Three independent seeds (241, 242, 243), 160 iterations each. Joint optimization of 24 public digit examples from a three-client arithmetic mean. Pooled label-matched MSE: 0.059282 / 0.055844 / 0.056824. All 24 class labels and their institution grouping were given to the attacker. No per-client model updates or raw images were provided. Evaluator-only target matching is optimistic and not target site attribution.

**Paired training statistics:** Post hoc IID versus alpha=0.1 difference +2.33 pp global and −11.67 pp worst-client, with 95% intervals crossing zero. No confirmatory statistical or causal conclusion. See `docs/v0.9-results.md`.

**Executed verification:** `pytest -q -m 'not slow'`: 88 passed, 2 skipped, 3 deselected. `verify_v070_evidence.py`: passed. `verify_aggregate_collusion.py`: passed. `verify_joint_aggregate.py`: passed. `run_local_stack.py --once`: passed. `python -m dashboard.build_dashboard`: generated offline artifact and passed dashboard tests. Combined slow training suite timed out; no new slow test completion claimed.

**Unverified:** Docker Engine and Compose, Flower 3-client runtime/parity, OpenDP native noise, Opacus DP-SGD, Flower SecAgg+, actual GitHub Actions execution, and Streamlit runtime. No formal DP epsilon/delta or cryptographic privacy claim is made.

**Next research gate:** Execute real protocol and privacy defense experiments on a network-enabled Python 3.12/Docker host, then perform matched defended/unprotected model and analytics comparisons with saved accountant and utility evidence.
