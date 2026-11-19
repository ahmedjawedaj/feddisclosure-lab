# Claude handoff — v0.9.0

Start with `AGENTS.md`, `PROJECT_CONTEXT.md`, `CURRENT_STATE.md`, `NEXT_SPRINT.md`, and `docs/v0.9-results.md`. Preserve completed v0.7/v0.8 benchmarks. For v0.9, run `python scripts/verify_joint_aggregate.py`, `python scripts/analyze_balanced_study.py`, and `python -m dashboard.build_dashboard`. Native Docker, Flower, OpenDP, Opacus, and SecAgg+ remain unverified. Labels for all contributors are granted by the joint-attack oracle and grouped by institution; this is NOT a standard aggregate-only observer.

---

**v0.8 handoff:** Read `AGENTS.md`, `CURRENT_STATE.md`, `NEXT_SPRINT.md`, `docs/v0.8-research-protocol.md`, and `docs/v0.8-results.md` first. Preserve separate observer roles and native DP execution gates.

## v0.7.0 Claude handoff

First read `CURRENT_STATE.md`, `NEXT_SPRINT.md`, `docs/v0.7-research-report.md`, and `docs/opendp-runtime.md`. Reproduce `scripts/verify_v070_evidence.py` and all fast tests; follow `.claude/skills/balanced-heterogeneity/SKILL.md` and `.claude/skills/private-query-budget/SKILL.md` when editing new experiments. PyTorch reference is not Flower. Fake DP transport tests are not native OpenDP or formal privacy evidence; do not claim epsilon guarantees before real OpenDP map and sampling pass. Update research decision log, evidence hashes and state after every loop.

# Claude project entry point

Follow all instructions in `AGENTS.md`. Read `CURRENT_STATE.md` and `NEXT_SPRINT.md` before editing. Repo-specific skills live in `.claude/skills/`. Never replace verified benchmark evidence with a planned result. Use small increments, deterministic configurations, and update the persistent handoff at the end of each tested sprint.

Claude can continue in the same working copy or in a clone of this repository. Local agent memory/chat history is **not** the canonical project source. Repository files and benchmark artifacts are.
