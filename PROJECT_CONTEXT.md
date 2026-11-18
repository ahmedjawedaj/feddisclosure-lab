# FEDDISCLOSURE LAB 2.0 — PROJECT CONTEXT

**Scope:** Reproducible privacy leakage and defense benchmark for federated analytics and federated learning. Two core tracks: statistical queries and distributed training. Optional later simulation/digital twin. Independent portfolio/research project, not currently a thesis or publishable novelty claim.

**Why:** Keeping raw data local does not by itself prevent disclosure via aggregated answers, gradients/updates or model outputs. Different channels imply different attacker privileges and defenses.

**Audience:** Researchers, academic supervisors, ML engineers and privacy/security reviewers.

**Working environment:** First phase: Python 3.11–3.13 and FastAPI multi-process local validation; Docker Compose created for a single future Docker host. Later: Flower + PyTorch, optional GPU/Colab experimentation. Cloud deployment **not** verified.

**Nonnegotiable rules:** No real confidential data. No fabricated benchmark results. No unsupported formal DP guarantees. No evaluator-truth leakage to attack code. No custom secure-aggregation crypto.

**What is done:** Sprint 0 repository/research infrastructure, Sprint 1 three-site analytics service and reproduced 110–107=3 leakage with history-aware rejection, results and 18 automated tests.

**Where to resume:** `CURRENT_STATE.md` and `NEXT_SPRINT.md` are canonical. For how to work, read `AGENTS.md`, `docs/execution-loops.md` and `.agents/skills/` or `.claude/skills/`.

**Official research specification:** The detailed FedDisclosure Lab 2.0 proposal, November 2026, supplied by the project owner. The exact text is summarized in `docs/roadmap.md`, `docs/research-questions.md`, `docs/architecture.md`, `docs/methodology.md`, and `docs/threat-models.md`. The canonical requirement-by-requirement transcription is `docs/project-spec.md`. See `configs/` for implementation-ready definitions; do not treat proposed work as implemented.
