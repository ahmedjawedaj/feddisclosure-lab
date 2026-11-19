**v0.6.0 handoff, 2026-11-19:** Completed three changing-minibatch local model-delta inversion experiments, 9 output-only membership evaluations across 3 independently trained model seeds, 8-round severe skew FedAvg/FedProx, and one 8-round moderate Dirichlet(alpha=0.5) FedAvg run. Last global accuracies: 63.33% severe FedAvg, 60.56% severe FedProx, 82.78% moderate Dirichlet FedAvg. Last worst-client: 6.25%, 0%, 79.59%. **Confounded partition study:** unequal site sample sizes and local SGD budgets. Native Docker/Flower, DP and SecAgg+ STILL NOT VERIFIED. See `docs/v0.6-experiment-protocol.md` and `docs/v0.6-research-report.md`.

# FEDDISCLOSURE LAB 2.0 — PROJECT CONTEXT

**Scope:** Reproducible privacy leakage and defense benchmark for federated analytics and federated learning. Two core tracks: statistical queries and distributed training. Optional later simulation/digital twin. Independent portfolio/research project, not currently a thesis or publishable novelty claim.

**Why:** Keeping raw data local does not by itself prevent disclosure via aggregated answers, gradients/updates or model outputs. Different channels imply different attacker privileges and defenses.

**Audience:** Researchers, academic supervisors, ML engineers and privacy/security reviewers.

**Working environment:** Python 3.13 for the local FastAPI analytic stack, PyTorch 2.10 CPU + scikit-learn for executed federated-training reference. Docker Compose is prepared but Docker is unavailable here. Flower ServerApp/ClientApp has been implemented but the Flower package/runtime could not be installed or executed in this restricted environment. Optional GPU/Colab later. Cloud deployment **not** verified.

**Nonnegotiable rules:** No real confidential data. No fabricated benchmark results. No unsupported formal DP guarantees. No evaluator-truth leakage to attack code. No custom secure-aggregation crypto.

**What is done:** Sprint 0 repository/research infrastructure; Sprint 1 3-site HTTP analytics, 110–107=3 leakage/history denial, audit and failure/concurrency tests; Sprint 2 executed PyTorch three-worker FedAvg IID/non-IID + separate evaluator, exactly reproduced IID, figures, native Flower Message API implementation and an automatic Flower workflow. 26 tests passed, one optional Flower test skipped. **No native Docker/Flower verification yet.**

**Where to resume:** `CURRENT_STATE.md` and `NEXT_SPRINT.md` are canonical. For how to work, read `AGENTS.md`, `docs/execution-loops.md` and `.agents/skills/` or `.claude/skills/`.

**Official research specification:** The detailed FedDisclosure Lab 2.0 proposal, November 2026, supplied by the project owner. The exact text is summarized in `docs/roadmap.md`, `docs/research-questions.md`, `docs/architecture.md`, `docs/methodology.md`, and `docs/threat-models.md`. The canonical requirement-by-requirement transcription is `docs/project-spec.md`. See `configs/` for implementation-ready definitions; do not treat proposed work as implemented.

**2026-11-19 v0.4 extension:** Ten paired-seed three-process training runs, 1/5-client pilot, controlled exact-gradient inversion (five exact/noisy matched seeds and an 8-image known-label experiment), output-only membership inference (five QUERY seeds, weak results), and research figures are executed. Threat model strengthened by separating label histograms from the coordinator metadata. `docs/disclosure-protocol.md` is now the Sprint 3 source of truth. Native Docker, Flower, DP, SecAgg+, external MNIST, Colab remain unverified. Git and the release ZIP carry evidence and active handoff.

**v0.4.1 measured extension:** same-image single-client SGD model-delta inversion with 1/4/16 actual local steps. Maintains clean distinction between exact observed gradients and update-derived approximation. Native Flower and actual multi-batch update inversion still unverified.

**v0.5.0 actual progress:** distinct-record 1/8/32 fixed-batch local SGD update inversion with explicit known-label privilege and five scorer runs; matched three-seed severe-non-IID 4-round FedAvg/FedProx (mu=0.3) PyTorch worker-process ablation, global means 55.19% vs 49.07%, zero worst-client in both. 44 fast tests passed, one optional Flower skip, two slow integration tests passed separately. Native Docker, Flower, OpenDP, Opacus, SecAgg+, external MNIST still unverified. Read `docs/fedprox-ablation.md` and `docs/disclosure-protocol.md` for actual metrics, limits and reproduction.
