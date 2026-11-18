# FedDisclosure Lab 2.0 — canonical implementation specification

**Date:** 18 November 2026. **Origin:** user-provided research proposal, preserved as actionable requirements. **Status:** specification, not a claim of completion.

## Project definition

Research theme: applied computer science, privacy-preserving AI, federated systems and security evaluation. Main question: when institutions keep raw data local, what sensitive information may still be inferred or reconstructed from statistical results, gradients/updates and released models, and which privacy safeguards work under which conditions?

Initial scope: synthetic and approved public datasets, three to five genuinely independent institutional services, real collaborative training, controlled disclosure-attack reproductions, established privacy defenses, reproducible metrics, open-source release and thesis-topic shortlist. No novelty or thesis publication claim at the start.

## Goals

| ID | Required outcome |
|---|---|
| G1 | Independent multi-client deployments with separate datasets, databases, processes and communication boundaries |
| G2 | Real federated neural network training under heterogeneous and non-IID data |
| G3 | Reproduce aggregate-query differencing, gradient inversion and other model leakage under explicit attackers |
| G4 | Compare statistical controls, differential privacy and secure aggregation against matched threats |
| G5 | Benchmark leakage, performance, statistical utility, privacy accounting and runtime/communication |
| G6 | Tested OSS code, results, demo, report, faculty presentation and thesis-question shortlist |

## Research tracks

**A. Federated Analytics (core):** cross-institution counts/summaries, small-cell and overlapping-query disclosure, history-aware query controls, DP-query mechanisms. Demo: sequential harmless-looking aggregate queries expose a restricted subgroup, then evaluate prevention.

**B. Federated Learning (core):** Flower + PyTorch, FedAvg, IID/non-IID, local/pooled/FedProx references, gradient inversion under individual-vs-batch-vs-updates, membership inference under output-only API. Demo: public MNIST original vs actual reconstruction attempt with scored failure/success.

**C. Simulation/digital twin (optional):** dynamic institutional populations and data distributions. A genuine digital twin is future-only unless a validated real-system reference/collaboration exists; never block core A/B.

## Architecture baseline

Research dashboard -> analytics gateway or Flower coordinator -> 3–5 independent institution services with private SQLite datasets -> attack and privacy evaluation lab -> reproducible reports. Evaluator sees synthetic truth, attacker sees only permitted observations. Begin on a single Docker Compose host, later optional GPU/Colab. FastAPI for site/gateway, SQL/Python for analytics, OpenDP for DP stats, Opacus for DP training, Flower SecAgg+ for secure aggregation, YAML runner, pytest CI, Streamlit read-only demo.

## Five core experiments

| ID | Method | Baselines / metrics |
|---|---|---|
| E1 | Aggregate differencing from overlapping/count queries | Inferred restricted count, release/error/rejection, latency |
| E2 | Gradient image reconstruction | MSE, PSNR, SSIM, label accuracy, success rate, runtime |
| E3 | Real federated training vs reference baselines | Mean and worst-client accuracy, convergence, bytes, training time |
| E4 | Defenses under consistent observations | Relative attack success, statistical error, model loss, epsilon/delta, overhead |
| E5 | Model-output membership inference | ROC-AUC, attack advantage, TPR at low FPR and target utility |

## Experiment matrix (pilot, not full Cartesian product)

- Clients: 1, 3, 5.
- Datasets: synthetic tabular and MNIST (public).
- Data distribution: IID, moderate non-IID, severe non-IID.
- Gradient batch sizes: 1, 8, 32.
- Local steps: one gradient step, multiple steps.
- Learning: local-only, pooled reference, FedAvg, FedProx.
- Protection classes: none, relevant statistical controls, OpenDP analytics, Opacus DP training, Flower SecAgg+.
- Repeats: 5 seeds after pilot; keep 1 seed smoke test fast.
- Observation: per-client gradients, local updates, aggregated updates, output-only model API, published aggregates.

Avoid uncontrolled combinations. Benchmark small core every commit; schedule expensive ablations selectively. Never aggregate unrelated privacy metrics into one invented privacy score.

## Threat model table

| Actor | Allowed observations | Experiment |
|---|---|---|
| Analytics user | Approved gateway-wide released aggregate answers | Query differencing / complementary inference |
| Honest-but-curious coordinator | Individual client updates **only if protocol exposes them** | Update reconstruction |
| Coordinator with SecAgg+ | Aggregated update only, meeting participant/threshold assumptions | Aggregate residual leakage |
| External model user | Queryable predictions and permitted confidence scores | Membership inference |
| Authorized evaluator | Synthetic ground truth, benchmark artifacts, reference labels | Scoring only; not an attack actor |

Track attacker query budget, labels known, sample/batch gradient vs local multi-step update, client and server roles, distribution, reconstruction budget, and null inference on denied observations.

## Seven sprints / twelve-week plan

| Sprint | Planned window | Exit gate |
|---|---|---|
| S0 Foundation | Week 1 | Repository, research questions, threat models, tests, YAML runner |
| S1 Analytics | Weeks 2–3 | Three Docker clients, cross-site gateway, statistical disclosure and defense tests |
| S2 FL Training | Weeks 4–5 | Three-client Flower + PyTorch training, baselines and saved metrics |
| S3 Model attacks | Weeks 6–7 | Genuine gradient reconstruction and membership inference, scored incl. failures |
| S4 Defenses | Weeks 8–9 | Matched OpenDP, Opacus, SecAgg+ configs and documented accounting/assumptions |
| S5 Benchmarks | Weeks 10–11 | Five-seed runs, controlled ablations, reproducible figures/tables and limitations |
| S6 Release | Week 12 | Demo, report, clean-install verification, CI and faculty presentation |

Checkpoints: W3 analytics demo, W5 federated ML, W7 model attacks, W9 privacy defenses, W11 benchmark report, W12 portfolio release. These are planning targets, not promised dates.

## Deliverable register

| ID | Deliverable | Priority | Current status |
|---|---|---|---|
| D01 | Research questions/hypotheses/threat models | Must | Initial docs built |
| D02 | Synthetic data generator with institutional distributions | Must | Deterministic baseline built; heterogeneity expansion pending |
| D03 | 3–5 independent Docker institutional clients | Must | Three processes built; container test pending |
| D04 | Federated statistical engine + disclosure | Must | Two-query version verified |
| D05 | Flower multi-client training | Must | Planned |
| D06 | Reproduced gradient inversion | Must | Planned |
| D07 | Model-output membership inference | Should | Planned |
| D08 | Statistical disclosure controls | Must | Threshold + limited history heuristic built |
| D09 | Differential privacy and secure aggregation comparisons | Must | Planned |
| D10 | Configurable experiment runner / metrics | Must | Analytics YAML/JSON runner built; expand |
| D11 | Automated tests, CI, reproducibility checks | Must | 18 tests + repeated local E2E + CI config; Docker CI not yet executed |
| D12 | Interactive research dashboard | Should | Planned |
| D13 | Benchmark dataset, results, visualizations | Must | Small analytics benchmark JSON + figure built |
| D14 | Research-style technical report | Must | Pilot draft; substantial work pending |
| D15 | Faculty presentation and thesis recommendations | Must | Planned |
| D16 | Digital-twin-inspired extension | Optional | Deferred |
| D17 | Validated real digital twin integration | Future | Excluded from v1 |

Minimum credible release includes D01–D06, D08–D11, D13, D14. D12 should never delay benchmark correctness.

## Release exclusions

No real patient data or external confidential systems, regulatory certification claims, bespoke cryptographic protocol, production hospital integration, LLM training/inversion, full digital twin, multi-cloud deployment, or promises of novel publishable contribution.

## Faculty and paper roadmap

First deliver reproducible open-source educational benchmark. Next write a methodical applied report. Then use evidence and faculty feedback (Dr. Shahzad Mumtaz and Dr. Zubair Khalid) to narrow a specific MS thesis candidate and assess novelty against literature. A paper requires new method, benchmark or well-substantiated empirical finding, not merely architecture or an existence proof.
