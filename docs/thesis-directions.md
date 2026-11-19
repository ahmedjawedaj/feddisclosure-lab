# Candidate MS thesis directions — evidence-based shortlist

These are hypotheses to evaluate with faculty and current literature, **not established novelty claims**.

## 1. Attacker auxiliary information under aggregate update visibility

**Motivation:** The v0.8 aggregate target-only probe had poor reconstruction, while v0.9 joint reconstruction improved recovery under a much stronger oracle disclosing all class labels in site-grouped order. This confounds algorithm quality and information privilege.

**Research design:** Build a nested attacker-knowledge ladder: only aggregate, known number of clients, class histogram, exact multiset of labels, client-grouped labels, partial collusion, all-but-one collusion. Keep initialization, data, update schedule, and attack compute fixed. Use genuine Flower SecAgg+ after native verification to ensure the protocol fits the intended observer class. Report absolute reconstruction MSE, label recovery, uncertainty and negative cases.

**Potential contribution:** A carefully controlled empirical benchmark identifying which auxiliary information changes aggregate leakage bounds. Novelty must be checked against existing reconstruction and secure aggregation literature.

## 2. Worst-client utility and privacy risk under distribution shift

**Motivation:** The five-seed equal-work training study shows a *descriptive* worst-client gap for alpha=0.1 versus IID, but the 95% interval includes zero and the run is short.

**Research design:** Extend to full MNIST and a second public domain, longer 8–20-round convergence, 10+ paired seeds if feasible, equal-client-count and equal-compute baselines; compare FedAvg/FedProx, honest utility, clipping, real Opacus DP-SGD and per-client leakage. Report measured epsilon/delta separately from empirical attacks.

**Potential contribution:** Reveal conditions in which global utility masks poor subgroup/client utility while privacy budgets are consumed. No guarantee of novel findings.

## 3. Governance-aware federated analytics release budgeting

**Motivation:** History-aware suppression blocks a particular synthetic differencing chain, but its 50% query rejection and inability to offer a formal privacy guarantee are important limitations.

**Research design:** Model heterogeneous site release policies, bound composition under OpenDP mechanisms, measure accepted-query rate, privacy expenditure and statistical utility; compare simple history rules and formally accounted release methods. Require per-site adjacency and accounting assumptions to be explicit.

**Potential contribution:** Reproducible decision framework or benchmark for policy heterogeneity. Must not substitute a risk score for formal DP.

## Gate before thesis selection

Native Docker/Flower/OpenDP/Opacus runs, external literature review, and faculty guidance are mandatory before deciding novelty or choosing a paper venue.
