# FedDisclosure Lab — Faculty Research Briefing

**Working title:** Privacy Leakage and Defense Evaluation in Federated Analytics and Federated Learning
**Status:** Independent research engineering prototype, not a submitted MS thesis or novelty claim.
**Audience:** Faculty discussion with Dr. Shahzad Mumtaz and Dr. Zubair Khalid.

## Suggested 10-slide structure (8–12 minutes)

1. **Problem.** Institutions can keep raw records locally yet reveal information through aggregate queries, client updates, and model predictions. Distinguish *non-disclosure of rows* from guarantees about inference.
2. **Research question.** How does attacker access (query answers, individual updates, aggregate-only, output-only) affect leakage, and what would actual defenses need to guarantee?
3. **System.** Three isolated logical institution datasets with independent FastAPI/SQLite services; a local PyTorch multi-worker reference and separate attacker/evaluator roles. Docker Compose and native Flower are prepared but **not verified**.
4. **Statistical disclosure result.** Synthetic 110 vs 107 query responses reveal subgroup count 3. A history-aware query control blocks this particular second query; it is not formal differential privacy.
5. **Federated training baseline.** 20 actual matched PyTorch runs across IID and capacity-constrained non-IID partitions (5 seeds; four rounds). Strong heterogeneity can have high global accuracy and lower worst-client utility, but confidence intervals cross zero.
6. **Individual-update leakage.** Exact/near-exact gradients and one-step updates can recover public 8×8 digit images under privileged attacker assumptions. Longer local training and larger/mixed batches complicate the tested inverse problem.
7. **Aggregate and collusion risks.** Five-seed idealized arithmetic mean attack shows aggregate-only target probe less successful than individual or all-but-one collusion. A newer three-seed **joint 24-image** attack reaches pooled MSE ≈ 0.0573, but it is granted all labels grouped by institution. Neither experiment is SecAgg+.
8. **Privacy guarantees still missing.** OpenDP/Opacus integration source and accounting tests exist, but real native execution, matched protected-versus-unprotected comparisons, Flower SecAgg+, and OS isolation remain unverified.
9. **Reproducibility and demo.** Show the offline dashboard's training, aggregate-observer, membership inference, and verification tabs. Explain hashes, evidence, separate evaluator, and saved negative cases.
10. **Faculty feedback sought.** Which observer model merits deeper empirical research? Is a novelty gap plausible in governance-constrained participation, utility under non-IID shift, or effective attacker auxiliary information? Seek a narrow, testable thesis question rather than promising a new defense upfront.

## Questions to ask faculty

- Is the strongest contribution likely to be a rigorously controlled benchmark or a method that improves a clearly documented vulnerability?
- Which deployment assumptions are meaningful for local institutions: server collusion, client dropout, query histories, data heterogeneity or budget-constrained release?
- What minimum number of datasets, baselines, and replications would support an MS thesis or paper submission?
- Which public dataset could provide a defensible second domain without introducing confidentiality concerns?

## Live demo path

Open `dashboard/dist/index.html` locally, choose **Training utility** to show the gap between global and worst-client performance; switch to **Aggregate disclosure** to compare declared attacker privileges; then **Verification & limits** to show what is not yet validated. The dashboard is read-only; it does not train models or initiate attacks.

## Publication restraint

Do not claim a novel algorithm, validated digital twin, formal differential privacy, secure aggregation, institutional deployment, or statistically significant group differences from the current pilot evidence.
