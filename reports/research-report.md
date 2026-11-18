# Research technical report — working draft

**Title:** FedDisclosure Lab: Reproducible Disclosure Risks in Federated Analytics and Learning

**Status:** Sprint 1 pilot; not a research paper.

## Abstract (pilot)

We build a synthetic three-institution federation and demonstrate that releasing individually acceptable aggregate counts can disclose a restricted subgroup through differencing. Each institution enforces a minimum count threshold of five. An analytics observer combines a total-count query (110) and an allowed negative-subgroup count (107) to recover three restricted records. A near-duplicate history rule rejects the second query in this specific sequence. The observed mitigation has a 50% rejection rate over the two attempted queries and should not be interpreted as a comprehensive or formally private mechanism.

## Method

See `docs/methodology.md`, `docs/threat-models.md` and `configs/analytics/differencing.yaml`. Four live local HTTP processes were used, with three separate site databases. Attacker code accepted released counts only; the evaluator independently held synthetic truth.

## Results

Source: `reports/results/sprint1_analytics_verified.json`. Naive mode inferred 3, absolute error 0 and a successful small-group disclosure. History mode emitted no second response and the attacker made no guess. A blocked guess has `absolute_error=null`, not zero.

## Validity and limitations

Single deterministic population and query order; no formal privacy proof; no heterogeneous policies, confidence intervals, randomized DP release or live Docker isolation validation. This does not measure risk at real institutions. Further experiments must include overlapping-query families, alternative orders and adaptive attack budgets.

## Next experiments

Complete Docker verification, add more statistical attack families, then integrate distributed Flower training and match gradient observability to the threat model. The initial results do not establish a novel scholarly contribution.
