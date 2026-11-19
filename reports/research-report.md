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

## Sprint 2: federated-learning reference experiment (2026-11-18)

**Status:** Executed PyTorch multi-process numeric reference, **not native Flower and not a thesis contribution**. The protocol is `docs/fl-experiment-protocol.md` and the config is in `configs/training/`.

A bundled public 8×8 digit dataset supplied 600 examples. A deterministic setup step reserved 120 for evaluator-only test and partitioned the rest among 3 institutional clients, with held-out validation inside each site. Each client trained a small PyTorch MLP locally and returned only its parameters, sample count, and scalar metrics. The coordinator performed sample-count-weighted FedAvg across two rounds. A separate evaluator process scored the global model. Evaluator-only public-data pooled/local-only comparators were trained from the same initial weights, but their optimization schedules do not exactly match federated training.

| Configuration | Global public test accuracy | Worst-site validation accuracy of final global model | Pooled public-data reference |
|---|---:|---:|---:|
| IID | 0.2083 | 0.1875 | 0.3667 |
| Severe label skew | 0.2000 | 0.0000 | 0.2500 |

Each round transmits an estimated 57,840 raw tensor bytes across 3 clients (up + down), excluding protocol overhead. The exact IID rerun matched both public test scores, final worst-client validation and reference baseline scores. The zero worst-site accuracy under label skew is a negative result worth studying at longer schedules and with multiple seeds, **not** evidence that real federated institutions will have zero accuracy.

The current benchmark cannot support a statistical effect claim: one random seed, only two training rounds, a small public dataset, and no hyperparameter search. It also cannot establish a Flower deployment or a production privacy guarantee. Individual client update access is granted to the server in this experiment. No secure aggregation or DP is used. Real Flower and Docker remain unverified pending a native runtime.

**Primary artifacts:** `reports/results/sprint2/torch_iid_reference.json`, `torch_non_iid_reference.json`, `torch_iid_repeat.json`, and `reports/figures/sprint2_training_reference.png`.


## Sprint 3: controlled disclosure observer comparison (2026-11-19)

**Status:** Executed reproducible *public-data* method demonstrations. The attacks and privileged evaluator are separate CLIs, not sandbox-isolated OS identities. All ground-truth original images or member tags are evaluator-only artifacts, while attacker input files contain only an explicit observer's allowed information.

For **exact, individual first-step gradients** on public 8×8 handwritten digits and a known MLP, a DLG-style Adam attack recovers single-image inputs with mean MSE 2.43e-9 across five public digit/image seeds. The final-layer bias-gradient minimum inferred the label without being supplied. A matched experiment injecting independent Gaussian noise of 0.01 standard deviation into each gradient entry produced mean reconstruction MSE 0.001211. This is an *uncalibrated noise ablation without DP accounting* and cannot establish an epsilon/delta privacy guarantee. A separate eight-image batch case had mean MSE 0.010676 but gave the attacker true labels, so its privileged conditions are not directly comparable to single-image label inference.

A more relevant **individual client model-update observer** was then tested. A tiny client repeatedly trained on one approved public image for 1, 4 or 16 SGD steps, revealing initial model parameters and its final parameter delta. The attacker formed an averaged-gradient proxy using known learning rate and step count. Reconstruction MSE increased from 2.03e-9 (1 step) to 0.02717 (4) and 0.14035 (16); the proxy-gradient relative error also increased. This tests a real multi-step local SGD model delta but **does not** generalize to Flower's multi-example local updates or secure-aggregated observations. Negative performance is evidence of this specific attack’s assumptions, not proof that local updates are safe.

A separate **output-only membership** attack ranked full softmax probabilities using an uncalibrated max-confidence heuristic, with the evaluator alone knowing train/held-out labels. Five query-subsampling seeds on one frozen two-round public-digit model had mean ROC-AUC 0.4781 (sample SD 0.0183), essentially at chance. These are not independently trained target models and do not justify broad membership-privacy conclusions.

The new machine-readable artifacts and figure-generation script support detailed review, reproducibility and honest negative findings. These are applied-research engineering benchmarks, not yet an original algorithm or publication-ready experimental study. See `reports/sprint3-evidence.md`, `docs/disclosure-protocol.md` and `docs/threat-models.md`.
