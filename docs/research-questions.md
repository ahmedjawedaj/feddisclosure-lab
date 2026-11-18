# Research questions and falsifiable hypotheses

**RQ-A1 (implemented):** Can two individually releasable aggregates reveal a group smaller than the local minimum cell threshold? **H-A1:** A baseline site-level cell threshold does not prevent differencing between an all-records total and its allowed complement. **Measure:** inferred count, absolute error, rejection rate. **Observer:** analytics user.

**RQ-A2 (partially implemented):** Can an institution-level near-duplicate history rule prevent this exact two-query sequence? **H-A2:** Rejection blocks this sequence, but may reject legitimate queries and does not guarantee security against more complex linear combinations. **Measure:** exposure/rejection trade-off over broader query workloads. **Observer:** analytics user.

**RQ-B1 (planned):** How does client heterogeneity affect FedAvg relative to local-only and pooled references? **Measure:** overall and worst-client accuracy, rounds, bytes, runtime.

**RQ-B2 (planned):** When are individual gradients sufficient for recognizable image reconstruction? **Measure:** MSE/PSNR/SSIM, label recovery, attack success and runtime across batch size and local steps. Distinguish gradients from accumulated local model updates.

**RQ-B3 (planned):** Can output-only queries distinguish training members from non-members? **Measure:** ROC-AUC and TPR at 1% or 0.1% FPR, reporting confidence intervals.

**RQ-D1 (planned):** What changes when a valid DP mechanism or secure aggregation removes an adversary observation? **Measure:** matched attack success, accuracy, budget and overhead. Do not compare across incompatible observers as though the scores were identical.

No paper novelty is asserted by these baseline questions. Reproduction failures are valid and informative outcomes.
