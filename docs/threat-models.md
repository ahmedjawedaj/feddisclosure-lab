# Threat models v0.2

## Analytics user (implemented)

**Control:** Makes preapproved, typed aggregate queries to the gateway with a research key. May combine multiple previously published answers and adapt the next query.

**Sees:** Query ID, policy chosen for this controlled experiment, gateway-wide count or a generic rejection. Does **not** see rows, site-level counts, membership sets or internal query history.

**Does not control:** Site DB contents, policy implementation, internal token, private evaluator.

**Success:** Guess a suppressed small subgroup count exactly by differencing two released counts. The demo infers `all_records - negative_records`.

**Defenses:** Site-level minimum release threshold, near-duplicate historical membership check. History control is a limited heuristic; overlapping multi-query attacks and information leakage from rejection patterns remain unresolved.

## Honest-but-curious FL coordinator (planned Sprint 3)

**Sees:** Per-client model updates only when protocol exposes them. No local raw data. This is different from individual per-example gradients and will be benchmarked separately.

**Success:** Recover data/labels under explicitly constrained model architecture, local steps, batch size and label knowledge.

## Coordinator with secure aggregation (planned Sprint 4)

**Sees:** Aggregate update only assuming participant threshold and no disallowed collusion/dropout. Not an individual update.

**Success:** Test residual leakage from the permitted aggregated observation; do not attribute attacks on per-client updates to this setting.

## External model-output user (planned Sprint 3)

**Sees:** Trained model predictions and possibly confidence scores under query budget; never gradients or member labels.

**Success:** Membership-inference ROC-AUC and TPR at defined low FPR versus matched non-member controls.

## Authorized evaluator

**Sees:** Separate known synthetic ground truth, run configs and private evaluation data, solely for measuring attacks. Cannot transfer this to attack implementations.

**Not an adversary.** Its capabilities must never be included in an attack success claim.

## Disclosure checklist for every experiment

Observer identity, training or query access, exact observations, number of participants, label knowledge, query and time budget, local step count, synthetic/public dataset, metric, failure criteria, exclusions and defense assumptions.

## Analytics gateway operator (documented, not attacked yet)

**Sees:** Individual institution-level aggregate responses in the gateway's internal HTTP traffic, plus request and policy. This actor is stronger than the external analytics user (which sees only the total or a generic refusal). Even if the public gateway refuses a query, the operator may have received responses from a subset of sites; those are not public responses.

**Implication:** The history rule for the external user cannot be claimed as protection against a curious gateway operator. Rejection can itself serve as a side channel. Site history updates are not a cross-service atomic commit; an interrupted/rejected federated query may leave local history entries, which deserves a future two-phase release-control design. No secure aggregation exists in Sprint 1.

## Federated learning coordinator (reference implemented; native Flower execution pending)

**Sees:** Per-client complete post-local-training model arrays and sample counts; aggregate scalar metrics. **Does not receive** raw sample arrays, private per-site label lists, or evaluator public test labels. Individual model updates are not individual-example gradients. One-client reconstruction experiments must not be described as multi-client leakage under this observer.

**Evaluator separation:** Offline reference evaluator and local client training execute as separate OS processes; the evaluator has permitted public ground truth. This does not enforce container-level filesystem permissions. CI and future Docker deployments must further isolate respective roles.

## Sprint 3: stronger controlled exact-gradient observer (EXECUTED, distinct from FL coordinator)

Receives a **full exact first-step gradient** for one isolated example or batch, known initial network weights, architecture and input dimensions. For batch 1 the true label is inferred through the output-layer bias-gradient minimum, assuming cross entropy/softmax. For batch 8 the oracle reveals labels (explicitly stronger privilege). It does **not** receive original image arrays, membership tags or evaluator-only optimal image assignment. This is not necessarily what a real coordinator sees after multiple local minibatches/steps and is not a SecAgg+-protected observer. Success is scored by the separate evaluator on approved public image data.

## Sprint 3: output-only model observer (EXECUTED OFFLINE REPLAY)

Receives softmax probability vectors only. It neither accesses model weights nor knows which predictions refer to member records; it returns confidence-based membership scores to the evaluator. The evaluator constructs matched member/non-member public digit queries, retains membership labels and computes ROC metrics. A deployed HTTP inference API with authentication and query budgeting is not yet tested. This is distinct from the gradient observer and the privileged evaluator.


## Sprint 3: unprotected individual model-delta coordinator (EXECUTED LOCAL SGD PILOT)

Sees initial and final client model parameters after known-step-rate SGD and may compute an **average-gradient proxy**, not exact intermediate step gradients. No original image pixels or ground-truth labels are passed to the attacker. One repeated public image under 1, 4 or 16 SGD steps is an intentionally privileged one-record client. This actor is closer to an unprotected FL coordinator than the exact-gradient observer, but still does **not** model multiple local batches, secure aggregation or a full Flower runtime. The evaluator separately measures proxy-gradient error and reconstruction quality.
