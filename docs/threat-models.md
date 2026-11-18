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
