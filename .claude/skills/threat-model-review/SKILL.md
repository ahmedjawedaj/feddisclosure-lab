---
name: threat-model-review
description: Review or change a FedDisclosure analytics, training or model-output attack by specifying the real attacker, allowed observations, forbidden information and success metrics. Invoke before implementing attacks or security assertions.
---

# Threat model review

Read `docs/threat-models.md` and `docs/architecture.md`.

For each experiment, fill:

1. Attacker role and objective.
2. Dataset type and permission to use it.
3. Precisely observed object: global count, site count, individual gradient, local update, secure aggregate, or prediction API.
4. Attacker controls: queries, gradients, labels, clients, timing, participation, dropout.
5. Explicitly **not visible** objects, especially evaluator truth and raw rows.
6. Query/compute/time budget, target metric and failure threshold.
7. Defense intended to block the observation and its assumptions.

Add a unit test guaranteeing the attack has no evaluator-truth import. Add negative test showing unavailable observations cannot be used to infer successfully. If changing attacker capabilities, create a **new** experiment ID and do not compare across observers as identical metrics.

Never treat this as penetration-test authorization against a third party. Only work with synthetic/public controlled experiment systems.
