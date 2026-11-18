---
name: privacy-claims
description: Validate statistical suppression, differential privacy, gradient clipping and secure aggregation claims against their assumptions and the exact attacker observation channel.
---

# Privacy claim checklist

Read `docs/threat-models.md`, `docs/methodology.md`, `docs/decisions.md`.

- **Suppression/history:** empirical heuristic only unless there is a valid theorem; explicitly assess query combination and rejection-channel leakage.
- **Differential privacy:** define adjacent datasets, bounded contribution, mechanism, sensitivity, (epsilon, delta), accountant and adaptive composition. Use established OpenDP or Opacus APIs. A noised count alone is not evidence of correctly accounted DP.
- **Secure aggregation:** use Flower SecAgg+ instead of custom crypto. Record threshold, participant count, dropout/collusion assumptions and exactly which updates the server sees.
- **Gradient clipping:** does not by itself give DP; label utility/security conclusions accordingly.
- **Evaluation:** pair protected and unprotected runs under the same observer, partitions, random seeds and attack budget. Report runtime, accuracy, attack success, and confidence intervals when available.

Never combine unrelated defense metrics into an unjustified single privacy score. Mark all unfounded claims as unverified.
