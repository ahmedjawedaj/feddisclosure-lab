# Experimental methodology and benchmark integrity

## Core requirements

Every run begins with a versioned YAML config, an explicitly defined attacker, a known observation channel and a metric. Separate ground truth from attack inputs. Use deterministic synthetic data, record environment/software versions, save failed or non-disclosing runs and maintain regression tests.

## Sprint 1 benchmark

- Synthetic sites have 40, 35 and 35 rows, each containing exactly one synthetic positive record.
- `all_records` and `negative_records` are the only allowlisted queries. There is deliberately no arbitrary SQL or direct `positive_records` release.
- `min_cell=5` prevents directly returning fewer than five records. Both baseline released counts exceed five at every institution.
- `naive` releases both queries, yielding 110 and 107. The attack module receives these **only** and infers 3.
- `history` checks whether the new membership set differs by 1–4 records from a prior release within the same experiment and prevents that second release. This comparison is done inside each site, and the release history is private.
- Record whether the attacker can make a guess, its absolute error (if any), small-group exposure and the proportion of attempted queries rejected. A blocked attack's error is `null`, **not zero**.

The two policy conditions are not claims of a globally optimal privacy–utility trade-off. Query rejection itself can reveal information; no history-composition proof is made.

## Sprint 2 and beyond methodology plan

Use identical dataset partitions, random seeds and training budgets for local-only, pooled, FedAvg and FedProx. For reconstruction report exactly whether the adversary saw an individual sample gradient, batch gradient, one local update, multiple local updates or only a secure aggregate. Report negative reconstruction results with the same effort budgets.

For statistically meaningful results move from 1 pilot seed to at least five independent seeds, publish confidence intervals and isolate all ablations. Match model architecture, epochs and early stopping across comparisons. Track training wall time, per-round communication and privacy accountant states.

## Reproduction

```bash
pytest -q
python scripts/run_local_stack.py --once --output reports/results/reproduction.json
```

No Docker daemon was available in the initial build environment. Docker gate is separate and must be executed before claiming containerized deployment was validated. Save the **actual** stdout, versions, hardware and result artifacts for each run.

## v0.4 controlled model-disclosure methodology

Exact-gradient observations are produced by a privileged setup utility using approved public digits. Attacker and evaluator run as different CLI processes and receive distinct artifact files. The attacker has no ground-truth path argument. This is a *logical software boundary only*. Detailed protocol and caveats: `docs/disclosure-protocol.md`.

Five paired image/gradient seeds compare exact gradients with Gaussian-perturbed gradients under fixed model initialization and record selection. These are empirical attack outcomes and carry **no formal DP guarantee**. The image evaluator uses an optimal assignment for multi-image batches; the attacker does not. Report MSE, PSNR, SSIM and attack budget separately.

Membership inference uses a frozen public-digit reference checkpoint and only softmax vectors exposed to an uncalibrated max-confidence heuristic. Membership ground truth is evaluator-only; class-matched balanced samples make this an evaluation scenario, not an assumed deployment membership prior. Five query seeds vary the evaluated subset; they are not training replicates.


Model-delta inversion pilot: a privileged simulator trains a copy of the public MLP with one approved public record repeated for 1/4/16 SGD steps. The attacker receives only initial model state, local parameter difference, learning rate and step count and constructs its own gradient proxy. Evaluation compares recovered images against separate public ground truth and measures proxy error against the first-step gradient privately. This intentionally narrow setup is a bridge to real multi-example FL updates, not a representative server-attack success rate.
