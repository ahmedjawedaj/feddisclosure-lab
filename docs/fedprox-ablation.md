# Matched FedAvg versus FedProx ablation (v0.5.0)

## Research question

Does a proximal regularizer (`mu=0.3`) improve client fairness or held-out
accuracy against unregularized FedAvg in a small severe label-skew experiment?

## Observation and data boundaries

This is **not native Flower**. Training runs three independent PyTorch worker
processes. Each holds only an approved public `sklearn_digits` data partition
with a SHA-256 check. The coordinator combines model parameters and receives
scalar metrics/sample counts. Its execution environment shares a filesystem
with these processes, so this is a logical rather than OS-enforced isolation
boundary. An independent evaluator process scores the public held-out test set.
The coordinator does **not** read raw image arrays or label histograms.

## Protocol

Three paired dataset/initialization seeds `241,242,243`; 900 sampled public
8x8 digits per seed; 20% evaluator-only test split; severe class-sorted
non-IID partition among three clients. Four FedAvg rounds, two local epochs
per round, mini-batch 32, SGD rate 0.15, identical site data order and sample
counts. Compared `FedAvg(mu=0)` with `FedProx(mu=0.3)`; FedProx locally
minimizes `cross_entropy + mu/2 * ||w-w_global||_2^2` each step. Both
strategies use identical sample-count weighted aggregation. Model traffic
counts tensor bytes only, not Flower/transport overhead. There is no secure
aggregation or differential privacy.

## Actual measurements

| Seed | FedAvg public test | FedProx public test | FedProx minus FedAvg |
|---|---:|---:|---:|
| 241 | 51.11% | 47.78% | −3.33 pp |
| 242 | 54.44% | 49.44% | −5.00 pp |
| 243 | 60.00% | 50.00% | −10.00 pp |
| Mean | **55.19%** | **49.07%** | **−6.11 pp** |

Paired-difference sample standard deviation is 3.47 percentage points. In
**all six runs**, final worst-client global-model validation accuracy was
**0%**. Each arm used the same estimated traffic budget of 231,360 tensor
bytes per run. FedProx did not outperform FedAvg at this single selected
`mu` in this short, highly skewed setting. Do not infer a general superiority
claim about FedAvg or FedProx, or that every real client has zero accuracy.

## Reproduction

```bash
python scripts/run_fedprox_ablation.py --seeds 241 242 243 --rounds 4 --mu 0.3
python scripts/plot_new_benchmarks.py
python scripts/verify_research_evidence.py
```

Results: `reports/results/sprint2/ablations/paired/paired_summary.json` and
six detailed per-run JSON files/checkpoints. Figure:
`reports/figures/sprint2_fedprox_paired.png`. Rerunning refuses changed
cached configs and verifies the immutable input manifests and checkpoint hashes.

## Limitations / next work

Only three seeds; hyperparameters were not selected against validation data.
Severe label skew is not representative of all non-IID data. Model remains
undertrained; a zero worst-client validation outcome prevents claiming a
practical fairness improvement. Need more rounds, matched centralized
optimizer budgets, multiple `mu` values, and actual Flower cross-checks.
