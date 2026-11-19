# Sprint 2 multi-seed training evidence — 2026-11-19

**Engine:** public sklearn 8×8 digit MLP + independent PyTorch client subprocesses + evaluator process; **not Flower**. 600 samples selected per seed, 20% separate evaluator test set, IID versus severe label-skew assignment of the same remaining public records. Each site holds local train/validation partitions. Five paired seeds 11, 22, 33, 44, 55; three clients, two rounds, one local epoch, SGD lr 0.15, batch size 32. Final global model evaluated on each site’s private validation. Model-payload communication is estimated, not measured transport traffic.

| Condition | n | Global test accuracy mean ± sample SD | Worst-client global validation mean ± sample SD |
|---|---:|---:|---:|
| IID | 5 | 0.2000 ± 0.06095 | 0.11875 ± 0.05135 |
| Severe label skew | 5 | 0.17833 ± 0.05420 | 0.0000 ± 0.0000 |

The results are **weak two-round smoke accuracies**. The worst-client zero is not a general statement about non-IID FL. No statistical significance test or model convergence claim is made. Paired per-seed differences and checkpoint SHA-256 values are recorded in `results/sprint2/sweep/sweep_summary.json`.

Single-seed client-count pilot (seed 11): 1 client IID accuracy 0.3833, 5 IID 0.1750, 5 severe label skew 0.1833; 3 IID 0.1833. **Optimizer step budgets and local batch counts differ**, so these are technical scaling smoke tests, not a fair 1/3/5-client methodological ablation yet. One client offers no concealment from its coordinator. Running 5 independent workers concurrently initially timed out on limited resources; revised runner caps concurrent OS workers at 3, and both 5-client conditions completed. Failed native Docker/Flower gates are separately recorded.

A schema-v2 data manifest now prevents coordinator-readable per-site label histograms. The distribution stats are evaluator-only. Legacy v1 manifests are intentionally rejected; recreate public data if resuming after the upgrade.

**Reproduce:** `python scripts/run_training_sweep.py --output reports/results/sprint2/sweep --seeds 11 22 33 44 55 --clients 3 --rounds 2`. See `docs/fl-experiment-protocol.md`.
