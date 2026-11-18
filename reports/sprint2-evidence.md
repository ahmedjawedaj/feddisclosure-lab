# Sprint 2 execution evidence (November 18, 2026)

## Measurement status

**Observed:** PyTorch 2.10 CPU baseline with three simultaneous independent client worker processes per training round and a separate evaluator process. Three clients, two rounds, public scikit-learn 8×8 digits, seed 20261118 and 600 selected examples. Client data split is immutable and per-site; no private records exist in this synthetic/public test.

**Not observed:** Native Flower runtime, Docker Compose containers, real MNIST download. Native import test skipped due missing `flwr` package, and package installation failed because package index DNS was unavailable.

## Exact reference commands

```bash
python scripts/run_torch_reference.py --config configs/training/fedavg_digits_iid.yaml --data-dir data/training/digits-iid-v2 --output reports/results/sprint2/torch_iid_reference.json --references
python scripts/run_torch_reference.py --config configs/training/fedavg_digits_non_iid.yaml --data-dir data/training/digits-non-iid-v2 --output reports/results/sprint2/torch_non_iid_reference.json --references
python scripts/run_torch_reference.py --config configs/training/fedavg_digits_iid.yaml --data-dir data/training/digits-iid-v2 --output reports/results/sprint2/torch_iid_repeat.json --references
python scripts/run_local_stack.py --once --output reports/results/analytics_local_sprint2_validation.json
pytest -q -m 'not slow'
pytest -q tests/training/test_runner_integration.py -k 'not label_skew'
pytest -q tests/training/test_runner_integration.py -k 'label_skew'
```

## Results

| Configuration | Observed global test acc. | Worst global-model site validation acc. | Public pooled reference acc. | Raw float bytes/round |
|---|---:|---:|---:|---:|
| IID | 0.2083333333 | 0.1875 | 0.3666666667 | 57,840 |
| Label-skew non-IID | 0.2 | 0.0 | 0.25 | 57,840 |

Record of repeat: IID public test (both rounds), final worst-client accuracy, evaluator reference metrics and checkpoint SHA-256 **matched exactly**.

`pytest -q -m 'not slow'` passed 24 with 1 Flower import skip and 2 slow tests deselected. Both independent-process tests passed in separate commands; 26 passing tests in total. An earlier full command displayed all 26 passes and 1 skip, but its surrounding runner timed out after pytest's result output; the smaller invocations have confirmed successful exit codes.

## Reproducibility and limitations

Config is embedded in each JSON artifact. The final models are stored as `.pt` checkpoints (ignored by Git, included only in the downloadable handoff ZIP for these approved public digits). Data partitions are generated from scikit-learn and excluded from Git/ZIP; rerun the command to recreate them. Model tensor bytes are estimates, not measured network payload size. One seed/two rounds are not a confidence interval, significance test or convergence experiment. The pooled reference sees public original training data by design, unlike federated deployment. No formal privacy claim applies.
