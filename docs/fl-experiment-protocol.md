# Sprint 2 experiment protocol (v0.3)

## Research question and hypotheses

**Q-FL-01:** With three institutions and identical public digit samples, how does IID versus severe label-skew partitioning affect model quality and worst-client validation? **H0:** no change; **H1:** performance differs. Two 2-round smoke runs are not a statistical hypothesis test. Follow-up requires five independently seeded repeats and matched partition sizes.

**Q-FL-02:** Does reference FedAvg compute a sample-count-weighted parameter mean? Test with exact known PyTorch tensors and independent NumPy weighted averages before interpreting training.

## Datasets and observers

- Offline smoke: scikit-learn bundled 8x8 digits, a 600-example subset split into 120 public test images and 480 per-site samples. Each site is further partitioned into train and held-out validation.
- MNIST: `configs/training/fedavg_mnist.yaml` supported by dataset-preparation code, but its public files must be downloaded on a network-enabled host. **Not run in this environment.**
- IID and label-sorted non-IID partition options; site volumes contain separate NPZs and never appear in a coordinator request/response.
- Privileged preparation role handles public original images once and writes immutable partitions. Model coordinator reads manifest and weights only; each worker validates hashes locally.
- Separate evaluator process receives public test ground truth and global model for accuracy scoring. The optional pooled/local-only baseline is evaluator-only; this is **not** proof that pooled training is permitted operationally.
- Training coordinator sees per-client model parameters. No clipping, DP, SecAgg+ or confidentiality guarantee exists.

## Engines and gates

**Engine A (verified):** `python scripts/run_torch_reference.py --config configs/training/fedavg_digits_iid.yaml --data-dir data/training/digits-iid --output reports/results/sprint2/torch_iid_reference.json --references` uses real PyTorch and three client processes but **not Flower**. It is a testable numerical reference and exposes individual updates. Reproduce similarly with the `non_iid` config and a distinct data directory.

**Engine B (implemented, unverified here):** `python scripts/run_flower_experiment.py` launches genuine Flower 1.27 ServerApp and three ClientApps through the supported `flwr run` CLI and records the runtime logs and final checkpoint. Needs a network-enabled install of `flwr[simulation]` and `ray`. Flower apps communicate ArrayRecords and MetricRecords. The Simulation Engine is not a substitute for verifying remote independent SuperNodes with isolated volumes.

**Docker gate (scripted, unverified here):** `bash scripts/verify_compose.sh` requires Docker Engine. It asserts no external client ports, three separate database volumes, internal-only network and gateway loopback binding; the benchmark must then pass.

## Important measurement notes

- `public_test.accuracy` is measured by evaluator against a separate 20% public test holdout.
- `final_global_worst_client_validation_accuracy` is measured by independently running the **aggregated global model** over each institution's held-out validation samples.
- `worst_client_local_update_validation_accuracy` is instead measured on each locally trained model, not the global model. Do not conflate the metrics.
- `communication_model_bytes` counts float tensor payload up/down only, not actual serialized network bytes or TLS overhead.
- Accuracy after 2 rounds is smoke evidence only, not convergence; a 5-seed, longer-round ablation is pending.
- Site separation in the offline reference is process/logical separation, not OS-enforced filesystem isolation. Flower local simulation similarly shares a host unless deployed to independent SuperNodes.
- When claiming FedAvg exactness, define the weighting units (number of site training examples), dtype and aggregation precision. The test uses an independent NumPy weighted average.

## Current official API references

- https://flower.ai/docs/framework/1.27/en/tutorial-series-get-started-with-flower-pytorch.html
- https://flower.ai/docs/framework/1.27/en/ref-api/flwr.serverapp.strategy.Result.html
- https://flower.ai/docs/framework/1.27/en/how-to-run-simulations.html

## Checkpoints and release archives

Final model checkpoints are saved under `reports/results/sprint2/checkpoints/`, named by experiment ID, with their SHA-256 in the corresponding result JSON. `.pt` is ignored by Git to prevent accidental publication of future private-trained weights. Only the generated public-digit sample checkpoints are optionally bundled with the handoff ZIP.
