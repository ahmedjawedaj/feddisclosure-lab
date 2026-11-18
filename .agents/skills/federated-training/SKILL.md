---
name: federated-training
summary: Reproducible and threat-model-safe Flower/PyTorch experimentation
---
# Federated training skill

Read `docs/threat-models.md`, `docs/fl-experiment-protocol.md`, `CURRENT_STATE.md`, and `NEXT_SPRINT.md` before changing ML experiments.

1. Specify exact data provenance, partition hash, random seed, held-out evaluation, optimizer, and observer permissions.
2. Data preparation is a privileged setup role. Client code loads *only* its own train/val partition. Coordinator receives model arrays, sample counts, scalar metrics; evaluator alone sees public test ground truth.
3. Test FedAvg weighted aggregation numerically against a separate NumPy calculation. Verify participant counts and fail closed on incomplete rounds.
4. Execute the offline independent-process reference with `python scripts/run_torch_reference.py`; this is *not* proof of Flower deployment.
5. On a dependency-enabled host, run `python scripts/run_flower_experiment.py`; only after the actual `flwr run` succeeds may native Flower claims be marked verified.
6. On a Docker host run `bash scripts/verify_compose.sh`; do not claim the Docker gate based on YAML parsing.
7. Keep IID and non-IID benchmarks separate. Do not compare local-only and pooled baselines without stating their different training schedules and privilege assumptions.
8. Record failed runs, exact commands, model/accuracy/communication metrics, and honest limitations before updating state files.
