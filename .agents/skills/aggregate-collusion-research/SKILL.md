# Aggregate observation and collusion research

1. Read `docs/v0.8-research-protocol.md`, `docs/v0.8-results.md`, `docs/threat-models.md` and `CURRENT_STATE.md`.
2. Keep `server_individual_model_delta`, `server_aggregate_model_delta_target_probe` and `server_collusion_isolated_delta` observer roles distinguishable in attack reports. `aggregate_only` uses a deliberately misspecified target-only proxy.
3. Never claim that an arithmetic mean is cryptographic secure aggregation. Flower SecAgg+ acceptance is separate and blocked until executed natively.
4. Preserve privileged known-target-labels assumption, exact participant count (3), equal sample weighting, one local SGD step and evaluator-only Hungarian matching.
5. Run `pytest -q tests/training/test_aggregate_collusion.py`, `python scripts/verify_aggregate_collusion.py` and `python scripts/plot_aggregate_collusion.py` before reporting. Do not silently regenerate evidence with different seeds or overwrite raw observation folders.
6. Do not commit raw observations, source images, checkpoint tensors or evaluator-only labels. Scalar scores/figures only after review.
