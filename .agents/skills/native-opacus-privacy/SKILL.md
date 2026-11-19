# Native Opacus research defense

1. Read `docs/v0.8-research-protocol.md`, `docs/opendp-runtime.md` and `NEXT_SPRINT.md`. The OpenDP query-count and Opacus DP-SGD tracks have different adjacency/observation claims.
2. Run `python scripts/run_opacus_dp_sgd.py` only with native `opacus` installed. Never fake per-sample noise or manually claim epsilon. Native `PrivacyEngine.get_epsilon(delta)` must be executed and stored.
3. Treat this as ONE local DP-SGD training invocation. Repeated client rounds require a continuous accountant per site, and any separately released private queries require separate composition analysis.
4. Record Poisson sampling, record adjacency, RDP accountant, clipping norm, actual noise multiplier, achieved epsilon, target delta and single-silo public-test utility.
5. Run `pytest -q tests/privacy/test_opacus_training.py` and the native Opacus CI gate on Python 3.12. If skipped/unavailable, mark code as unverified and do not provide a protected model result.
6. `secure_mode=False` is research-only; do not advertise cryptographic-quality randomization or production DP.
