# NEXT SPRINT — Native gates and realistic-observation benchmarks

**At handoff (2026-11-19, v0.4):** Core analytics, five-seed offline PyTorch reference, exact-gradient reconstruction and output-only membership baseline are executed. Docker and Flower **not verified**; do not misrepresent these as complete.

## P0 — Native runtime acceptance (still blocked here)

1. **Compose:** On a host with Docker Engine/Compose v2, execute `bash scripts/verify_compose.sh`. Require actual healthy services, three distinct SQLite volumes, internal network, loopback-only gateway and executed analytics benchmark. Save logs, system versions, `docker_isolation.json` and JSON metrics. Fix and rerun any failing gate.
2. **Flower:** Use supported Python 3.12 on a network-enabled host: `pip install -e '.[ml,test]'`. Run `python scripts/run_flower_experiment.py`. Confirm 3 actual ClientApp participants over 2 rounds, privacy boundary, aggregated validation and checkpoints; preserve full logs and failure evidence. Compare one Flower round against an **identical** Torch numerical reference including data order, initialization, weight units, dtype, local batch schedule. Do not claim local Flower simulation establishes filesystem-isolated SuperNodes.
3. Trigger the `flower-smoke.yml`, `research-smoke.yml` and Compose CI workflows on an authorized GitHub remote and store actual green run IDs/artifacts; workflows have not run in this environment. If using remote institution clients or Colab, require separately provisioned SuperNodes, explicit mount/identity/hash controls, versioned data and secure coordinator connectivity. Do not assume indefinite Colab uptime.

## P1 — More meaningful FL training and observer alignment

1. Run approved MNIST with SHA-256 provenance and split manifest, including a real image dataset license check. Keep datasets out of Git.
2. Train long enough to obtain stable, nontrivial utility, with 5 **training** seeds and matched optimizer budgets for local-only, pooled reference and FedAvg/FedProx. Preserve worst-client global-model validation and distribution-shift details.
3. Reproduce inversion from an **actual single-step local model delta**, then 2+ local steps, 1/8/32 examples, partial/no labels, different observer capabilities. Compare with exact-gradient benchmark, measure failures and isolate oracle information.
4. For output-only membership, train and score multiple independently seeded models, test reasonable stronger attacks and report uncertainty/low-FPR reliability. Keep model-user confidence-only threat model separate from gradient-observing server.
5. Add OS-level attacker/evaluator isolation and atomic artifact access controls; currently separation is logical only.

## P2 — Safeguards under matched scenarios (not implemented)

- Integrate **OpenDP** with explicit adjacency, sensitivity and accounting for aggregate releases, and **Opacus** with accountant, clipping and epsilon/delta for PyTorch training. Do not label the current Gaussian gradient noise DP.
- Integrate Flower **SecAgg+** rather than implementing custom cryptography. Require participant threshold, dropout/collusion assumptions and a native Flower runtime.
- Evaluate protected and unprotected matched observations, prediction accuracy, communication and operational cost. Avoid one combined privacy score.

## Reproduction from handoff archive

```bash
python -m pip install -e '.[research,test]'  # network-enabled setup; core already installed here
pytest -q -m 'not slow'
pytest -q tests/training/test_runner_integration.py::test_independent_worker_training\[iid\]
pytest -q tests/training/test_runner_integration.py::test_independent_worker_training\[label_skew\]
python scripts/run_local_stack.py --once
python scripts/verify_research_evidence.py
python scripts/plot_disclosure.py
```

See `docs/disclosure-protocol.md` for exact individual attack reproduction commands and `docs/fl-experiment-protocol.md` for reference training.

## Exit criteria

Do not mark Sprint 1 Docker or Sprint 2 native Flower closed without actual runtime logs and cross-checks. Future attacks must operate under stated observations, with evaluator-only truth. Every milestone requires tests, machine-readable artifacts, limitations, changes to `CURRENT_STATE.md`/`NEXT_SPRINT.md`/`docs/decision-log.md`, and a coherent Git commit. Preserve observed failures and negative findings. No claims of production privacy, formal DP, validated digital twin or publishable novelty.
