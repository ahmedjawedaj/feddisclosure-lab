# NEXT SPRINT — Native runtime gates and stronger disclosure evidence (post v0.5.0)

## Newly completed (observed, not just designed)

- [x] One/8/32 **distinct-record** fixed local-batch model-update inversion across 1/4 SGD steps (five scored runs, attacker/evaluator separated).
- [x] FedProx locally implemented/tested and three matched severe non-IID FedAvg/FedProx four-round PyTorch references completed (one selected mu=0.3, negative results preserved).
- [x] Five attack case scores and six trained checkpoints plus hashes checked in read-only evidence verifier; two new plots regenerated.
- [x] Fast tests 44 passed, optional Flower test skipped; two slow integration tests passed **separately**; four-process HTTP disclosure regression passed.

## P0 — Still a release blocker; do not bypass

1. Run `bash scripts/verify_compose.sh` on a Docker Engine + Compose v2 host. Require the native network/volume/isolation evidence and real HTTP benchmark. Docker binary is absent here.
2. On Python 3.12 with package access run `pip install -e '.[ml,test]'` and `python scripts/run_flower_experiment.py`. Confirm genuine three-node Flower ServerApp/ClientApp evidence, weights/metrics/checkpoint and strict participant count. Cross-check one identical Flower round against reference initial weights, site samples, order and aggregation. Flower is absent here.
3. Trigger GitHub CI on an authorized remote and preserve workflow run IDs and artifacts. No remote or completed Actions execution is claimed.

## P1 — Experimental depth

1. Improve model convergence: approved MNIST with data hashes, longer rounds, multiple training seeds, matched local-only / pooled reference optimizer budget. Preserve worst-client scores and full per-round curves. Existing four-round results on 8×8 digits are insufficient for a scientific accuracy claim.
2. Compare multiple non-IID severities and several FedProx `mu` values with registered selection protocol; investigate zero worst-client accuracy and quantify uncertainty. Current data partitions are intentionally severe.
3. Invert **actual changing-minibatch** local updates with batch 1/8/32, local 1/4/16 steps, and (where feasible) Flower ClientApp training. Record known vs unknown-label threat privileges, including attack failures and timing/communication costs.
4. For output-only membership inference, train five independent target models rather than merely resampling API requests to one frozen model. Record TPR at low FPR and model utility.
5. Enforce attacker/evaluator OS-level permissions rather than relying only on directory roles and CLI contracts.

## P2 — Defenses, benchmarks, thesis readiness

- Implement OpenDP with explicit count adjacency, sensitivity, epsilon/delta and release composition; Opacus with calibrated DP-SGD and accountant; Flower SecAgg+ with valid threshold/dropout settings. Run defended/unprotected matched baselines. No cryptographic protocol reimplementation.
- Benchmark actual bytes, accuracy/utility, latency, information budgets and failure rates. Run multiple seeds, regenerate figures and technical report, prepare faculty review.

## Reproduction commands for new evidence

```bash
python -m pip install -e . --no-build-isolation --no-deps
pytest -q -m 'not slow'
pytest -q 'tests/training/test_runner_integration.py::test_independent_worker_training[iid]'
pytest -q 'tests/training/test_runner_integration.py::test_independent_worker_training[label_skew]'
python scripts/run_local_stack.py --once
python scripts/run_fedprox_ablation.py --seeds 241 242 243 --rounds 4 --mu 0.3
python scripts/run_multirecord_update_inversion.py --site-file data/training/digits-iid/institution-00/train.npz --output-dir reports/results/sprint3/multirecord_update --attack-iterations 160 --seed 241
python scripts/plot_new_benchmarks.py
python scripts/verify_research_evidence.py
```

The extended scripts can preserve completed run files and refuse stale configs/partition/checkpoint mismatches. See `docs/fedprox-ablation.md` and `docs/disclosure-protocol.md` for exact observer privileges.

---

# Previous v0.4.1 next-sprint ledger

**At handoff (2026-11-19, v0.4.1):** Core analytics, five-seed offline PyTorch reference, exact-gradient reconstruction and output-only membership baseline are executed. Docker and Flower **not verified**; do not misrepresent these as complete.

## P0 — Native runtime acceptance (still blocked here)

1. **Compose:** On a host with Docker Engine/Compose v2, execute `bash scripts/verify_compose.sh`. Require actual healthy services, three distinct SQLite volumes, internal network, loopback-only gateway and executed analytics benchmark. Save logs, system versions, `docker_isolation.json` and JSON metrics. Fix and rerun any failing gate.
2. **Flower:** Use supported Python 3.12 on a network-enabled host: `pip install -e '.[ml,test]'`. Run `python scripts/run_flower_experiment.py`. Confirm 3 actual ClientApp participants over 2 rounds, privacy boundary, aggregated validation and checkpoints; preserve full logs and failure evidence. Compare one Flower round against an **identical** Torch numerical reference including data order, initialization, weight units, dtype, local batch schedule. Do not claim local Flower simulation establishes filesystem-isolated SuperNodes.
3. Trigger the `flower-smoke.yml`, `research-smoke.yml` and Compose CI workflows on an authorized GitHub remote and store actual green run IDs/artifacts; workflows have not run in this environment. If using remote institution clients or Colab, require separately provisioned SuperNodes, explicit mount/identity/hash controls, versioned data and secure coordinator connectivity. Do not assume indefinite Colab uptime.

## P1 — More meaningful FL training and observer alignment

1. Run approved MNIST with SHA-256 provenance and split manifest, including a real image dataset license check. Keep datasets out of Git.
2. Train long enough to obtain stable, nontrivial utility, with 5 **training** seeds and matched optimizer budgets for local-only, pooled reference and FedAvg/FedProx. Preserve worst-client global-model validation and distribution-shift details.
3. **Done in idealized one-record SGD:** reconstruction from genuine 1-, 4-, and 16-step local parameter deltas. **Next:** actual multiple *different* local examples/minibatches, varying batch 1/8/32, known vs unknown labels, realistic Flower client updates, and honest-but-curious server observation boundaries. Compare with exact-gradient benchmark, measure failures and isolate oracle information.
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
