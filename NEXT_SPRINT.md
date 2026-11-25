## After v0.16.2 (2026-11-25)

Done in v0.16.2: second framework for the DP measurements (FedML 0.9.6).

## After v0.16.1 (2026-11-25)

Done in v0.16.1: stronger inversion attack on CIFAR-10 (plan item P3, R1 and R3 fail).

- [ ] Establish a plaintext baseline above the floor before reading aggregate rows again. Needs a new pre-registration (wider total variation grid, more local steps or a larger batch).
- [ ] Pre-register a rule for the SSIM reading before any claim is built on it.
- [ ] One more framework for the DP measurements (Flower and FedML are done). Check its release history and tracker first.
- [ ] Inventory of what the server sees with protections on (plan item P2).
- [ ] Composition and sampling amplification.

## After v0.16.0 (2026-11-23)

Done in v0.16.0: measured privacy of Flower's DP mechanisms (plan item P1, first framework).

- [ ] Run `scripts/audit_flower_dp.py` style measurements on two more frameworks. Check each framework's release history and tracker before writing predictions.
- [ ] Stronger inversion attack as the primary attack, on CIFAR-10 (plan item P3).
- [ ] Inventory of what the server sees with protections on (plan item P2).
- [ ] Composition and sampling amplification, which v0.16.0 did not measure.

## After v0.15.0 (2026-11-23)

Done in v0.15.0 (see `CHANGELOG.md`, `docs/preregistration/v0.15.md` and `docs/v0.15-*.md`):

- [x] One-round SecAgg+ error proof, property tests and a convex multi-round L2 test (E1, E3, M1 pass, E2 fails).
- [x] DP bound against an informed attacker for one full-batch step (D3 fails, the rest pass).
- [x] Wire-level validation on a deployed Flower 1.27.0 stack, 4 clients, one round (W1 to W6 pass, with the caveats listed).
- [x] Seeded-fault matrix (F1 and F2 pass, F2 on the white-box audit only, and the plaintext fallback evades the specified detectors).
- [x] External baselines (B1 fails, B2 passes).
- [x] Reply-order study and opt-in `DeterministicFedAvg` (P1 replay only, P2 fails, P3 cannot fail), six live hosted runs.
- [x] Scaling to 8 clients (S1 vacuous, S2 and S3 pass) and Fashion-MNIST (X1 fails by 0.6 points).
- [x] Two hosted 8-client Docker rounds for memory. This closes the old item "Docker topologies beyond 4 clients" for memory only.
- [x] Hosted CI has run since 2026-11-22. The old hosted CI items below are superseded.

Open, in the order they would change a conclusion:

- [ ] **Native 8-client attack capture.** The 8-client attack results (S1, S2) are in-process and the hosted 8-client rounds measured memory only. Run the wire capture at 8 clients and judge the same rules.
- [ ] **Informed attacker with a public prior against aggregate views.** Every statement that the aggregate sits at or below the label floor is conditional on a weak attack, and S1 passed vacuously. The attacker should start from labelled public images and model the quantiser.
- [ ] **Stronger inversion on protected views.** B1 failed (ratio 1.86 against a tuned `breaching` run). Tune the repository attack on tuning seeds, equalise iterations and rerun the aggregate, collusion and DP views, then reread S1, S2, X1 and the wire results.
- [ ] **Live MNIST reply-order configs for P2.** The configs that fail upstream in replay (4-client MNIST 10 rounds) and in the saved native runs (3-client MNIST seed 3 and label skew) have not been run live, with or without the strategy.
- [ ] **A CI job with the full environment** (`flwr`, `opacus`, `hypothesis`, research extras). The hosted unit job skips every module that needs `flwr` or `opacus`, so most of the v0.15 tests run only locally.
- [ ] **Tune X1 on the recognisable rate.** The `fashion-mnist-study` workflow ran once on a hosted runner (collusion +12.5 points against +9.4 locally, a second observation that leaves X1 at FAIL). Still open: more seeds, and tuning on the recognisable rate, because the MSE criterion disagreed with it for the collusion view.
- [ ] **E2 float32 term.** The float32 client path adds a bias of up to 3/16 of a quantisation unit that the float64 statement leaves out. State the bound with the term, or show it stays inside for other clipping and quantisation ranges.
- [ ] A gate that requires the SecAgg+ workflow marker and the expected log lines, because the plaintext fallback evades the current gates (F3).
- [ ] Seed a mask-reuse fault in the fault matrix, and inject partial faults (one client or one round).
- [ ] DP informed attack over several local steps, with Poisson sampling and an accountant.
- [ ] Inversion on captured multi-step traffic, and wire capture over several rounds.
- [ ] MNIST licence stays unresolved.

## After v0.14 Phase 1

- [x] Native Docker/Flower run, parity, repeat, non-IID and fault scenarios (smoke scale).
- [x] Phase 2: OpenDP releases + ledger, Opacus DP-SGD, upstream Flower SecAgg+ (executed, see docs/v0.14-privacy-secagg-results.md).
- [x] Phase 3 (partial): full MNIST native Docker run, multi-seed DP studies with CIs.
- [x] SecAgg+ with n=4: baseline, 1-round bound check, replicates and a five-scenario mid-round dropout matrix (rule confirmed, earlier claim withdrawn).
- [x] Native Docker non-IID MNIST run and two extra seeds (executed, 2 of 3 failed the 1e-5 parity gate, explained by float32 replay).
- [x] Stronger attack: LiRA shadow-model membership inference on non-private vs Opacus DP models (see results doc).
- [x] Fail-fast for lost participants (exit 125, validated in 3 Docker scenarios).
- [x] Float32 Flower-arithmetic replay gate added beside the float64 gate (7 of 7 native runs pass it).
- [x] Update-inversion and SecAgg-view comparison against protected vs unprotected observers (10 seeds).
- [x] Statistical acceptance rule for multi-round SecAgg+ error (Monte Carlo, 300 runs). A proven bound is still open.
- [x] LiRA at 64 shadows and 16 targets on Digits and MNIST.
- [ ] DP-aware inversion attacker, and inversion on captured multi-step Docker traffic.
- [ ] Prior-initialised aggregate attacker that uses labelled public images, to compare with the label-only floor (the v0.14.2 attack uses none, so the v0.14.3 aggregate result is not a safety claim).
- [x] (superseded, see top of file) Docker topologies beyond 4 clients and a second dataset for the SecAgg experiments.
- [x] (superseded, hosted CI runs) Hosted CI (needs an authorised repository). MNIST licence stays unresolved.
- [x] (superseded, hosted CI runs) Hosted CI on an authorized private repository.

## Immediate priority following v0.13.0

- Do not count Linux namespace fallback as Docker or Flower completion.
- Run `.github/workflows/flower-docker-native.yml` on a real Docker-enabled
  runner after the source is published to a dedicated private repository.
- Diagnose native SuperLink/SuperNode/SuperExec starts, per-round ClientApp
  participation, and strict `--require-deployment` checkpoint parity.
- Preserve every failure and do not weaken minimum-three-client guarantees.

# NEXT SPRINT — Native Docker/Flower certification execution (v0.12.0)

**P0. All items below are open until run on an actual Docker host.**

- [ ] Run `python scripts/run_flower_docker.py --prepare-only` on a clean Ubuntu/Python 3.12 runner. Verify three private public-digit partitions and matching digest manifest.
- [ ] Run `python scripts/run_flower_docker.py` with Docker Engine/Compose. Investigate and fix real errors. Capture `flower_cli_stdout.log`, `docker_services.log`, `docker_topology.json` and `client_attestations.json`.
- [ ] Inspect eight actual service containers and three distinct read-only `/site` mounts. No raw client dataset may be mounted into the SuperLink/ServerApp/CLI.
- [ ] Confirm three active ClientApps complete both rounds with matching per-site receipts and observed aggregated model checkpoint.
- [ ] Run `scripts/run_torch_reference.py` on EXACT same public partition root and `scripts/verify_flower_parity.py --require-deployment` with a matching config; fix any numerical mismatches without loosening tolerances prematurely.
- [ ] Run the workflow in `.github/workflows/flower-docker-native.yml`, retain actual Actions run ID and downloadable CI artifact. Existing repository has no dedicated remote; avoid claiming hosted checks ran.
- [ ] Repeat non-IID and intentionally stop one client to test minimum-three participation failure. Later run MNIST, formal DP and SecAgg+ independently.

**Verified this turn:** Python syntax/contract tests, source YAML topology, public-data staging, negative Docker CLI preflight and existing analytics regression. **Never confuse these with native deployment success.**

See `docs/v0.12-docker-flower-certification.md` for exact commands, instrumentation and known limitations.

---

# NEXT SPRINT — v0.11.0 native execution and thesis-quality gate

**Priority 0: these remain genuinely unverified. Do not mark them complete from source code, a static workflow or a dummy measurement.**

1. Create or authorize a dedicated `feddisclosure-lab` GitHub repository, set it as remote, and push a reviewable branch with the tracked source and CI workflows. No remote exists in this local release, so GitHub Actions have **not** run.
2. On a Python 3.12 Ubuntu runner with package-index access and Docker daemon, execute `python scripts/native_preflight.py --strict`, then `.github/workflows/native-v1-certification.yml`. This runs native Docker isolation/history, Flower FedAvg and tensor parity, OpenDP count releases and budget exhaustion, Opacus RDP-accounted DP-SGD, and upstream Flower SecAgg+ smoke. Collect actual workflow URLs and full logs.
3. Fix any native API incompatibility found in Flower 1.27, OpenDP, Opacus or the SecAgg+ legacy adapter without weakening the acceptance checks. Save negative and failed evidence.
4. Only after the native jobs pass, build matched protected-vs-unprotected attack/utility/communication evaluations, longer 28×28 MNIST learning, multiple seeds and uncertainty, a source-audited paper and a release report. OS-level adversary/evaluator isolation is still pending.
5. Redo the release percentage against the original mandatory criteria. A working Dockerless research lab is NOT a 100%-verified federation.

## Executed v0.11 milestones

- Persistent request-ID cache in SQLite for exact repeat randomized answers, transactional with budget accounting; mismatched query IDs/epsilons rejected, cross-restart/concurrent behavior tested.
- Three-site HTTP fault-injection benchmark with a deterministic NON-DP test mechanism: legacy exhausted .5 epsilon units per site over three attempts, retry-safe consumed .25 per site and replayed the same answer.
- Separate upstream Flower SecAgg+ integration and native CI workflows prepared, Flower runner timeout/stale-artifact checks hardened, parity verifier made safe under `python -O`, release evidence checker fails closed.
- Runtime preflight explicitly reports absent Docker, Flower, OpenDP and Opacus. Actual native certification BLOCKED.

## Commands from archive

```bash
PYTHONPATH=src:. pytest -q -m 'not slow'
PYTHONPATH=src:. python scripts/benchmark_retry_safety.py
python scripts/native_preflight.py --strict # EXPECT EXIT 2 without native runtimes
python scripts/verify_native_release.py # EXPECT EXIT 2 without real native evidence
PYTHONPATH=src:. python scripts/run_local_stack.py --once
```

See `docs/v0.11-runtime-and-retry.md`, `CURRENT_STATE.md`, `AGENTS.md` and the native workflow for exact verification criteria.

---

# NEXT SPRINT — After v0.10.0 (native release-certification blocker)

## Verified milestones this cycle

- [x] Demonstrated and fixed the experiment-ID reset bypass with durable `principal_history` mode and failure-safe policy locking. Three-client HTTP observations saved.
- [x] Executed 90-case, four-condition, matched statistical disclosure control matrix with independent verifier, SHA-256 and research figure.
- [x] Reused five IID + five non-IID separately trained models for 30 output-only attack evaluations with matched *query* cohorts; recorded weak negative results, saved hashes, uncertainty and limitations. Training cohorts are not identical.
- [x] Added optional Compose overlay/CI, research protocols and offline dashboard query control view and a 12-slide interim faculty briefing.

## P0 — still not closed

1. Execute `bash scripts/verify_compose.sh` **AND** `bash scripts/verify_compose_history.sh` under Docker Engine/Compose v2. Preserve native logs, volumes/isolation evidence, and actual result JSON. Overlay needs fresh named volumes.
2. Execute Flower 1.27 ServerApp/ClientApp three-client training on native supported Python host and strict tensor parity versus PyTorch reference. Require actual three clients/rounds, not source inspection.
3. Run native OpenDP count releases and Opacus DP-SGD with official privacy accountant. Compare matched defended/unprotected cases and compose costs appropriately. The fake measurement tests do NOT count. Record epsilon/delta/adjacency assumptions.
4. Implement and execute upstream Flower SecAgg+ with threshold/collusion/dropout tests. Never imply an arithmetic mean is cryptographic security.
5. Only after real runs close the above: launch CI on an authorized GitHub remote, store actual workflow URLs/run IDs, and perform complete clean-install reproducibility certification.

## P1 — scientific depth

- Full public MNIST (28×28) with license and source SHA-256, second independent dataset, five genuine training seeds, longer convergence studies and uncertainty. Existing runs use public sklearn 8×8 digits.
- Explain and repair IID versus non-IID training-set mismatch if estimating causal membership-risk effects; the current paired-query MIA study is descriptive only.
- Query-history defense under more complex overlapping allowlisted queries, adversarial budgets and legitimate-workload rejection measurements. Single-principal scope is not multi-tenant authorization.
- OS-level attacker/evaluator process identities and storage isolation; currently role separation is logical.
- Faculty-ready results audit and paper related-work/novelty review before claiming a submission-ready thesis.

## Verify from full archive

```bash
PYTHONPATH=src:. pytest -q -m 'not slow'
PYTHONPATH=src:. python scripts/run_history_scope_benchmark.py --output reports/results/sprint1/repeat_history_http.json
python scripts/verify_history_matrix.py
python scripts/verify_balanced_membership.py
python scripts/verify_v070_evidence.py
python scripts/verify_aggregate_collusion.py
python scripts/verify_joint_aggregate.py
python -m dashboard.build_dashboard
# Native host only, unverified here:
bash scripts/verify_compose_history.sh
```

---

# NEXT SPRINT — After v0.9.0

## P0: close native-runtime and defense gates

1. On a Python 3.12, network-enabled and Docker-enabled host, run `bash scripts/verify_compose.sh`. Preserve actual network/volume evidence. No local Docker was available for v0.9.
2. Run Flower ServerApp/ClientApp against three approved public-digit partitions and verify its tensors against the independent worker FedAvg reference (`scripts/verify_flower_parity.py`). Preserve logs and CI run ID, do not infer parity from a prepared workflow.
3. Execute native OpenDP count mechanism (not injected fake measurements), validate privacy map, adjacency/sensitivity, epsilon composition, budget exhaustion and actual statistical error across repeated defended queries.
4. Execute native Opacus DP-SGD with real RDP accountant over matched public datasets and model setups. Report achieved epsilon/delta, clipped gradient settings, accuracy and warning on repeated FL rounds. Never claim standard manually-added noise is DP.
5. Integrate upstream Flower SecAgg+ using at least three participants and a meaningful threshold. Test eligible collusion and dropout assumptions. No custom cryptography.

## P1: research-quality improvements

- Run the joint aggregate-only oracle attack with five prespecified seeds and consider a realistic **unknown contributor-label** method. Do not expose ground truth and do not treat evaluator matching as attacker identification.
- Extend multi-round FL on 28x28 MNIST, five seeds and meaningful confidence intervals. Replicate with a second approved public dataset; document source/checksum/license.
- Evaluate attack/utility tradeoffs matched within threat models; separately report individual-client, aggregate-only, and output-only observers.
- Introduce OS-level evaluator permissions; current folder distinction is a software boundary only.
- Expand dashboard from static results to predefined experiment launch controls only after authorization and resource isolation.

## Reproduce and maintain

```bash
PYTHONPATH=src:. pytest -q -m 'not slow'
python scripts/verify_joint_aggregate.py
python scripts/verify_aggregate_collusion.py
python scripts/verify_v070_evidence.py
python scripts/analyze_balanced_study.py
python -m dashboard.build_dashboard
python scripts/run_local_stack.py --once
```

After each native or research milestone, verify original artifacts, save failures and negative results, update `CURRENT_STATE.md`, `NEXT_SPRINT.md`, `docs/decisions.md`, and commit separately. Do not treat v0.9 dashboard or idealized aggregate reconstruction as completing Sprint 4 formal defenses.

---

# NEXT SPRINT — post v0.8.0

## Completed in this development cycle

- [x] Three-client, one-step full-batch exact-mean observation construction; unprotected individual, aggregate-only, and all-but-one collusion cases use explicitly declared privileges.
- [x] Five seeds × three roles = 15 actual reconstruction/evaluation runs against the same targets, with independent evaluation, SHA-256 hashes, strict metric replay, a figure and research-methodology report.
- [x] Native Opacus **integration code** (DP-SGD, RDP accountant, Poisson sampling, eps/delta target check, SGD reference, evaluator subprocess), CI workflow, CLI and validation tests. **NOT executed natively.**

## P0 blockers (must retain)

1. Run real Docker Compose and isolation verifier on a Docker Engine host; preserve logs and distinct institution-volume evidence.
2. Run native Flower ServerApp/ClientApp three-client training and weight parity. No same-process PyTorch reference can clear this gate.
3. Run native OpenDP three-site count Laplace mechanism with privacy-map proof, budget ledger and matched utility. No fake measurement qualifies.
4. Install real Opacus on Python 3.12, execute `python scripts/run_opacus_dp_sgd.py` against approved public digits. Check native Opacus RDP `get_epsilon(delta)`, privacy assumptions, achieved accuracy and comparison against SGD; persist actual artifacts and failures. Existing package-network DNS failure is environmental only.
5. Then integrate real Flower SecAgg+ using established upstream workflow with threshold >1 and documented dropout/collusion policy. **The v0.8 arithmetic mean is not SecAgg+.**

## P1 scientific next steps

- Strengthen aggregate-only attacks by optimizing the full joint 3-client observation, not only the naive target-only proxy. Add matched known-label and unknown-label scenarios, consider correlated client auxiliary information and participation changes. Preserve all failures.
- Isolate attackers/evaluators via OS-level access controls, not just folder naming.
- Compose per-site DP-SGD privacy across repeated local FL rounds. Preserve accurate participation, sampling and accountant state. No cross-round DP claims from the current one-shot runner.
- Longer five-seed equal-work 8–20-round FL study, convergence and worst-client uncertainty. Add a second approved public dataset with provenance.
- Demonstrate query defense versus real native OpenDP releases and release refusal rate; document output-only attacker query budgets.

## Reproduction and acceptance commands

```bash
pytest -q -m 'not slow'
python scripts/verify_v070_evidence.py
python scripts/verify_aggregate_collusion.py
python scripts/plot_aggregate_collusion.py
python scripts/run_local_stack.py --once
# Native verification on Python3.12 network-enabled host:
python -m pip install -e '.[privacy,research,test]'
python scripts/run_dp_analytics.py
python scripts/run_opacus_dp_sgd.py --data-root data/training/digits-iid --output-dir reports/results/sprint4/opacus_native
bash scripts/verify_compose.sh  # Docker host only
python scripts/run_flower_experiment.py # Flower host only
```

---

# NEXT SPRINT — post v0.7.0 gates

## Sprint 5 equal-work sensitivity: COMPLETED (short pilot)

- [x] Implemented deterministic class-conditioned capacity-constrained Dirichlet α=.1,.5,10, equal client sizes and isolated evaluator class histograms.
- [x] Executed 5 true matched training seeds × 4 partitions = 20 FedAvg four-round runs using independent private-data workers and separate evaluator, with identical 192/48 client train/val counts and 12 local minibatches per client per round.
- [x] Stored per-run checkpoint SHA-256, manifest hashes, all 20 scalar results, per-client accuracy and round curves, and one regenerated figure. All matching and integrity gates passed.
- [ ] **Research refinement:** longer 8–20-round five-seed study plus an independent confirmed replication, alternative public MNIST and controlled label-stratified validation. Current five-seed short pilot cannot establish convergence or significance.

## Sprint 4 DP count mechanism: IMPLEMENTED, NATIVE ACCEPTANCE OPEN

- [x] OpenDP-only count measurement adapter, ε map check, pure-DP count adjacency/sensitivity documented, budget ledger and fail-closed errors.
- [x] DP-only protected-site and gateway HTTP routes, raw-query and prior-audit bypass protections, transactional budget accounting, concurrent exhaustion tests, and gateway partial-failure tests. **Tests use injected fake measurements and do not establish DP**.
- [x] Created native runner, privacy Docker overlay and native OpenDP CI workflow, all currently UNEXECUTED.
- [ ] **P0 native:** On a supported Python 3.12 host with OpenDP, run actual measurement privacy-map and HTTP smoke. Save measured noisy releases, epsilon usage, MAE and budget exhaustion from real native noise; match against no-defense and history baselines. Do NOT substitute NumPy or a fake mechanism.
- [ ] **P0 Docker:** verify `compose.yaml` and `compose.privacy.yaml` on a clean Docker Compose host with clean volumes and `bash scripts/verify_compose.sh`; review protected-mode isolation.
- [ ] **P0 Flower:** install flwr[simulation], run 3 real clients, save observable native run and strict tensor-level parity against matched reference. Existing separate-process PyTorch work does not clear the Flower gate.
- [ ] Opacus DP-SGD on approved public digits/MNIST, real accountant epsilon/delta composition, matched utility against non-DP client training; test privacy budget and key assumptions.
- [ ] Flower SecAgg+ with participant threshold >1 and tested dropout/collusion assumptions (established Flower implementation ONLY, no custom crypto).
- [ ] Real container filesystem and attacker/evaluator OS permissions; named authorized roles, no evaluator truth shared with attacker.
- [ ] CI actual workflow artifacts and run IDs after the owner authorizes a remote; no GitHub push was performed.

## Next working loop

1. Reproduce `pytest -q -m 'not slow'` and `scripts/verify_v070_evidence.py` from the archive.
2. Run the actual OpenDP smoke on a supported host. If unavailable, preserve failure and implement the next measured utility/attack ablation without inventing epsilon claims.
3. Independently run the Flower native/parity and Docker gates, documenting exact runtime, version, logs, failures.
4. For every research run: predefine observer/config/metric, save raw allowed observation separately from evaluator truth, retain negative cases, verify hashes, regenerate plots, update `CURRENT_STATE.md` and this file.

---

# NEXT SPRINT — post-v0.6.0 research execution gates

1. **P0 native protocol:** Run real Docker Compose isolation and Flower 1.27 three-SuperNode simulation in Python 3.12 with successful logs. The Flower CI workflow now runs the independent-worker reference then `scripts/verify_flower_parity.py` with matched data manifest/config and final tensor tolerance. **Do not mark passed until actually executed.** Preserve logs, CI URL/run ID, and reproducible hashes.
2. **P1 matched heterogeneity ablation:** Design equal per-client sample-count and equal local optimizer-step controls for class-conditional Dirichlet alpha (e.g. 0.1, 0.5, 10) vs severe skew and IID. Run >=5 matched *training* seeds with global and worst-client accuracy, uncertainty, and accurate communication bytes. Current v0.6 Dirichlet differences are *confounded* by site sizes (133/244/200 vs 192/192/192 training records) and steps.
3. **P1 more realistic privacy:** Attack genuine client multi-epoch/multi-minibatch Flower updates once Flower runs, with unknown labels and explicit attacker observations. Current changing-minibatch pilot is distinct-record one-pass local SGD with *all labels disclosed* and no real Flower runtime. Explore stronger output-only attacks with query budgets on multiple independently trained models and characterize low-FPR uncertainty.
4. **P2 formal defenses:** Integrate established OpenDP and Opacus libraries, defining adjacency, accounting, clipping, epsilon/delta and composition; then Flower SecAgg+ with a legitimate participation threshold, dropout/collusion assumptions and actual native execution. No homemade cryptographic protocol, no claims for manually added noise.
5. **P2 scientific hygiene:** Add attacker/evaluator OS filesystem isolation, alternative public MNIST with verifiable provenance, additional datasets, configuration-specific reproducibility and saved negative experiments. Prepare faculty demo/report only after these controls are credible.

**Completed this cycle:** three disjoint-minibatch inversion attacks, nine membership evaluations on three model seeds, eight-round FedAvg/FedProx and Dirichlet pilot, parity-check tool/CI workflow, research figure generation, Claude/Codex skills, integrity verification and handoff documents.

**Reproduce:** use `docs/v0.6-research-report.md` and `docs/v0.6-experiment-protocol.md`, plus the previous sprint history below.

---

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
