## 2026-11-26 v0.16.13 server view measurement (see `CHANGELOG.md`, `docs/preregistration/v0.16-p2.md` and `docs/v0.16-p2-results.md`)

`flwr` 1.27.0, rows S4, S6 and S6b of the inventory, all three held. S4 (two Docker runs, 4 clients): the three metrics each client returned under SecAgg+ were in the reply the ServerApp received and in the `FitRes` given to `aggregate_fit` in 8 of 8 replies, with zero arrays in the array records of those replies. S6 (in process, 2,000 rounds per arm, assumed dependence of participation on the clipped norm): release bit accuracy 0.8425, participant set accuracy 0.8515, controls 0.5045 and 0.5035, and a control has 80% power for a participation difference of 0.062. S6b: effective noise multiplier 1.0001 with 10 results and 0.9001 with 9 (ratio 0.9000), exact epsilon 4.38 and 4.95 per round. Statements about channels the server observes, not proofs of leakage. Not measured: other releases, the Message API strategies, several rounds, timing and size, an active server, any inference from metric values.
Next: see `NEXT_SPRINT.md`.

## 2026-11-26 v0.16.12 documents refresh (see `CHANGELOG.md` and `docs/novelty-status.md`)

Stale statements in the novelty status, the README, the inventory header and two audit notes were corrected. No result changed. The prior-art gate for a paper stays partly open because release notes were not read.

## 2026-11-26 v0.16.11 TFF distributed DP aggregator measurement (see `CHANGELOG.md`, `docs/preregistration/v0.16-tff-ddp.md` and `docs/v0.16-tff-ddp-audit.md`)

`tensorflow-federated` 0.87.0, the real `DistributedDpSumFactory` (Skellam and discrete Gaussian, two rotations, modular clipping, adaptive clip), 2048 coordinates, test C++ execution context (the default one rejects `next`). 8 of 12 rows and 194 of 199 checks held. Three statistical checks held on replication. Two failed checks stand: the noise-free norm at 20 clients (9.97287 against 10) and row J, whose call did not run. Delivered noise over `noise_multiplier * l2_clip` is 1.00121, 0.99871, 1.00104 and 1.00196 at 10 clients and 16 bits. The documentation divides the local noise by `expected_clients_per_round` and the code by its square root (ratios 0.99641 and 1.00015 at 4 and 25 clients). With 5 and 20 clients against 10 expected the noise is 0.70807 and 1.40720 times the configured level. Sensitivity of a client above the clip is 1.00000, 1.00708 and 1.45129 times the clip at 16, 8 and 5 bits. The discrete Gaussian noise is 2.41687 and 1.67157 times the configured level at 4 and 5 bits. The state reports an `UnsupportedDpEvent` and no epsilon. A scratch run after the evidence saw 7 of 500 Skellam calls and 0 of 500 discrete Gaussian calls with an RMS above 1.06, consistent with two clients sharing a noise draw (cause not isolated, not in the evidence file). No privacy guarantee is claimed.
Next: see `NEXT_SPRINT.md`.

## 2026-11-26 v0.16.10 P3b results (see `CHANGELOG.md`, `docs/preregistration/v0.16-p3b.md` and `docs/v0.16-p3b-results.md`)

A second CIFAR-10 inversion study with `breaching` was pre-registered and run on a Colab T4 (20 evaluation seeds, 100 to 119). The control passed (+4.71 dB, interval +2.41 to +7.11, seed spread 1.69 times the assumed one so its power statement is void). At 4 records per client both plaintext gates failed, so that arm's collusion and aggregate rows are NOT INTERPRETABLE. At 2 records both plaintext gates passed (+2.79 dB, +0.294 SSIM). Behind them collusion kept the attack above the floor and was close to plaintext, and mixing cost the attacker. The 8 mixed image aggregate had a PSNR gain of +0.02 dB (A2P failed, labelled underpowered) and an SSIM gain of +0.110 (A2S failed). Fewer records helped the attacker (T1 passed). The SSIM rows are calibrated after P3 and are not confirmatory. The winner of the total variation tuning (0.2) was reused at 2 records and for the control without a test. These results describe the attacks that ran and say nothing about privacy. The P3 results in the 0.16.1 entry below are unchanged. Clauses that no script check covers are listed in the pre-registration.

## 2026-11-26 v0.16.9 TFF DP aggregator measurement (see `CHANGELOG.md`, `docs/preregistration/v0.16-tff-dp.md` and `docs/v0.16-tff-dp-audit.md`)

`tensorflow-federated` 0.87.0 (Python 3.11, `jaxlib` 0.4.14 from Google's release index), the real fixed and adaptive Gaussian `DifferentiallyPrivateFactory`, a `MeanFactory` around it and `dp_aggregator`. 15 of 16 rows and 126 of 127 checks held, and the failed check is an arithmetic slip in the pre-registration (weight sum 14 where 10 was sent). Delivered noise equals the configured noise multiplier times the clip over `clients_per_round` (ratios 1.0021, 1.0003, 0.9983), also when 5 or 20 clients report against 10 (0.9997 and 1.0012). The adaptive clip falls when more clients are below it, the noise uses the old clip (0.9988 and 0.9989 of its value, against 1.1003 and 1.1005 for the new clip), and the composed delivered multiplier is 0.9989 and 1.0049 of the configured one. The counted fraction is against the configured cohort size, a zeroed client counts as below the clip and the Python seed does not control the noise. The aggregators measured report a per-round `DpEvent` and no epsilon (no name containing epsilon in the six files scanned), and none was computed. The fast suite records were re-recorded after the branch was built (673 passed and 30 skipped with flwr and opacus, 478 passed and 70 skipped without). Not covered by any script check, read or judged by hand: the source and install statements, the five quotes (the tutorial text came from one page fetch), that 0.87.0 is the newest release on PyPI, the paper reading (so "as in the paper" for the clip direction is a hand read clause), that the TFF documentation read states no adjacency relation, that `MeanFactory` around the DP factory is not shown as a DP example in the files read, that nothing outside six scanned files reports an epsilon, the false failure estimate and the what-a-pass-covers paragraph, that the C++ worker runs the same kernels as the Python side, and the wording of the README, the document and the Flower comparison. They are also listed in the pre-registration, the results document and the changelog.
Next: see `NEXT_SPRINT.md`.

## 2026-11-26 v0.16.8 NVFlare events check (see `CHANGELOG.md`, `docs/preregistration/v0.16-nvflare-events.md` and `docs/v0.16-nvflare-events-audit.md`)

The real `SVTPrivacy` was called 1,000,000 times for each of six configurations: larger d and n (d 10 to 40, n 4 to 10), `replace` True, input values 0.2 and 0.8 gamma, the release values and a joint selection and release event. All 10 rows and 83 checks held. Lower bounds were 0.414, 0.206, 0.224 and 0.280 against claims of 1, 0.272 against 0.3 (release) and 0.502 against 1.2 (joint). The smallest excess that the check would find with 80% power was 4.9% to 8.3% of the claim, and none was found. The release event sits at 0.976 of its claim. Only 2.9.1 was run because the filter files are byte identical in 2.8.3, 2.9.0 and 2.9.1. The fast suite records were re-recorded after main was merged into the branch (663 passed and 28 skipped with flwr and opacus, 468 passed and 68 skipped without). Clauses that no script check covers stay untested or read by hand, as listed in the pre-registration: the chance of one block rejection among 12 worlds, the power of the block agreement test, the 992 second timing sum, the 1e-9 against 1e-12 ledger tolerance, and the eight items of its "Not covered" section.
Next: see `NEXT_SPRINT.md`.

## 2026-11-26 v0.16.7 release gate in CI (see `CHANGELOG.md`)

`.github/workflows/release-gate.yml` runs the release gate on every pull request. No result changed. A pull request that edits the gate without its pins file fails the integrity check on purpose, one that edits both passes it, and pushes to main are not gated.

## 2026-11-26 v0.16.6 adaptive clipping whole mechanism measurement (see `CHANGELOG.md`, `docs/preregistration/v0.16-adaptive-guarantee.md` and `docs/v0.16-adaptive-guarantee-audit.md`)

`flwr` 1.27.0 and 1.44.0, four adaptive and four fixed clipping variants over simulated clip trajectories, 640 rows per version, all held their predictions. The released count is a plain sum of bits of 0 or 1 (sensitivity 1, the paper's centred bit has magnitude 1/2 as read once). The model noise uses the clip after the update, so against the clip the clients used the model sum multiplier is the stored one times C(t+1) / C(t) (0.9054 with every update clipped at the defaults, 0.606 at clip learning rate 1.0). The composed multiplier with Flower's count is 0.8952 of the configured one in the default cell and 0.56949 to 0.99627 of it at a clip ratio of 1 for count noise 0.6 to 10. Fixed clipping classes have the configured multiplier. No epsilon or delta was computed and the intended sensitivity convention was not established.
Next: see `NEXT_SPRINT.md`.

## 2026-11-26 v0.16.5 NVFlare selection check with stated power (see `CHANGELOG.md`, `docs/preregistration/v0.16-nvflare-selection.md` and `docs/v0.16-nvflare-selection-audit.md`)

The v0.16.4 selection check could not fail at its claim (ceiling 10.9 against 20). The new check calls the real `SVTPrivacy` 1,000,000 times for each of three small configurations. Lower bounds on the epsilon of the selected set were 0.418, 0.357 and 0.015 against claims of 1, 1 and 0.1, and the smallest excess found with 80% power was 2.1%, 2.6% and 26% of the claim. All 7 rows and 44 checks held, one unscripted prediction held partly. The two filter files are byte identical in 2.8.3, 2.9.0 and 2.9.1 and a seeded call of each gave the same digest. The release gate now pins its own files and the commit hooks and checks that a pre-registration with predictions also states power.
Next: see `NEXT_SPRINT.md`.

## 2026-11-25 v0.16.4 NVFlare filter measurement (see `CHANGELOG.md`, `docs/preregistration/v0.16-nvflare.md` and `docs/v0.16-nvflare-dp-audit.md`)

`nvflare` 2.9.1 SVTPrivacy and PercentilePrivacy, 18 rows and 63 checks, all held. The default release budget makes one call epsilon 255.9 at 25,578 parameters. The acceptance step accepts a whole pass (656 of 1,000 for 100 wanted) and has no pass limit. No lower bound exceeded the claim. Release gate script and commit hooks added.
Next: see `NEXT_SPRINT.md`.

## 2026-11-25 v0.16.3 adaptive clipping measurement (see `CHANGELOG.md`, `docs/preregistration/v0.16-adaptive-clipping.md` and `docs/v0.16-adaptive-clipping-audit.md`)

`flwr` 1.27.0 and 1.44.0, four adaptive clipping variants, 82 rows per version, all held their predictions. The bit that feeds the clip update is set for clipped updates (the paper's is the reverse), so the clip moves away from the target quantile. The model noise is scaled with the clip after the round's update. No epsilon was measured.
Next: see `NEXT_SPRINT.md`.

## 2026-11-25 v0.16.2 FedML DP measurement (see `CHANGELOG.md`, `docs/preregistration/v0.16-fedml.md` and `docs/v0.16-fedml-dp-audit.md`)

`fedml` 0.9.6 DP classes called directly, 130 checks, 128 held. Gaussian and Laplace noise match their formulas within 0.5%. `sensitivity` is one argument for an L2 and an L1 mechanism, so a configuration that is right for the L2 mechanism gave epsilon 4.00 instead of 1.00 on the L1 mechanism for an update of norm 1 spread evenly over 16 coordinates. `LocalDP` does no clipping. Accountant paths raise errors in the configurations tried.
Next: see `NEXT_SPRINT.md`.

## 2026-11-25 v0.16.1 inversion attack on CIFAR-10 (see `CHANGELOG.md`, `docs/preregistration/v0.16-p3.md` and `docs/v0.16-p3-breaching-results.md`)

Pre-registered `breaching` study on real CIFAR-10, 20 seeds. Control passes (+5.50 dB over the label-only floor). With 8 records per client the attack does not beat the floor in PSNR in the plaintext view (-0.12 dB) or with n-1 collusion (-0.01 dB), so R1, R3 and O1 fail. For these attacks the aggregate view is 1.47 dB below the floor and its pass is vacuous. This is not evidence of privacy. The repository's own attack is 2.59 times worse in mean squared error (B1c fails).
Next: see `NEXT_SPRINT.md`.

## 2026-11-23 v0.16.0 first phase of the v0.16 plan (see `CHANGELOG.md`, `docs/preregistration/v0.16-p1.md` and `docs/v0.16-p1-flower-dp-audit.md`)

Measured privacy of Flower's DP mechanisms on `flwr` 1.27.0 and 1.44.0 with `scripts/audit_flower_dp.py`. Equal weights and full participation match the configured multiplier. Unequal weights, `num_sampled_clients` set higher than the number of aggregated results, and `LocalDpMod` at large epsilon or with `sensitivity` equal to `clipping_norm` do not. With adaptive clipping and all clients clipped the update noise is below the reference algorithm's. In 1.44.0 an optional accountant on the Message API raises an error on the first and releases nothing on the second. All predictions held except the mechanism named in C6.
Next: run the same harness on two more federated learning frameworks, then the stronger inversion attack on CIFAR-10. See `NEXT_SPRINT.md`.

## 2026-11-23 v0.15.0 attack-phase release (see `CHANGELOG.md`, `docs/preregistration/v0.15.md`, `docs/v0.15-*.md` and `docs/v0.15-technical-report.md`)

Eight work packages ran against pre-registered numeric rules, and no threshold was changed. Verdicts live in `reports/results/privacy/v0150/prereg_check_*.json`.
Pass: E1, E3, M1 (one-round SecAgg+ error proof and property tests), D1, D2, D4, D5 (DP bound against an informed attacker), W1 to W6 (wire-level validation, W1 vacuous, W3 pooled with low power, W5 counts only), F1 and F2 (fault matrix, F2 on the white-box audit only), B2 (LiRA against a reference), P1 (replay only) and P3 (the metric cannot fail), S1 (vacuous), S2 and S3 (8 clients).
Fail, kept as written: E2, D3, B1 (ratio 1.86, bootstrap 1.47 to 2.51, so every aggregate-view statement is conditional on the weaker attack), P2 and X1 (+9.4 points against +10).
What is deployed and what is not: the wire capture and the live parity runs used a real Flower 1.27.0 stack in Docker on hosted runners (4 clients, and parity at 3 and 4). Everything else is simulation with real components. The 8-client attack results are in-process, and the hosted 8-client rounds measured memory only (peak 3.25 GB of the 6 GB rule). The fault matrix injects into real components or copies of saved evidence and deployed no faulty stack. The Fashion-MNIST hosted workflow ran once after the release as a second observation (collusion +12.5 points, would pass, X1 stays FAIL on the local run).
Reply order: Flower's `FedAvg` sums float32 updates in reply order, which was already discussed publicly. The two native runs that fail the float64 gate fail for every order, so their verdict does not depend on the order. `DeterministicFedAvg` is an opt-in workaround, and with it on the float64 parity gate no longer tests Flower's aggregation.
Reproducibility: `docs/reproducibility-v0.15.md` says which torch produced which evidence and what review R4 reproduced. The fast suite count depends on the environment, so the README gives one count per environment, recorded by `scripts/record_fast_suite.py` with the commit, and `scripts/verify_readme_claims.py` checks them, the v0.15 numbers and the PASS or FAIL words of the README.
Open items are in `NEXT_SPRINT.md` (native 8-client capture, informed attacker with a public prior, stronger inversion on protected views, live MNIST reply-order configs, a full-environment CI job, X1 tuning on the recognisable rate and more seeds, the E2 float32 term).

## 2026-11-23 v0.14.3 corrections to the update-inversion reading (see `CHANGELOG.md` and `docs/v0.14-privacy-secagg-results.md`)

No attack was rerun and no saved v0.14.2 evidence changed. The v0.14.2 claims that the SecAgg+ aggregate "still leaves 55 to 59% of
victim images recognisable" and that DP "stopped this attack" are withdrawn. The attacker knows the labels, so the baseline is the
label-only floor (class-average image of a 1765-image labelled public pool): MSE 0.046 and 67.5% recognisable on seeds 0 to 9 (54 of 80 images, Wilson 57% to 77%).
Plaintext (94%) and n-1 collusion (91%) stay 26 and 24 points above it. The aggregate attack with max weight 1000 (55%) does not beat it.
With max weight 16 (58.8%) it cannot be told apart from it. The floor uses public images and the attack uses none, so a
prior-initialised attacker could do better, and the floor depends on the pool size (31% at 20 images, 57% at 100, 67.5% at 1765).
The DP rows only show that a naive attacker fails. New files: `reports/results/privacy/v0143/` (`label_floor.json`, `view_intervals.json`,
`prereg_check_corrections.json`, `fast_suite.txt`). `python scripts/verify_readme_claims.py` checks the README numbers and verdict words against them.
Still open: DP-aware inversion attacker, prior-initialised aggregate attacker, inversion on captured multi-step Docker traffic.
Later note (v0.15): the 67.5% floor and the 24 to 26 point gaps are for seeds 0 to 9. On 30 seeds the floor is 75.0% (seeds 10 to 29 are easier) and the gaps are plaintext +20.0, collusion +15.4 at 4 clients and +16.3 at 8, aggregate only -28.8 and -39.6, see `docs/v0.15-scaling.md`.

## 2026-11-22 v0.14 remaining items executed (see `docs/v0.14-privacy-secagg-results.md` sections 1, 2, 6, 7)

Executed in the cloud container since the completion run: protected versus unprotected update-inversion
comparison (plaintext, SecAgg+ aggregate with upstream quantiser arithmetic, n-1 collusion, DP views, 10 seeds),
a Monte Carlo acceptance rule for multi-round SecAgg+ error (300 runs), a float32 replay parity gate (7 of 7 native
runs pass, the float64 1e-5 gate still passes 5 of 7), LiRA at 64 shadows and 16 targets on Digits and on MNIST,
the MNIST file provenance check, and a static workflow check.
Results that matter (the first and third readings were corrected in v0.14.3, see the next entry): plaintext and n-1
collusion inversion stay 24 to 26 points above the label-only floor, the aggregate-only SecAgg+ attack does not beat that floor,
and the DP rows are a naive attacker only. LiRA at epsilon
6 to 8 still shows a detectable signal (AUC 0.54 to 0.58), and the earlier 24-shadow run understated it.
Kept as FAIL: float64 parity on label skew and seed 3, the naive 2-round tolerance on the original 4-client run.
Still open: MNIST licence unresolved (no explicit licence from the originators), no multi-round SecAgg+ bound for the non-convex model (an L2 bound for convex losses is in docs/proofs/multi-round-note.md),
no DP-aware inversion attacker, no inversion on captured Docker traffic, Docker topology tested at 3 and 4 clients only.

## 2026-11-21 v0.14 Phase 2 and 3 executed (see `docs/v0.14-privacy-secagg-results.md`)

Executed with real packages: Flower SecAgg+ (upstream workflow and mod) inside the Docker
topology with a matched plaintext control, OpenDP 0.16 Laplace releases with a persistent
budget ledger, and Opacus DP-FedAvg with achieved epsilon and delta over 10 seeds (Digits)
plus a 3 seed MNIST study. A full 60,000 image MNIST run completed natively on Docker +
Flower (10 rounds, parity True, 3 x 10 receipts, accuracy 0.9568).
Known negatives preserved: SecAgg+ gives nothing against n-1 colluders, n=3 t=2 halts on
dropout, DP did not lower measured membership AUC (attack is near chance), DP hurts the
worst client badly under label skew.
Still open: hosted CI (no authorised remote), fail-fast fault handling, SecAgg+ n>=4 dropout
experiment, native non-IID MNIST run, stronger attacks, MNIST licence text unverified.

## 2026-11-21 v0.14 native Docker + Flower EXECUTED (Phase 1 PASS at smoke scale)

Real Docker Engine 29.10 / Compose containers ran Flower 1.27.0 Deployment Runtime:
SuperLink, ServerApp executor, three SuperNodes, three ClientApp executors, private
network, one read-only partition mount per institution. Two FedAvg rounds with receipts
from all three clients, native `final_model.pt`, strict `--require-deployment` parity
with the independent PyTorch reference (max diff 2.98e-8), a byte-identical fresh repeat,
a severe label-skew run, and three fault scenarios that the gate correctly refused.
See `docs/v0.14-native-execution.md` and `reports/results/native_flower/v014/`.
Still NOT executed at this point: OpenDP/Opacus/SecAgg+ privacy-defense experiments, full
MNIST, multi-seed studies, hosted CI.

## 2026-11-20 v0.13.0 execution milestone (Linux namespaces; **NOT Docker/Flower**)

On this cloud container, three OS-namespace-isolated PyTorch workers executed two
FedAvg rounds on disjoint public digit partitions with each site's dataset
read-only and other staged site paths masked. Round accuracy 15.00%, then
20.83%; independent-worker PyTorch checkpoint parity max tensor diff 1.49e-8.
Six namespace receipts and an evidence validator passed. See
`docs/v0.13-cloud-execution.md`. Native Docker/Flower attempt exited 2 because
Docker CLI is absent. No native Flower checkpoint was produced, and the native
acceptance gate **remains open**.

# CURRENT STATE — v0.12.0 native Flower Docker implementation (2026-11-20)

## Implemented this cycle (verified static/application only)

- Added `compose.flower.yaml`: pinned Flower 1.27.0 SuperLink, three real SuperNodes, one separate ServerApp executor, three ClientApp executors and optional nonprivileged CLI container. It explicitly uses the **Deployment Runtime** profile `fdl-docker`, not Flower local simulation.
- Added a reproducible CPU SuperExec Dockerfile without any raw client rows, model observations or evaluator truth in the image build context.
- Added `scripts/prepare_flower_deployment.py`: generates 3 disjoint public-digit client partitions with per-site SHA256 digests, read-only mount paths and independent attestation folders; checks manifest/record integrity and refuses mismatched source data.
- Added `scripts/verify_flower_topology.py`: static Compose isolation contract and optional actual Docker `inspect` verification. Static pass is **not** container certification.
- Added per-site ClientApp identity checks (`FDL_SITE_ID` must agree with `node-config partition-id`), digest checks and privacy-safe training/evaluation receipts. Added `scripts/verify_flower_attestations.py` to require actual receipts from all 3 clients and every training round.
- Added `scripts/run_flower_docker.py`, `scripts/verify_flower_docker.sh`, strict deployment-only parity guard, and GitHub Actions native 3-client workflow. Local staging and static tests execute successfully.
- Local fast tests **122 passed / 2 skipped**, two separate slow training integration tests passed. Fresh staged-Docker public-digit PyTorch reference: 2 rounds, accuracy 0.2083333333, final checkpoint SHA256 `5ae8fe93cc5f5dd73b32016f60016c1ba445aba5cbb31005d9ff96397e391fff`. The live HTTP analytics regression also passed. Detailed evidence: `docs/v0.12-results.md`.
- **CRITICAL: native Docker/Flower remains UNVERIFIED.** This runtime has no Docker CLI/daemon or installed Flower; a native invocation returned exit code 2 with `Docker CLI missing`. No model was trained inside a Docker container in this cycle.
- Details: `docs/v0.12-docker-flower-certification.md` and `reports/results/native_flower/v012/` (local evidence bundle only).

## Next acceptance gates

1. Run `.github/workflows/flower-docker-native.yml` in an authorized dedicated repository, or run `python scripts/run_flower_docker.py` on a Docker-capable Python 3.12 Ubuntu host.
2. Fix any actual Docker/SuperExec/Flower runtime incompatibilities and rerun, collecting logs, container identities, mounts, participation receipts, real server checkpoint and rounds.
3. Run identical independent-worker PyTorch reference and `scripts/verify_flower_parity.py --require-deployment`; accept only if all tensors match and both native evidence files exist.
4. Then run a non-IID repeat, client dropout/participation threshold tests and full MNIST; keep formal privacy defenses as separate gates.

---

# v0.11.0 — Verified retry safety, native certification preparation (2026-11-20)

- **Measured:** Three-site fault-injection HTTP benchmark, persistent idempotent DP query retry and conflict rejection. `reports/results/sprint4/retry_benchmark.json` uses a deterministic TEST mechanism. It offers NO differential-privacy guarantee.
- **Implemented but NOT executed natively:** Upstream Flower SecAgg+ app, strict native CI gate, hardened Flower evidence runner and parity checker. Docker, Flower, OpenDP and Opacus are absent here.
- **Verified:** 115 passing fast tests after the final changes, two independent-worker training integration passes, live analytics benchmark, historical evidence verification. See `docs/v0.11-runtime-and-retry.md` and the latest saved verification logs for final counts.
- **Action required to run hosted native CI:** Configure a dedicated GitHub remote for this repository (none is present) and run `.github/workflows/native-v1-certification.yml` on GitHub's Ubuntu runner. Never claim 100% or formal DP/SecAgg+ before actual native results.

# CURRENT STATE — 2026-11-20, v0.10.0

## Completed and executed since v0.9

- **Attack + defense runtime:** Three separate local HTTP institution processes, one gateway, fresh per-site SQLite. Reproduced caller-controlled experiment-ID bypass against the old history control: 110 − 107 = 3. New `principal_history` mode denied the inference when experiment IDs differed: HTTP [200, 409, 200, 403] including an intentional policy-downgrade request; harmless repetition remained permitted. New persistent site release-policy lock refuses restart without the same enforcement, and refuses enabling it over sites that previously issued unprotected responses. SQLite transaction tests cover concurrent conflicting queries.
- **Executed release-control matrix:** 90 matched synthetic populations over 15 seeds, 3 minimum count thresholds and 2 prevalence bands, 4 control conditions and 3 sequential queries: 1,080 local policy-engine evaluations. Exact aggregate inference succeeded 90/90 without history, 14/90 with same-ID history, 90/90 with experiment-ID reset against old history, 14/90 with principal-wide history despite the reset. This is a restricted two-query benchmark, NOT differential privacy or statistical population risk. Artifact SHA-256 and independent verifier pass.
- **Output-only membership inference:** Existing 10 independently trained PyTorch FedAvg checkpoints from five IID/severe-skew seed pairs, 30 attack evaluations, 120 matched query records/model, three heuristics. Each pair sees identical shared member and held-out nonmember query records. Actual training record sets differ (455–466 overlap), so the contrast is descriptive only. Mean AUC max confidence IID 0.4935/skew 0.4876; negative entropy 0.5011/0.5043; margin 0.4949/0.4894. Per-sample labels never enter attack inputs. All 30 evaluator scores and SHA-256 replay successfully.
- **Engineering:** optional protected Compose overlay, native Compose verifier and CI workflow (source/static only); offline evidence explorer updated with new history matrix; 12-slide interim faculty briefing, results protocol, figures, regression/role-separation tests, Claude and Codex handoff updated.

## Native gate honesty

Docker Engine/Compose is absent. Flower, OpenDP, and Opacus are not installed, and package-index DNS requests fail. New history Compose verifier exits 2 in this environment: **NOT VERIFIED**. Upstream Flower SecAgg+ is not implemented/executed. Static Compose and application tests cannot verify native network/volume boundaries. Full 28×28 MNIST remains not run. No novel research/paper or DP certification claim.

## Next

See `NEXT_SPRINT.md` for native privacy and federated execution gates. For new results, start with `docs/v0.10-results.md` and verify `scripts/verify_history_matrix.py` and `scripts/verify_balanced_membership.py` before changing claims.

---

# CURRENT STATE — 2026-11-19, v0.9.0

## Verified research deliverables in this cycle

- New **joint all-label-oracle aggregate reconstruction**: 3 executed seeds (241–243), 160 attack iterations each, 3 equally weighted one-step SGD institutional updates; 24 pooled public-digit examples per seed. Results were scored by a separate evaluator using label-constrained image matching. Pooled MSE per seed: 0.0592816, 0.0558440, 0.0568235. Target-oracle MSE: 0.0478832, 0.0501742, 0.0330903. Saved observations/recoveries are integrity-verified with SHA-256 in `reports/results/sprint4/joint_aggregate_oracle/`.
- **Additional oracle privilege:** All 24 class labels are supplied in client-grouped order. This reveals contributor label grouping; claims of anonymous per-record site assignment are inappropriate. Historical three-seed artifacts retain their original report text; this correction is documented here and in the v0.9 report. The attacker does NOT receive pixels or individual updates. This is NOT Flower SecAgg+.
- Exploratory **paired uncertainty report** over existing 5-seed equal-work training. IID vs Dirichlet α=0.1 mean difference: global +2.33 pp (95% t CI −1.73, +6.40), worst client −11.67 pp (95% t CI −36.38, +13.05), exact sign-flip p = 0.375 / 0.4375. No statistically reliable effect claim.
- **Faculty discussion brief and research-direction shortlist** prepared with explicit novelty and acceptance caveats. See `docs/faculty-briefing.md` and `docs/thesis-directions.md`.
- **Interactive offline dashboard** generated and tested: `dashboard/dist/index.html` and snapshot JSON; curated scalar metrics only, no raw/evaluator images. Optional Streamlit UI source exists but native Streamlit isn't installed, so only the dependency-free static exporter is executed.
- **Fast suite: 88 passed, 2 skipped, 3 slow deselected.** Optional Flower/OpenDP skips are native-dependency blockers. See `reports/results/v090_verification/fast_tests.log`.

## Native gates NOT VERIFIED

- Docker CLI/Engine absent. No Compose deployment or OS-level per-client filesystem isolation.
- Flower absent and network DNS blocked. No native three-node Flower parity result.
- OpenDP/Opacus absent and package installs blocked. Application-level DP ledger tests are NOT formal OpenDP execution or ε/δ evidence. SecAgg+ remains unexecuted.
- GitHub workflows are source only, not run; no remote push or run IDs are claimed.

## Methodology limitations

- Target-oracle matching is an **evaluator-only optimistic metric**; it cannot identify which recovered example belongs to target site from an anonymous aggregate.
- Joint label oracle has substantially more attacker information than the v0.8 target-only proxy, so its MSE is not a like-for-like estimator comparison.
- Five-seed training confidence intervals are exploratory and underpowered; the independent processes are not container-isolated.
- The combined slow integration test was interrupted in this runtime; do not count a successful combined slow test. Historical v0.8 slow tests remain verified in the previous release.

## Next

Read `NEXT_SPRINT.md`. Primary blockers: native Docker/Flower/OpenDP/Opacus and SecAgg+ gates, protected-vs-unprotected matched DP comparison, validated stronger attacker observation models, full MNIST, multi-seed longer training, and publication-style report.

---

# CURRENT STATE — 2026-11-19 — v0.8.0

**Research milestone 1 (EXECUTED):** Five-seed, three-site idealized model-update aggregation/collusion study with 15 attacker/evaluator reconstructions. Same 8 target public digit images and known target labels across individual, aggregate-only target probe and all-but-one collusion roles, 160 Adam iterations per attack. Average target image MSE **0.01664135 individual**, **0.10847777 aggregate-only**, **0.01664135 collusion**. The collusion role algebraically isolates the target update with tensor max-error <2e-6; aggregate-only is a deliberately misspecified target-only attack, not a best possible attack. No native SecAgg+ or real cryptographic protection was used. See `docs/v0.8-research-protocol.md`, `docs/v0.8-results.md` and `scripts/verify_aggregate_collusion.py`. Source images, observations, scores and summary shipped in archive but private research tensors NOT committed.

**Research milestone 2 (IMPLEMENTED, NATIVE BLOCKED):** Native Opacus adapter `defenses/opacus_training.py`, single-silo SGD versus DP-SGD runner `scripts/run_opacus_dp_sgd.py`, predeclared defense YAML, validation and native integration tests, and CI workflow `opacus-native.yml`. The runner checks actual Opacus accountant epsilon and private optimizer noise multiplier and refuses to claim DP if missing. This environment has no `opacus`, so the CLI exited status 2 and wrote no benchmark. No formal DP-SGD guarantee or private utility comparison is claimed. OpenDP native measurement, Flower, Docker and SecAgg+ remain unverified and blocked.

**Verification:** 80 fast tests passed, 2 optional skips (Flower and native OpenDP), 3 slow deselected (2 independent worker cases passed separately and native Opacus case skipped); 15 attack scores and observations independently verified; live four-process analytics reproduced 110−107=3 and history rejection; v0.7 evidence integrity reverified. Opacus and OpenDP runners exited 2, Docker absent (127), no native results written. v0.7 20-run balanced training study and recorded historical evidence preserved. New figure `reports/figures/sprint4_aggregate_collusion.png` is generated from saved runs. Native blockers are not accepted as passes. See `NEXT_SPRINT.md` for gates.

---

# CURRENT STATE — 2026-11-19 — v0.7.0

**Two workstreams advanced:** (1) Sprint 5 matched heterogeneity ablation completed on three independent PyTorch client processes with 20 saved runs; (2) Sprint 4 OpenDP query-release application transport and principal-wide budget ledger implemented and regression-tested using an injected *fake measurement* (NOT actual DP). Formal Sprint 4 acceptance remains BLOCKED because OpenDP is unavailable.

**Executed training evidence:** Five seeds 241–245, four partitions IID, capacity-constrained Dirichlet α=0.1,0.5,10, four FedAvg rounds, 900 public sklearn digit examples, 180 global held-out examples. Every client has 192 train + 48 validation records, identical SGD minibatch budgets and 231,360 estimated raw model bytes per run. Global mean accuracy IID 68.89%, α=.1 71.22%, α=.5 67.67%, α=10 71.56%; mean worst-client accuracy 65.00%, 53.33%, 60.00%, 65.42%. The apparent global gain for severe α=.1 does not imply good client equity. The preceding unbalanced v0.6 comparison remains confounded and is not superseded by a claim of causal superiority. All 20 checkpoint hashes and site-data hashes, cohort/size contracts, scalar summaries and the new figure verified with `scripts/verify_v070_evidence.py`.

**DP application path:** `defenses/opendp_count.py` constructs an OpenDP scalar Laplace measurement and checks its privacy map at add/remove Δ=1; `defenses/dp_release.py` transactionally enforces an experiment-ID-independent SQLite budget. `FDL_DP_ENABLED=1` enables a separate gateway and site count route, blocks raw routes, prevents reuse of databases with previous unprotected query activity, and rejects downgrading after spending DP budget. Tests use a deterministic fake to validate app/ledger only; no real OpenDP release, formal epsilon guarantee, or runtime utility benchmark is claimed. Native runner exits status 2 without OpenDP rather than inventing results.

**Observed verification:** 76 fast tests passed, 2 optional runtime tests skipped (Flower, native OpenDP), and 2 slow tests deselected. Each of the two independent-worker slow integration tests passed separately (18.82 s and 16.12 s). Four-process analytics passed 110−107=3 and history rejection. Historical and new evidence validators passed; all 20 checkpoints and data manifest hashes verified. A repeat IID seed241 run exactly matched all four test accuracy rounds and the final SHA-256 checkpoint. The combined full-suite invocation stalled and hit a tool timeout, so is NOT counted as a pass. Docker verifier exit 2, native OpenDP runner exit 2, Flower runner exit 1 (missing packages). See `reports/results/v070_verification/`. Docker Engine absent; Flower absent; OpenDP absent; Opacus/SecAgg+ not executed. No GitHub remote/Action run is claimed. Git release v0.7.0 once tagged.

**Continue:** Read `NEXT_SPRINT.md`, `docs/v0.7-pre-registered-protocol.md`, `docs/opendp-runtime.md`. On Python3.12 with network run `pip install -e '.[privacy,test]'`, `python scripts/run_dp_analytics.py`; on Docker host run `bash scripts/verify_compose.sh`; on Flower host run native Flower + strict parity. Finish formal DP runtime gate before any epsilon claim.

---

# CURRENT STATE — 2026-11-19 — v0.6.0

**Actual completed milestone:** 3 changing-minibatch individual client model-delta attack/evaluator runs, 3 separately trained FedAvg models evaluated against 3 predeclared output-only membership attacks each (9 runs), 2 matched 8-round severe non-IID FedAvg/FedProx runs, 1 8-round Dirichlet(alpha=0.5) FedAvg experiment, deterministic Dirichlet partition tests and a native Flower checkpoint-parity *validator*. See `docs/v0.6-research-report.md`, `docs/v0.6-experiment-protocol.md`, and `reports/results/`.

**Evidence:** changing-minibatch MSEs 0.0149083, 0.0756175 and 0.0794448; independent-model MIA mean ROC-AUC 0.542/0.544/0.520; 8-round severe skew FedAvg 0.63333 global and 0.0625 worst-client; matched FedProx(mu=.3) 0.60556 and 0.0; Dirichlet(alpha=.5) FedAvg 0.82778 and 0.79592 worst-client. Dirichlet has unequal site sizes and optimizer-step counts; do not interpret as a causal benefit of partition skew alone. All are public 8x8 digit, small pilot studies; no DP claim. The eight-round severe runs exactly reproduce their earlier four-round accuracy trajectories.

**Verification this cycle:** 55 fast tests passed, 1 Flower skip, 2 slow tests deselected; two independent-worker slow tests passed separately. Live four-process synthetic analytics repeated 110 - 107 = 3, with the history defense refusing the second query. Legacy and v0.6 evidence verifiers passed. Three current research figures regenerated.

**Native Flower and Docker: NOT VERIFIED.** This runtime still lacks the Docker daemon and the Flower package/network access. `scripts/verify_flower_parity.py` has only passed *fixture contract tests*, which are not real Flower runs. Do not claim actual Flower parity. OpenDP/Opacus/SecAgg+ are NOT IMPLEMENTED/VERIFIED.

**Handoff:** Run `pytest -q -m 'not slow'`, separately test the two slow integration cases, `python scripts/run_local_stack.py --once`, `python scripts/verify_research_evidence.py`, and `python scripts/verify_v060_evidence.py` on the complete archive. For the native protocols, run `bash scripts/verify_compose.sh`, `python scripts/run_flower_experiment.py` and `python scripts/verify_flower_parity.py` on a capable host. Continue from `NEXT_SPRINT.md`.

---

# CURRENT STATE — 2026-11-19 — v0.5.0

**Status:** Executed Sprint 0/1 analytics, PyTorch multi-process FedAvg/FedProx (NOT Flower), and controlled gradient/model-update disclosure. Native Docker, native Flower, OpenDP, Opacus and SecAgg+ are **NOT VERIFIED**. Prior evidence remains below.

## This milestone: measured v0.5.0 progress

- Actual unprotected *individual-client model deltas* from fixed batches of 1, 8 or 32 distinct public 8x8 images, with one/four full-batch SGD steps. Attacker receives only initial model, delta, optimizer metadata, and privileged labels for batches larger than one. It has no image truth. Five single-seed attack/evaluator cases were executed at 160 iterations. MSE: 1:1 9.59e-9; 8:1 0.00740; 8:4 0.04949; 32:1 0.06339; 32:4 0.07000. Fixed batches were reused across steps; not real changing minibatches or native Flower. See `reports/results/sprint3/multirecord_update/` and `docs/disclosure-protocol.md`.
- FedProx added to the shared PyTorch training code and wired to future Flower ClientApp train configuration. `mu=0` exactly reproduced FedAvg updates; `mu=3` reduced client parameter drift in unit tests. **Three matched severe non-IID seeds** 241, 242, 243 executed for FedAvg `mu=0` versus FedProx `mu=0.3`, with three independent clients, four rounds and two local epochs. Mean held-out global accuracy 55.19% versus 49.07%; paired change −6.11 percentage points (SD 3.47 pp). Final worst-site global-model validation accuracy remained zero in all six runs. Both methods transmitted an estimated 231,360 raw tensor bytes each run. No general algorithm superiority or fairness claim; undertrained tiny public-data models.
- Revalidated first research benchmark with four separate live HTTP processes: 110 - 107 = 3 inferred without history protection; history-aware query blocked. All tests and figures recorded as research evidence, not production privacy.
- Verification: **44 passed, 1 optional Flower skip, 2 slow deselected** fast tests. Each of two independent-worker slow integration tests **passed separately**. Read-only evidence verification passed, including hashes of all six paired training checkpoints, five multi-record cases, older baseline evidence, and regeneration of two new figures. The combined slow+fast invocation was not claimed successful.
- Environment limitations: native `docker` absent; `flwr` package absent and package-network DNS unavailable. Fresh extraction required an editable install; `python -m pip install -e . --no-build-isolation --no-deps` worked with preinstalled dependencies. An initial longer 8-round experiment timed out and generated no complete evidence; replaced with successful matched four-round runs. Do not count timed-out work.

**Next:** native Docker/Flower runtime gate on a suitable host; longer MNIST and IID/non-IID training with supported distributed orchestration; real changing-minibatch update inversion and independent attacker OS sandbox; matched defended DP/SecAgg+ comparisons using established libraries. See `NEXT_SPRINT.md`.

---

# Prior state — v0.4.1

**Version:** 0.4.1. **Working baseline:** Sprint 0, Sprint 1 analytics, offline Sprint 2 FedAvg reference and first Sprint 3 model-disclosure experiments. **NOT a completed native Flower/Docker/DP release.**

## Executed here: analytical disclosure (Sprint 1)

Three separate FastAPI institution processes, three private SQLite databases and one HTTP gateway execute the allowlisted query benchmark. On 2026-11-19 the live local stack repeated the naive `110 - 107 = 3` restricted-group inference; the history-aware policy denied the second query. Result: `reports/results/sprint1_local_v0.4_validation.json`. Static Compose safety contract also passes. The native Compose runtime is **NOT VERIFIED**: `docker` binary is absent; actual verifier returned exit 2. See `reports/results/runtime_gates/status.json` and logged stderr.

## Executed here: PyTorch federated-training reference (Sprint 2)

- The CPU reference uses separate, concurrent OS processes for institutional clients and an independent evaluator, private public-digit NPZ partitions, immutable SHA-256 checks, FedAvg weighted model aggregation, and final global per-client validation. This is **NOT Flower** and its processes do not enforce filesystem isolation.
- Ten paired reference runs: 5 seeds each for IID and severe label-skew partitions, 3 clients, 600 selected public 8×8 digit images, 2 rounds. Mean public test accuracy **0.2000 ± 0.06095 SD** IID and **0.17833 ± 0.05420 SD** non-IID. Mean worst-client global-model validation **0.11875** IID and **0.0000** severe non-IID. Short, weak convergence smoke results, not a general effect or statistical significance claim.
- One-seed client-count pilot: 1 client IID public accuracy 0.3833; 5 clients IID 0.1750, severe non-IID 0.1833. **The different client counts do not have matched optimizer steps; do not compare these as fair efficiency or privacy effects.** Two initial timed-out five-client attempts were resolved by limiting simultaneous worker processes to 3. One institution cannot provide secure aggregation concealment.
- Schema-v2 manifest now removes coordinator-readable `label_histogram` fields, saving distribution statistics in evaluator-only files. New training refuses legacy v1 manifests. Flower ClientApp additionally validates client partition hashes. A tampered client-partition test passes (fails closed). `scripts/run_training_sweep.py` can resume previously saved run JSONs without losing matching-condition summaries.
- Evidence: `reports/results/sprint2/sweep/sweep_summary.json`, ablations under `reports/results/sprint2/`, figure `reports/figures/sprint2_five_seed_accuracy.png`; full public-digit checkpoints and hashes in the release archive.
- Native Flower ServerApp/ClientApp source and CI exist, but `flwr` is absent and the Python package network is unavailable in this environment. `scripts/run_flower_experiment.py` exited 1 with a documented dependency error; **NO native Flower run or one-round Flower aggregation cross-check has occurred**.

## Executed here: controlled model disclosure (Sprint 3)

**Gradient inversion.** Implemented exact-gradient observation construction from approved public data in a privileged setup role, an independent DLG-style Adam optimization attack and an independent image evaluator. Batch 1 uses inferred label (iDLG-style output-layer bias rule), 450 steps; batch 8 uses explicitly known labels, 400 steps. Five matched single-example exact-vs-Gaussian-noise seed pairs executed. Exact mean MSE **2.42678e-9**; manually added noise 0.01 mean MSE **0.00121135**. The single-seed batch-8 reconstruction MSE was 0.010676. **No privacy guarantee or DP claim applies to the noise.** This measures highly privileged exact first-step gradients, not multi-step FL updates or SecAgg+ observations.

**Output-only membership inference.** On the frozen two-round public-digit reference checkpoint, the attacker saw only class confidence vectors, scoring maximum confidence; the evaluator alone saw balanced class-matched membership truth. The first pilot ROC-AUC was 0.5475. Across five **query-subset** seeds (one fixed model), mean ROC-AUC **0.47811 ± 0.01832 SD**. This is near-chance/negative evidence, not proof of model privacy, and not five independently trained models.

**Actual local-update inversion (additional measured loop):** The simulated unprotected coordinator receives an initial model and an actual SGD client parameter delta, *not* an exact gradient. When the attacker divides that delta by the known learning rate and number of local steps, one-step inversion MSE was **2.02613e-9**, four-step MSE **0.0271682**, sixteen-step MSE **0.140351**; the proxy-gradient relative L2 error increased from **1.55e-7** to **0.4757** and **0.7569**. Same public digit and initialization across cases, one record reused for local SGD (a deliberately idealized tiny client). The label was inferred in all three cases. This is not a realistic multi-record Flower update, and does not involve secure aggregation. Full `reports/results/sprint3/model_update_inversion/` evidence and a generated figure are included.

- Evidence: `reports/results/sprint3/gradient_b1/`, `gradient_b1_noisy/`, `gradient_b8_known_labels/`, `matrix/matrix_summary.json`, `membership_smoke/`, `membership_matrix/membership_matrix_summary.json`. `scripts/plot_disclosure.py` regenerated three figures from original measured artifacts.
- Attack CLI arguments cannot include original public images or membership truth. These live under `evaluator_only/` and are passed only to the evaluation subprocess. This is a code/interface boundary, NOT an OS security boundary.

## Test and release evidence

- Executed `pytest -q -m 'not slow'`: **40 passed, 1 optional Flower import skipped, 2 slow deselected**.
- Ran each of the two slow three-client subprocess integration tests independently: **both passed**, each with successful exit code. An unsplit full-suite command timed out during subprocess tests; do not claim that invocation passed.
- Ran the live four-process analytics harness and research evidence checker successfully. Recorded structured failed attempts for unavailable Docker/Flower gates. Three new figures generated from real saved results.
- Git history includes baseline commits, sprint hardening, disclosure work, tests and agent-skill handoffs. Files/data are approved public or synthetic only. No GitHub remote/Actions execution is claimed.

## Known limitations and next steps

Docker, Flower, MNIST network download, five independently trained membership models, realistic multi-record/multi-minibatch FedAvg update inversion, OpenDP, Opacus, SecAgg+, dashboard and digital-twin validation remain **unimplemented, blocked, or unverified**. The short training baseline has poor quality, and attacker processes share a development filesystem. Revisit after native runtime gates. `NEXT_SPRINT.md`, `docs/disclosure-protocol.md`, `docs/fl-experiment-protocol.md`, and `AGENTS.md` are the next agent's first reads.
