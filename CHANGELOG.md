# Changelog

All notable changes are listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are release tags in this project's own history, not hosted releases. Failed gates and withdrawn claims are listed on purpose.

## [0.16.10] - 2026-11-26

Second CIFAR-10 inversion study (tracker item T6b), pre-registered, run on a Colab T4 and reported. No earlier result number, verdict or threshold changed.

**Result in short.** The control passed (+4.71 dB, lower limit +2.41, with a seed spread 1.69 times the assumed one, so its power statement is void). At 4 records per client both plaintext gates failed (PSNR gain +0.74 dB, interval -0.03 to +1.57, and SSIM gain +0.182), so every row of that arm behind the gates is NOT INTERPRETABLE. At 2 records both plaintext gates passed (+2.79 dB and +0.294 SSIM). Behind them collusion keeps the attack above the floor (C2P, C2S), collusion is close to plaintext (Q2P, Q2S) and mixing costs the attacker (D2P, D2S). The aggregate view of 8 mixed images had a PSNR gain of +0.02 dB and rule A2P failed (labelled underpowered, upper limit +1.01 against a margin of +1.0). Its SSIM gain was +0.110 and A2S failed. Fewer records helped (T1 passed). These are statements about the attacks that ran and not about privacy. Details and every row are in `docs/v0.16-p3b-results.md`.

### Added
- **Pre-registration of P3b** `docs/preregistration/v0.16-p3b.md` (commit `71a0bda`, committed alone before the script). P3 found the plaintext view at the label-only floor with 8 records per client, so its aggregate and collusion rows had nothing to be read against. P3b runs two arms with 4 and 2 records per client and a control with 1 record, `breaching` 0.1.2, 20 fresh evaluation seeds (100 to 119) and 3 fresh tuning seeds, a wider total variation grid (0.05, 0.2, 0.8, 3.2) and SSIM as a second metric next to PSNR.
- **Results** `docs/v0.16-p3b-results.md` and the saved run output in `reports/results/privacy/v0160/p3b_breaching/` (summary, tuning, 20 seed files and 3 tuning seed files, from a tgz with SHA-256 `07066551be678fb9ff5dafc9621091276bafdfa1ac2d3b15dca7b2b7615a5e09`). Image files and the model checkpoint are not in the repository.
- **A gate before any aggregate or collusion row is read.** Plaintext must clear the floor by +1.0 dB (lower limit) in PSNR or +0.20 in SSIM for its own arm and metric, and the 1 record control must reach +2.0 dB. A row behind a failed gate is declared NOT INTERPRETABLE in advance and gets no PASS or FAIL, so the vacuous pass of P3 cannot repeat. Confidences, numeric rules and a power section (simulated with the exact rules, 4,000 simulations per cell) are in the file, with a list of the clauses that no script check covers.
- `scripts/p3b_breaching_cifar10.py` (resumable, writes one JSON per seed and a summary with the gates), `scripts/p3b_power_table.py` (the power simulation, its SHA-256 is in the pre-registration), tests `tests/privacy/test_p3b_breaching.py` and the runbook `docs/v0.16-p3b-colab-runbook.md`.

### Recorded on purpose
- Dropped levers and the reasons are in the pre-registration: more local steps, a larger batch, more iterations, the repository attack, the `max_weight` 16 and exact aggregate rows and separate tuning of the 2 record arm. The exact aggregate row is gone, so an aggregate difference cannot be split into mixing and quantisation.
- The SSIM margins (and the margins and expected values of the rules A, D and Q) were chosen after reading the P3 numbers, which the pre-registration says in its deviation log. The SSIM rows are calibrated after P3 and are not confirmatory. The deviation log also records, before any run, that a failed control withholds the gates and T1 as well, and that rule Q gets the label FAIL, underpowered when its point estimate is inside the margin.
- The two failed predictions with confidence above 0.5 are A2P and A2S (confidence 0.8 each). Both were rule A at 2 records. Failed gates G4P (0.45) and G4S (0.25) were the low confidence ones.
- The 2 record arm and the control reused the total variation winner tuned at 4 records (0.2 for all three views, not at the grid edge). That transfer was not tested.
- The run took about 72 minutes in total (about 15 minutes of tuning and 57 minutes of evaluation, 168 to 175 s per seed), below the pre-registered estimate of about 90 minutes.
- A resumed invocation after a Colab reconnect found all 20 seeds present, skipped them and rewrote `summary.json`. It also overwrote the live `run.log`. The seed and tuning files were not touched. This is deviation entry 9 of the pre-registration.
- Package version is 0.16.10.

## [0.16.9] - 2026-11-26

Measurement of the differential privacy aggregators of TensorFlow Federated 0.87.0. No earlier result number, verdict or threshold changed.

### Added
- **Measured noise of TFF's DP aggregators against their configuration.** `scripts/audit_tff_dp.py` runs the real `DifferentiallyPrivateFactory.gaussian_fixed` and `gaussian_adaptive`, a `MeanFactory` around the fixed one and `tff.learning.dp_aggregator` through TFF's runtime, with 14 pre-registered rows plus an environment row and a timing row, and 127 checks. Pre-registration `docs/preregistration/v0.16-tff-dp.md` (commit `7a22b12`, committed alone before the script was written), results `docs/v0.16-tff-dp-audit.md`, evidence `reports/results/privacy/v0160/tff_dp/tff-0.87.0/tff_dp_audit.json`, tests `tests/privacy/test_audit_tff_dp.py`. The tests that call TFF skip when `tensorflow_federated` is not importable, as in CI.
- A README section and claim checks in `scripts/verify_readme_claims.py` for the new numbers, including a check that the evidence has exactly one failed check and that it sits in row F5.

### Recorded on purpose
- 15 of 16 rows and 126 of 127 checks held. The one failed check is a pre-registered number that was an arithmetic slip (output norm 0.0035714 with a weight sum of 14, where the weights sent sum to 10 and the arithmetic gives 0.005). It is kept as written, and the deviation log of the pre-registration says so. No statistical check failed, so no replication ran.
- No delivered value differed from its configured value outside the stated tolerance. Fixed clipping noise equals the noise multiplier times the clip over `clients_per_round` (ratios 1.0021, 1.0003 and 0.9983), and stays so when 5 or 20 clients report against 10. The adaptive clip falls when more clients are below it, the noise uses the clip that the clients used, and the composed delivered multiplier is 0.9989 and 1.0049 of the configured one. The counted fraction is computed against the configured number of clients. The default zeroing step of `dp_aggregator` counts a zeroed client as below the clip. The Python global seed does not control the noise.
- The aggregators measured report a per-round `DpEvent` and no epsilon (no name containing epsilon in the six files scanned), and no epsilon or delta was computed here.
- Clauses that no script check covers, as listed in the pre-registration and the results document: the source and install statements, the five quotes (the tutorial text came from one page fetch), that 0.87.0 is the newest release on PyPI, the paper reading, that the TFF documentation read states no adjacency relation, that `MeanFactory` around the DP factory is not shown as a DP example in the files read, that nothing outside six scanned files reports an epsilon (and what `python/analytics/differential_privacy.py` holds), the false failure estimate, that the C++ worker runs the same kernels as the Python side, and the wording of the README, the document and the comparison with Flower.
- Install note: 0.87.0 pins `jaxlib` 0.4.14, which is not on PyPI, so it was installed with `--find-links https://storage.googleapis.com/jax-releases/jax_releases.html` in a Python 3.11 environment.
- The fast suite records were re-recorded after the branch was built (673 passed and 30 skipped with flwr and opacus, 478 passed and 70 skipped in the CI environment) and the README suite counts were updated.
- Package version is 0.16.9.

## [0.16.8] - 2026-11-26

Events check of the NVFlare SVTPrivacy filter: other events, larger sizes, input values between 0 and gamma and the release values. No earlier result number, verdict or threshold changed.

### Added
- **Events check for SVTPrivacy** (v0.16.5 limit 1). `scripts/audit_nvflare_events.py` runs the real filter 1,000,000 times for each of six configurations (500,000 per input), with 10 pre-registered rows and 83 checks. Pre-registration `docs/preregistration/v0.16-nvflare-events.md` (commit `a1d1ea5`, committed alone before the script was committed), results `docs/v0.16-nvflare-events-audit.md`, evidence `reports/results/privacy/v0160/nvflare_events/nvflare_events.json`, tests `tests/privacy/test_audit_nvflare_events.py`. Each null result has a stated power. The configurations are: d 10 with n 4, d 24 with n 6 and `replace` True, input values 0.2 and 0.8 gamma, d 40 with n 10, the release values (sign flip pair, claim `epsilon_release`) and one event that joins selection and release (claim `epsilon_total`).
- Claim checks in `scripts/verify_readme_claims.py` for the new README section.

### Recorded on purpose
- All 10 rows and 83 checks held. Lower bounds were 0.414, 0.206, 0.224 and 0.280 against a claim of 1, 0.272 against 0.3 for the release values and 0.502 against 1.2 for the joint event. No lower bound exceeded its claim and no replication was run.
- The release row sits closest to its claim, with the event epsilon predicted at 0.976 of `epsilon_release` and measured at 0.962 of it.
- The pre-registration says which clauses no script check covers and what stays untested (other pairs, the marginal event, explicit release or threshold budgets, `replace` True in the release rows, larger sizes, composition across calls). The script used a ledger tolerance of 1e-9 where the text says 1e-12.
- The filter files are byte identical in 2.8.3, 2.9.0 and 2.9.1, so only 2.9.1 was run. The run took 1,048.7 seconds.
- The suite records were re-recorded after main was merged into the branch.
- Package version is 0.16.8.

## [0.16.7] - 2026-11-26

The release gate now runs in CI. No result number, verdict or threshold changed.

### Added
- `.github/workflows/release-gate.yml` runs `scripts/release_gate.py` on every pull request against the base commit of that pull request, with the head commit checked out and the full history fetched. Before this the gate was run by hand and the v0.16.5 limits said it was not enforced in CI.
- `tests/test_release_gate_workflow.py` pins the trigger, the checkout of the head commit and the base commit argument.

### Recorded on purpose
- A pull request that changes `scripts/release_gate.py` or the hooks without changing `scripts/gate_pins.json` fails the gate integrity check in CI. That is intended. A pull request that changes the gate and the pins file together passes the integrity check, and the pins file, the workflow file and `scripts/check_commit_attribution.sh` are not pinned, so a reviewer has to read those files and the owner decides.
- The workflow checks the pull request only. Pushes to main are not gated.
- Package version is 0.16.7.

## [0.16.6] - 2026-11-26

Whole mechanism measurement for Flower adaptive clipping. No earlier result number, verdict or threshold changed.

### Added
- **Measured effective noise multiplier of Flower's whole adaptive clipping mechanism** (plan item T8c). `scripts/audit_flower_adaptive_guarantee.py` runs the four adaptive and four fixed clipping strategy classes (server side and client side, legacy and Message API) over simulated trajectories of the clip on `flwr` 1.27.0 and 1.44.0, 640 rows per version. Evidence `reports/results/privacy/v0160/flower_adaptive_guarantee/flwr-1.27.0/`, `flwr-1.44.0/` and `cross_version/`, results `docs/v0.16-adaptive-guarantee-audit.md`, pre-registration `docs/preregistration/v0.16-adaptive-guarantee.md` (committed alone before the harness existed). Tests in `tests/privacy/test_audit_flower_adaptive_guarantee.py`.
- Claim checks in `scripts/verify_readme_claims.py` for the new README section, including a check that every saved row held in both versions.

### Recorded on purpose
- All 640 rows per version held their pre-registered predictions, and the clip sequences of the two versions agree exactly. The findings are measured multipliers and ratios, not failed predictions.
- The count that Flower releases is the plain sum of bits of 0 or 1 with no shift, so its sensitivity is 1 under add or remove, where the paper's centred bit has magnitude 1/2 as read once.
- The model noise is scaled with the clip after the round's update, so measured against the clip the clients used, the model sum multiplier is the stored multiplier times the ratio of the new clip to the old clip. In the default configuration with every update clipped the composed multiplier with Flower's count is 0.8952 of the configured one, and 0.56949 to 0.99627 of it at a clip ratio of 1 across the count noise values tried.
- No epsilon or delta is reported. The composed numbers depend on assumptions listed in the results document, among them the sensitivity convention, which was not established for Flower.
- The four adaptive variants and both versions share seeds, so their agreement is not independent evidence. One quick smoke run and several choices the pre-registration left open are in its deviation log.
- The suite records were re-recorded after main with v0.16.5 was merged into this branch (649 passed and 27 skipped, and 454 passed and 67 skipped in the CI environment).
- Package version is 0.16.6.

## [0.16.5] - 2026-11-26

Selection check of the NVFlare SVTPrivacy filter with stated power, and release gate hardening. No earlier result number, verdict or threshold changed.

### Added
- **Selection check with stated power for SVTPrivacy.** `scripts/audit_nvflare_selection.py` runs the real filter 1,000,000 times for each of three small configurations, with 7 pre-registered rows and 44 checks. Pre-registration `docs/preregistration/v0.16-nvflare-selection.md` (commit `cdd2ce0`, committed alone before the script) states predictions and the smallest effect found with 80% power for every null result. Evidence `reports/results/privacy/v0160/nvflare_selection/nvflare_selection.json`, results `docs/v0.16-nvflare-selection-audit.md`, tests `tests/privacy/test_audit_nvflare_selection.py`.
- Version identity by running. The two filter files from the wheels of `nvflare` 2.8.3, 2.9.0 and 2.9.1 have equal sha256 and one seeded call of each filter gave the same digest in all three.
- Release gate: a gate integrity check (`scripts/gate_pins.json` pins the sha256 of `scripts/release_gate.py` and the two commit hooks, and a mismatch fails unless `GATE_REPIN=1` is set, which prints the diff against the base), and a pre-registration power check (a changed file with a Predictions heading needs a Power heading). `docs/preregistration/TEMPLATE.md` has the five sections. Tests in `tests/test_release_gate.py`.

### Recorded on purpose
- All 7 rows and 44 checks held and no lower bound exceeded its claim. One unscripted prediction (the log ratio as a fraction of the claim) held only partly, and the results document says where.
- The v0.16.4 selection check (row N6b) could not fail at its claim of 20, because its run count caps the lower bound at about 10.9. The v0.16.4 text stays as written.
- The Monte Carlo ran on 2.9.1 only. Version identity covers the two filter files and one seeded input.
- Two test runs of the script before the full run are in the deviation log.
- Package version is 0.16.5.

## [0.16.4] - 2026-11-25

Fourth framework for the v0.16 DP measurements, plus release tooling. No earlier result number, verdict or threshold changed.

### Added
- **Measured behaviour of NVIDIA FLARE's SVTPrivacy and PercentilePrivacy** (plan item T10). `scripts/audit_nvflare_dp.py` runs the real classes of `nvflare` 2.9.1 with 63 checks built from 18 pre-registered rows. Evidence `reports/results/privacy/v0160/nvflare_dp_audit/nvflare-2.9.1/nvflare_dp_audit.json`, results `docs/v0.16-nvflare-dp-audit.md`, pre-registration `docs/preregistration/v0.16-nvflare.md` (committed before the harness). Tests in `tests/privacy/test_audit_nvflare_dp.py`.
- `scripts/release_gate.py`, which runs the mechanical release checks (attribution, pre-registration integrity, prose rules, held back strings, version agreement, README claims). `scripts/hooks/` and `scripts/install_hooks.sh` add a commit-msg hook that rejects AI trailers and a pre-commit hook for the prose rules.

### Changed
- `CLAUDE.md` was cut to the current rules.
- The paper's bit definition, update rule, order of operations and defaults were checked against the text of the arXiv PDF (through a summarising fetch), and they match the readings used in 0.16.3. `docs/v0.16-adaptive-clipping-audit.md` limit 6 says so.

### Recorded on purpose
- All 18 rows and 63 checks held. The notes in the README section are about behaviour to know and not about failed predictions.
- The selection check (row N6b) has little power at the run count used, and the results document says so.
- Only `nvflare` 2.9.1 was run. The two filter files are identical by hash in 2.8.3 and 2.9.0 and that was not run.
- Exploratory runs before the full run and the choice of unnamed input values are listed in the deviation log.
- Package version is 0.16.4.

## [0.16.3] - 2026-11-25

Adaptive clipping measurement for Flower. No earlier result number, verdict or threshold changed.

### Added
- **Measured adaptive clipping in Flower against Andrew et al. (2021)** (plan item T8b). `scripts/audit_flower_adaptive_clipping.py` runs the four adaptive clipping strategy classes (server side and client side, legacy and Message API) on `flwr` 1.27.0 and 1.44.0, 82 rows per version. Evidence `reports/results/privacy/v0160/flower_adaptive_clipping/flwr-1.27.0/` and `flwr-1.44.0/`, results `docs/v0.16-adaptive-clipping-audit.md`, pre-registration `docs/preregistration/v0.16-adaptive-clipping.md` (committed before the harness existed). Tests in `tests/privacy/test_audit_flower_adaptive_clipping.py`.

### Changed
- `docs/v0.16-p1-flower-dp-audit.md`: two sentences about adaptive clipping were corrected (see below). README gains a section for this measurement.

### Recorded on purpose
- All 82 rows per version held their pre-registered predictions. The findings are differences from the paper's reading and not failed predictions. The main one is that the clipped bit has the opposite sign to the paper's as read once, so the clip moves away from the target quantile (rows A3, A4, A10, A11).
- Correction: the v0.16.0 Flower document said the client-side adaptive variant adds noise before it updates the norm. Row A6 measured the opposite on both client-side variants. The document also said the direction of the clip movement was not compared with the paper, and it now is.
- Smoke runs were done before the full runs, one row configuration was changed before any full run (A8), and the paper's ordering was read from one summarising fetch of a web copy. All three are in the deviation log.
- No privacy guarantee of the whole mechanism was measured and no epsilon is reported.
- Package version is 0.16.3.

## [0.16.2] - 2026-11-25

Second framework for the v0.16 DP measurements. No earlier result number, verdict or threshold changed.

### Added
- **Measured privacy of FedML's DP mechanisms** (plan item P1, second framework). `scripts/audit_fedml_dp.py` calls the classes under `fedml/core/dp` of `fedml` 0.9.6 directly, with 130 checks built from the pre-registered rows (18 were added after the first quick run). Evidence `reports/results/privacy/v0160/fedml_dp_audit/fedml-0.9.6/fedml_dp_audit.json`, results `docs/v0.16-fedml-dp-audit.md`, pre-registration `docs/preregistration/v0.16-fedml.md` (committed before any mechanism ran, with a prior art gate). Tests in `tests/privacy/test_audit_fedml_dp.py`.
- `docs/novelty-status.md` and `docs/v0.16-p2-server-view-inventory.md` (scope only, nothing measured).

### Recorded on purpose
- 128 of 130 checks held. The two that did not concern the error an accountant path raises (a `NameError` where a `TypeError` was predicted, and an error before the first step where none was predicted).
- Wording of limit 8 in `docs/v0.16-p1-flower-dp-audit.md` was clarified so that it does not claim more than the history shows.
- Package version is 0.16.2.

## [0.16.1] - 2026-11-25

Second phase of the v0.16 plan, inversion attack part. No earlier result number, verdict or threshold changed.

### Added
- **Pre-registered breaching study on CIFAR-10** (plan item P3). `scripts/p3_breaching_cifar10.py` runs `breaching` 0.1.2 and the repository's own attack against plaintext, SecAgg+ aggregate and n-1 collusion views, with a label-only floor, 20 evaluation seeds and a tuning rule fixed in advance. Pre-registration `docs/preregistration/v0.16-p3.md` was committed before the script, with a dated deviation log. Results `docs/v0.16-p3-breaching-results.md`, evidence under `reports/results/privacy/v0160/p3_breaching/` (20 seed files, 3 tuning seed files, summary and tuning). Run on a Colab T4. According to the run notes it took two runtimes (seeds 0 to 14 and 15 to 19, same model hash), and the seed files do not record the split.
- Colab runbook `docs/v0.16-p3-colab-runbook.md` and tests in `tests/privacy/test_p3_breaching.py`.

### Verdicts, kept as written
- Pass: K0 (control, +5.50 dB, lower limit +4.71), R2 and R2r (vacuous, the aggregate sits below the floor), O2 (collusion beats the aggregate).
- Fail: R1 and R3 (plaintext -0.12 dB and collusion -0.01 dB, neither beats the floor in PSNR, while SSIM gain is positive), R1r and R3r (repository attack), B1c (mean squared error ratio 2.59 against a limit of 1.5), O1.
- The pre-registered claim that plaintext and collusion sit clearly above the label-only floor is not supported for this model, batch size and number of local steps. The aggregate result is not evidence of privacy.

### Recorded on purpose
- Deviation items 1 to 7 are in the pre-registration. They cover the parameter count (25,578), pre-run probes, the interrupted and resumed run, the transcribed and then replaced summary files, and tuning at the edge of the grid.
- `images_seed0.npz`, `images_seed1.npz` and `model.pt` are not in the repository. The model hash is in every seed file.
- Package version is 0.16.1.

## [0.16.0] - 2026-11-23

First phase of the v0.16 plan. No v0.15 result number, verdict or threshold changed.

### Added
- **Measured privacy of Flower's DP mechanisms.** `scripts/audit_flower_dp.py` runs the real `flwr` strategy and mod classes in process, measures sensitivity and noise scale, and reports exact (epsilon, delta) next to the configured values, with a Clopper-Pearson empirical lower bound where it has power. Evidence for `flwr` 1.27.0 and 1.44.0 is under `reports/results/privacy/v0160/flower_dp_audit/`. Results and limits are in `docs/v0.16-p1-flower-dp-audit.md`.
- **Pre-registration** `docs/preregistration/v0.16-p1.md`, committed before any run, with a deviation log.
- Tests in `tests/privacy/test_audit_flower_dp.py`.

### Recorded on purpose
- Prediction C6 named the wrong mechanism. No privacy mismatch was measured in that row.
- The prior-art check for this phase covered the literature and not Flower's release history. The optional accountant in `flwr` 1.44.0, which raises an error on one measured configuration and releases nothing on another, was found only during the re-check on 1.44.0.
- Review R9 found that row C4 is not reached by a client failure, that the release which introduced the accountant was not checked, and that the adaptive rows compare update noise with the reference algorithm only. The text was corrected before release. No number changed.
- Package version is 0.16.0.

## [0.15.3] - 2026-11-23

Documentation only. No code, result number, verdict or threshold changed.

### Added
- **Related work.** `docs/v0.15-technical-report.md` and `docs/v0.15-wire-validation.md` now point to an open upstream issue and pull request on the mask generator of SecAgg+ (flwrlabs/flower #7522 and #7570). This repository does not analyse mask generation. The wire-level check W1 treats a masked vector as opaque and runs only the gradient-matching attack on it, so W1 says nothing about that question.
- Package version is 0.15.3.

## [0.15.2] - 2026-11-23

Follow-ups after 0.15.1. No verdict or threshold changed.

### Added
- **Hosted Fashion-MNIST rerun.** The `fashion-mnist-study` workflow ran once on a hosted runner (run 45153504593, the same 20 evaluation seeds and not the 30 named in deviation entry 17, its own tuning, torch 2.5.1 as pinned in the workflow). n-1 collusion is +12.5 points over the floor (seed bootstrap +4.4 to +21.3) against +9.4 locally, so the X1 rule would pass on that run. The run was declared a robustness check before it ran, so X1 stays FAIL on the local result. Evidence in `reports/results/privacy/v0150/fashion_mnist/hosted/`, write-up in `docs/v0.15-fashion-mnist.md`. `scripts/recompute_fashion_seed_bootstrap.py` takes `--dir`.

### Changed
- **References checked.** The reference list of `docs/v0.15-technical-report.md` now gives the full author list and pages for Bonawitz et al. (CCS 2017), pages for Bell et al. (CCS 2020) and Carlini et al. (IEEE S&P 2022), and the NeurIPS 2023 venue for Hayes, Mahloujifar and Balle. `docs/v0.15-dp-informed-attack.md` says that the attack is Algorithm 2 of that paper and that the paper states its bound through a blow-up function estimated by Monte Carlo, which reduces to the closed form used here for one Gaussian step. Two statements about `mialibrary` that could not be confirmed (its authors' affiliation and its first release date) were removed from `docs/v0.15-external-baselines.md`.
- **Host paths redacted in v0.14 evidence.** The home directory of the execution host was replaced by `/home/fedlab` in 156 saved v0.14 evidence files, and one log line that listed proxy host names was replaced by a marker. No number or verdict changed. Three file hashes in `dashboard/dist/evidence_snapshot.json` were regenerated. Two tree digests recorded in `fault_matrix/gates/gate_results.json` were computed before the redaction, match the files as of v0.15.1 and were left as recorded. Details in `docs/data-provenance.md`. This supersedes the sentence in the 0.15.0 notes that no v0.14 evidence was changed.
- `scripts/gen_compose_clients.py` uses a simpler regular expression with the same output (an alternation with two overlapping branches was reported by code scanning).
- **Wording corrected after a further review pass (R7).** The 0.15.1 statement that the cause of the 5 of 7 parity result is float32 accumulation over 10 rounds, with reply order changing only the last bits, was too strong. The two failing runs fail for all 6 orders from round 2 on, a single round never fails the gate, two other 10-round runs pass, and reply order changes the final gap by a factor of about 2.4 in one run. The float32 cause is likely but was inferred and not isolated. Fixed in `README.md`, `CURRENT_STATE.md`, `docs/v0.15-reply-order.md` and the 0.15.1 section below. The README list of readings with no recorded time now also names W5, S1 and S3. One test comment about an upstream docstring was reworded, and the quick start bullet of the 0.15.1 section was reworded.
- Package version is 0.15.2.

## [0.15.1] - 2026-11-23

Wording and housekeeping corrections after a further review pass (R6) of the 0.15.0 release. No result number, verdict or threshold changed. The 0.15.0 section below carries the corrected wording.

### Corrected
- **P2 and the Determinism paragraph.** 0.15.0 said two of the seven 3-client native replays show reply-order dependence. They fail the float64 parity gate for all 6 orders, so their failure does not depend on the order. Order dependence of the gate verdict is shown only for the exploratory 4-client MNIST setup (83.3% of orders pass). The README no longer names reply order as the cause of the 5 of 7 parity result. The likely cause is float32 server arithmetic against a float64 reference, amplified over rounds of training. This is inferred and was not isolated (wording corrected again in 0.15.2). Fixed in `README.md`, this file, `CURRENT_STATE.md`, `docs/v0.15-technical-report.md` and `docs/v0.15-reply-order.md`.
- **F2.** The technical report, `NEXT_SPRINT.md` and the README weak-pass sentence now say that F2 passes on the white-box noise audit only.
- **X1.** The README now says that the registration names no max weight and that the verdict is read at 1000.
- **Readings fixed after the fact.** The README now says that the deviation log gives no time for the D5, W3, P1 and F2 readings, and that W3 and F2 would fail under the other reading.
- **Scaling.** `docs/v0.15-scaling.md` said the fewer-shares check was not run in a native 8-client round. It was run once (7 shares, threshold 4).
- **Quick start.** The full-environment pass count (551 at that release) is for an environment without `opendp` and `torchvision`.
- **Authorship and timing statements.** `docs/preregistration/v0.15.md` now gives the exact commit range for convention 7, notes that merge commits made on GitHub use the account's noreply address, and says that the order of local runs relative to the registration rests on commit order.
- **Reviews.** R2 to R4 are now described as separate review passes.
- **Other wording.** Deviation entries 13 and 21 and `docs/v0.15-fashion-mnist.md` no longer call X1 a coin flip, and the numbers are unchanged. The E3 line of the proof note gives both shared-seed figures. `docs/v0.15-fault-matrix.md` dropped a remark about the authoring machine, and `docs/v0.14-native-execution.md` describes the host as a cloud workspace.

### Changed
- `docs/proofs/one-round-secagg-error.md` states the limit of 1023 clients as an assumption of the note. `scripts/run_secagg_bound_search.py` and its test check that assumption only (`sum_range_check`), and the matching block of `secagg_bound_search.json` was renamed to `sum_range` and reduced to that check.
- Seven `scripts/run_*.sh` files default to `$HOME/runs` and `$HOME/venv` and no longer to a fixed home directory.
- Package version is 0.15.1.

## [0.15.0] - 2026-11-23

Release 0.15.0 is the attack-phase release. Every prediction was registered with a numeric pass rule in `docs/preregistration/v0.15.md` before the runs, and no threshold was changed afterwards. Where a rule was ambiguous the reading used and the verdict under the other reading are in that file's deviation log. No v0.14 evidence was changed.

What was simulated and what was deployed. The proofs and property tests, the DP bound study, the fault matrix, the scaling attack results (4 and 8 clients) and the Fashion-MNIST study are in-process or simulated, with the real Opacus engine, the real upstream SecAgg+ quantiser arithmetic and real PyTorch training where stated. Wire-level validation and the live parity runs used a deployed Flower 1.27.0 stack in Docker on GitHub hosted runners, at 4 clients (parity at 3 and 4) and one or ten rounds. The 8-client Docker runs measured memory only. No attack was run on captured traffic from an 8-client stack.

### Added
- **One-round SecAgg+ error proof and property tests** (E1 to E3, M1). `docs/proofs/one-round-secagg-error.md`, `scripts/run_secagg_bound_search.py`, `tests/training/test_secagg_error_bound.py` (uses `hypothesis`). The bound 2cn/T has zero counterexamples in 20,000 float32 cases. The Hoeffding bound holds with independent client randomness (99th percentile 0.67 to 0.73 of the bound for 4 to 256 clients) and fails when all clients share one seed (3.9 to 7.2 times the bound at 256 clients). A two-round ReLU counterexample is reproduced and the convex L2 bound has no violation in 1,000 trials. The note assumes at most 1023 clients at the default settings.
- **DP against an informed attacker** (D1 to D5). `src/feddisclosure/attacks/dp_informed.py`, `scripts/run_dp_informed_attack.py`, `docs/v0.15-dp-informed-attack.md`. A replication of the reconstruction-robustness setting of Hayes, Mahloujifar and Balle for one full-batch Opacus step: the closed-form bound, the exact optimal success on the real candidate sets and an inner-product attack, 1,000 trials per cell. Record-level adjacency, one step, no accountant, so no epsilon claim.
- **Wire-level validation** (W1 to W6). A recording proxy at the grid (`src/feddisclosure/training/secagg_server_capture.py`, `wire_capture.py`), a TCP relay in front of each SuperNode (`docker/wire_relay/`, `compose.wire.yaml`), `scripts/run_wire_capture.py`, `scripts/analyze_wire_capture.py` and the `wire-capture` workflow. Hosted run 45075971652, 4 clients, 5 seeds. Evidence in `reports/results/privacy/v0150/wire/live/`.
- **Seeded-fault matrix** (F1, F2). `src/feddisclosure/evaluation/fault_matrix.py`, `scripts/run_fault_matrix.py`, `docs/v0.15-fault-matrix.md`. Five faults (wrong DP noise scale, skipped clipping, plaintext fallback, wrong threshold or dropout handling, stale global model) against the existing verifiers and four added detectors. Tampered copies are refused inside saved evidence. F2 passes on the white-box noise audit only, which needs the honest clipped sum. The black-box detector has power 0 at noise factor 2, so F2 would fail on that reading.
- **External baselines** (B1, B2). `src/feddisclosure/evaluation/external_baselines.py`, `scripts/run_external_baselines.py`, `requirements-external.txt`, `docs/v0.15-external-baselines.md`. The repository inversion attack against `breaching` 0.1.2 and the repository LiRA against `mialibrary` 0.1.5 on matched setups, with an independent LiRA reimplementation as a cross check. `mialibrary` declares no licence, so it is used as an installed dependency only.
- **Reply order and an opt-in deterministic aggregation** (P1 to P3). `DeterministicFedAvg` (`src/feddisclosure/training/deterministic_fedavg.py`), enabled by `aggregation="deterministic"` or `FDL_AGGREGATION=deterministic`. It sorts replies by site id and uses the float64 reference arithmetic, so the float64 parity gate then tests transport and client determinism and no longer tests Flower's own aggregation. Default behaviour is unchanged. `scripts/study_reply_order.py`, `docs/v0.15-reply-order.md`, the `parity-live` workflow (hosted run 45076810526, six runs, all pass the unchanged gate).
- **Scaling to 8 clients and Fashion-MNIST** (S1 to S3, X1). `scripts/run_scaling_study.py`, `scripts/run_scaling_n8.py`, `compose.flower.n8.yaml`, the `scaling-n8` workflow, `src/feddisclosure/training/secagg_params.py` (fewer shares than clients, rejected when threshold equals shares in the upstream workflow), `scripts/run_fashion_mnist_study.py`, `src/feddisclosure/training/fashion_mnist.py`, `docs/v0.15-scaling.md`, `docs/v0.15-fashion-mnist.md`. The hosted 8-client rounds (8 shares and threshold 5, and 7 shares and threshold 4) finished and were attested. The `fashion-mnist-study` workflow is written and statically tested and has not been run.
- **Reproducibility tooling.** `docs/reproducibility-v0.15.md` (which torch produced which evidence, what review R4 reproduced). `scripts/record_fast_suite.py` records the commit, the packages and the closing pytest line per environment. `scripts/verify_readme_claims.py` now also checks the numbers and the PASS or FAIL words of the v0.15 README section against the `prereg_check_*.json` files, and flags a suite record older than the code. Tests that regenerate `prereg_check_parity.json` and `prereg_check_external.json` from the saved inputs and compare them with the committed files.

### Changed
- **README.** The headline section now leads with the v0.15 work packages and a list of failed predictions, and keeps the corrected v0.14.3 reading. Updated statements: the Determinism paragraph, the Scale, Data and Hosted CI limits, the statement that the inversion study is an observation-model study (now partly checked by wire capture at 4 clients only) and the DP caveat (an informed-attacker study for one step now exists). New capability rows for proofs, external baselines and the fault matrix. The quick start gives the CPU torch install hint and the fast suite count for each environment.
- **Pins.** `docs/v0.14-reproduce.md` paired torch 2.5.1 with Opacus 1.6.0. The Opacus 1.6.0 package metadata says `torch>=2.6.0`. The Docker based runs use torch 2.5.1 and no Opacus. Anything that imports Opacus needs torch 2.6.0 or newer.
- **Safer script defaults.** `scripts/run_flower_docker.py` without `--run-dir` now writes to a new folder under `reports/results/native_flower/local/` and not into the saved v012 evidence. `scripts/write_prereg_checks_wp9_wp10.py` has argparse, so `--help` writes nothing. `scripts/study_reply_order.py summarize` and `scripts/run_external_baselines.py check` no longer erase wording added to their output files by hand. That wording is in the scripts, and `check` keeps saved package versions for packages that are not installed. `replay` says `checkpoints not in checkout` when its git-ignored inputs are absent.
- **Tests.** The two fault matrix gate tests skip with a reason when `data/` partitions and `*.pt` checkpoints (git ignored) are absent, and the Opacus step test in `test_view_study.py` skips without `opacus`. `hypothesis` is in the `test` extra.
- `NOTICE` names the small Digits subset committed as wire evidence and the Fashion-MNIST licence (MIT, Zalando Research, loaded at run time and not redistributed).
- Package version is 0.15.0.

### Withdrawn or corrected
- No v0.14 claim was withdrawn in this release. The v0.14.3 corrections stand. Later notes: the 67.5% label floor and its 24 to 26 point gaps are for seeds 0 to 9, and on 30 seeds the floor is 75.0% (see `docs/v0.15-scaling.md`). The recorded fast suite count of 234 passed and 4 skipped was stale and is replaced by one count per environment.
- Wording fixes made after separate review passes R2, R3 and R4 are listed in the deviation log of `docs/preregistration/v0.15.md` and in the saved `prereg_check_*.json` notes. None changed a threshold or a result number.

### Failed predictions
- **E2** (float32 path misses the float64 bound). The float32 client path adds a bias of up to 3/16 of a quantisation unit, but a client error never exceeds one unit, so the float64 bound 2cn/T holds.
- **D3** (attack near the bound at sigma 4). Same-label candidates are 0.055 below the bound against a 0.05 limit. Any-label candidates pass (0.022). The rule needed both.
- **B1** (repository inversion within 1.5 times the external baseline). Against the tuned `breaching` configuration the ratio is 1.86 (bootstrap 95% interval 1.47 to 2.51). Against the two shipped library configurations it would pass (0.24 and 0.47). The repository attack was not tuned. Every aggregate-view statement is conditional on the weaker attack.
- **P2** (upstream parity depends on reply order at 4 clients). The shipped 4-client config passes for all 24 orders. An exploratory 4-client MNIST 10-round setup passes for 83.3% of orders. Two of the seven 3-client native replays fail the gate for all 6 orders (MNIST seed 3 and MNIST label skew), so their failure is not order dependent. The other five pass for all 6 orders.
- **X1** (Fashion-MNIST shows the same ordering). n-1 collusion is +9.4 points over the floor against a +10 rule, with a seed bootstrap interval of +0.6 to +18.8 over 20 seeds. At `max_weight` 16 the same rule would pass (+13.7), and the verdict stays FAIL.

Passes that need care: F1 passes by its numeric rule (4 of 5 detected) but the plaintext fallback evades the specified detectors, so the prediction text is partly falsified. W1 and S1 pass vacuously (the attack is far below the floor, which says the attack is weak and is not privacy evidence). W3 passes on pooled figures with 5 seeds, and a per-seed reading would fail. W5 covers sample counts only because the client returns no metrics. P1 holds in replay only. P3 passes but the metric cannot fail on that setup.

### Notes
- **Credit.** Reply-order dependence and the resulting non-determinism of Flower aggregation were already discussed publicly, in a thread on the Flower forum about non-determinism and in the BlazeFL project (Azuma and Nishio, arXiv 2604.03606), which works on deterministic federated simulation. The forum thread is https://discuss.flower.ai/t/non-determinism-only-when-number-of-clients-is-increased/807. This repository does not claim to have discovered it. It measures how the effect interacts with its own 1e-5 parity gate, and `DeterministicFedAvg` is an opt-in workaround. The proof and bound work uses standard tools (Hoeffding's inequality, Neyman-Pearson), and the DP study replicates Hayes, Mahloujifar and Balle.
- **Reproducibility.** A separate review pass (R4) regenerated the scaling study (61 JSON files), the DP informed trials, the fault matrix summary and the external inversion summary byte for byte, and the SecAgg+ bound search, the label floor and the interval recomputation numerically, with only runtime or version fields differing. Local v0.15 runs used torch 2.15.0+cpu. The Docker image and the hosted workflows use torch 2.5.1. Details in `docs/reproducibility-v0.15.md`.
- **Fast suite.** The count depends on what is installed. The README gives it for the hosted CI unit job environment and for a full environment.
- **Open.** A native 8-client attack capture, an informed attacker with a public prior against aggregate views, a stronger inversion on protected views (B1 failed), live MNIST reply-order configs for P2, a CI job with the full environment, tuning and rerunning X1 on a hosted runner, and the float32 term of E2 are listed in `NEXT_SPRINT.md`.

## [0.14.3] - 2026-11-23

Release 0.14.3 corrects how the v0.14.2 update-inversion results were read. No attack was rerun and no saved v0.14.2 evidence was changed. New files hold recomputed statistics and the missing baseline.

### Withdrawn or corrected
- **Aggregate-only leakage claim, withdrawn.** v0.14.2 said SecAgg+ aggregate-only inversion "still leaves a lot of signal" (victim MSE 0.057, 55% recognisable, 55 to 59% across the two quantiser settings). The attacker knows the labels. A label-only floor, which rebuilds each victim record as the class-average image of a 1765-image labelled public pool, scores MSE 0.046 and 67.5% recognisable (54 of 80 images) on the same seeds. The attack with max weight 1000 does not beat that floor. With max weight 16 (58.8%) the paired difference to the floor is -8.8 points with an interval of -23.8 to +5.0, so it cannot be told apart from the floor. The attack uses no public images, so a prior-initialised attacker could beat the floor. The floor depends on the pool size (31% recognisable at 20 images, 48% at 50, 57% at 100, 64% at 300), and the aggregate row is below it only for a pool of roughly 100 or more labelled public images. This is a statement about this attack and not a proof that the aggregate is safe. The old numbers are unchanged.
- **DP "stopped this attack" reading, withdrawn.** The 0% recognisable result at sigma 0.2 (MSE 0.179, worse than the mean image at 0.073) only shows that a naive attacker, one that does not model clipping or noise, fails. The same holds for the other sigmas, and sigma 0.05 is already worse than the mean image (0.101 against 0.073), so the point does not depend on a threshold. None of it is evidence that DP protects against a stronger attacker.
- **"SecAgg+ adds no visible further protection once DP noise is there", withdrawn.** It was read off a naive attacker that already fails against the noise, so it cannot show that SecAgg+ adds nothing.
- **"DP keeps working under collusion because the colluders only recover the noised update", withdrawn.** It states a protection result for DP. The experiment shows only that the naive attacker also fails on the noised update.
- **t-intervals on rates, replaced.** Recognisable shares were given with t-intervals over 10 seeds, which can exceed 1. They are now reported with Wilson and Clopper-Pearson intervals pooled over the 80 victim images. MSE intervals are percentile bootstrap intervals over seeds. Point estimates are identical to the saved summary (difference below 1e-9).
- **Weak mean-image baseline, replaced.** The v0.14.2 floor was the overall mean image (MSE 0.073). That is the wrong baseline for an attacker that knows labels, so every inversion number is now shown next to the label-only floor. The mean-image floor is renamed "mean-image floor (ignores labels)". The old name "no-information floor" is retired.
- **Faculty briefing slide 9, superseded.** The v0.14.0 deck said SecAgg+ "still leaves over half the victim images recognisable" and that DP noise "pushed reconstruction past the no-information floor". Slide 9 of `reports/faculty-briefing-v0.14.0.pptx` has been rebuilt with the corrected reading. Copies of the deck made before 0.14.3 carry the withdrawn claims on slide 9 and are superseded.

### Added
- `label_floor` and `identification_rate` in `feddisclosure.attacks.view_study`, and `scripts/compute_label_floor.py` writing `reports/results/privacy/v0143/label_floor.json`, including a pool-size sensitivity block (pool sizes 20, 50, 100, 300 and full, 20 random subsets per seed with a fixed generator). The floor uses a 1765-image labelled public pool, against the 32 records the clients hold, and is a prior-knowledge reference, not an attack. `identification_rate` is reported for the floor only. It is a separate metric, not comparable to MSE, and no attack view has an identification rate.
- `scripts/recompute_view_intervals.py` writing `view_intervals.json` and `prereg_check_corrections.json` (checks C1 to C5 of `docs/preregistration/v0.15.md`) in the same directory. C1, C2 and C4 are marked as consistency checks on data already seen (post-probe), not confirmations. C5 does not count skipped tests as passed and also checks the MSE intervals.
- `scripts/verify_readme_claims.py`, a hygiene check that the README table, the numbers quoted in its prose and its verdict words match the saved evidence files. It fails closed (NaN and infinity never pass, exponents are parsed). `scripts/record_fast_suite.py` runs the fast suite and records pytest's own closing line, which the check reads.
- `feddisclosure.attacks.view_seed`, the data selection and per-client seed shared by the view scripts.
- Tests for the label floor and its pool, the interval recomputation (hand-computable cases, threshold boundaries), the pooled Wilson helper and the overwrite guard of `run_view_comparison.py`, and the claim verifier (tolerance, NaN, stale verdicts).

### Changed
- **Docs and README.** `README.md`, `CURRENT_STATE.md`, `NEXT_SPRINT.md` and `docs/v0.14-privacy-secagg-results.md` now state the corrected reading, the pool-size table, and that the attack uses no public images. `docs/v0.14-reproduce.md` explains the `--output` and `--overwrite-summary` guard.
- **Dashboard.** `dashboard/evidence.py`, `dashboard/build_dashboard.py` and `dashboard/dist/` show the label-only floor, its pool size and the rate difference to the floor for every inversion view. DP rows carry the note that the attacker is naive and that this is not a protection claim.
- **Briefing deck.** `scripts/build_faculty_briefing_v014.py` and the rebuilt `reports/faculty-briefing-v0.14.0.pptx` use the label floor on slide 9.
- `scripts/run_view_comparison.py` has a `--legacy-seeds` flag, on by default so the saved v0.14.2 runs still reproduce. `--no-legacy-seeds` uses `feddisclosure.evaluation.intervals.client_seed` and writes to its own directory. Writing into the saved v0.14.2 directory needs `--overwrite-summary`. New runs also store the label floor and use pooled Wilson intervals for rates.
- Package version is 0.14.3.

## Published with 0.14.2: hosted CI and repository fixes

These changes were made while the repository was first published on GitHub and its workflows first ran on hosted runners. They ship in the 0.14.2 release and do not change any research result.

### Added
- `commit-attribution` workflow and `scripts/check_commit_attribution.sh`. Pushes and pull requests fail if any commit has an AI assistant as author, committer or co-author.

### Fixed
- `scripts/verify_compose_isolation.py` assumed all four services share one network. The gateway now also joins a separate `edge` network so the host can reach its loopback port. The check now requires the three institutions to share exactly one internal-only network and allows the gateway one extra network. The first hosted `tests` run failed on this and the other five jobs passed.
- The gateway had no healthcheck, so `docker compose up --wait` could return before it was listening. The history gate then failed intermittently on a connection error, passing on one commit and failing on the next. The gateway now has a `/health` healthcheck.
- The native Flower Docker run failed on the unprivileged hosted runner with `Permission denied` on `clients/site-0/events.jsonl`. The container wrote the attestation file as mode 0600 under its own uid, and the runner user could not read it. It is now mode 0644. The file holds digests and counts only. This never showed in the root sandbox where the earlier native runs were produced.
- Flower simulation runs (`flower-smoke`, the upstream SecAgg+ step of the v1 gate) failed on the hosted runner with "Connection to the SuperLink is unavailable" because the CLI connected before the local SuperLink was listening. `scripts/run_flower_experiment.py` and `scripts/run_secagg_native.py` now go through `feddisclosure.training.flwr_cli.run_flwr_cli`, which retries only that exact message, at most 6 times, and records every attempt in the evidence (`cli_attempts`). Other failures and timeouts are final. Both scripts also print the last lines of Flower's output when they fail.
- Hosted CI evidence showed that the SuperLink retry alone was not enough. The first `flwr run` attempt registered a run and then lost its connection. The second attempt started a new SuperLink on top of the half-initialised database and crashed with `table node already exists` (an earlier run of the same workflow hung until its 15 minute script timeout). `run_flwr_cli` now stops leftover `flower-superlink` processes and clears `~/.flwr/local-superlink` before each retry, and records the output tail of every attempt in `cli_attempts`. The SecAgg+ smoke script also prints a process list and the SuperLink log when it fails. A third hosted failure (`Exception calling application: 'config'`) occurred on the first `flwr` call on a clean runner, when Flower migrates the legacy `[tool.flwr.federations]` table out of `pyproject.toml` (it rewrites the file while the run starts). The legacy table is removed from the root and `experiments/flower_secagg` `pyproject.toml` files. `scripts/run_flower_experiment.py` and `scripts/run_secagg_native.py` now call `ensure_simulation_connection()`, which writes a three-supernode `local-simulation` connection to `~/.flwr/config.toml` (never overwriting a conflicting one), and run `flwr run . local-simulation`. Locally, five fresh-home SecAgg+ runs passed in 4 of 5 attempts and the other failed on a Ray raylet start-up timeout, which is a separate, still-open source of flakiness. Hosted `flower-smoke` then showed the underlying cause, which is consistent with a race in the SuperLink's state database set-up (Alembic) on a clean host: on a clean runner the local SuperLink can crash while creating its SQLite tables (`table node already exists`), which the CLI reports as "Connection to the SuperLink is unavailable", and the retry could hit the same race again. `run_flwr_cli` therefore retries on four named start-up signatures only (`superlink_unavailable`, `sqlite_table_exists`, `ray_startup_timeout`, `superlink_servicer_error`), matched after removing colour codes and box drawing from the CLI output. `superlink_servicer_error` covers the SuperLink's own StartRun handler failing before any application code runs; on hosted runners it surfaced as `'config'`, `table node already exists` and `invalid interpolation syntax in 'sqlite:///%3Amemory%3A'`, a different message on different runs, which points at a thread-safety problem in Flower's state-database set-up rather than at this repository. Up to 8 attempts are made, 3 seconds apart. A SuperLink that is still booting (`superlink_unavailable` while its Control API port is listening or comes up within 20 seconds) is kept and reconnected to, because resetting it restarted the boot and lost the same race again on all 8 attempts of the hosted v1 gate. Every other race, and an unavailable SuperLink that is not listening, gets a clean reset first. The artifact of the failed hosted gate run showed eight attempts with five different errors (`'config'`, `'script'`, `table node already exists`, a dropped connection), which is Flower running its Alembic state migrations concurrently in the SuperLink, its handler threads and the simulation subprocess. `feddisclosure/training/flwr_shim/sitecustomize.py` is put on `PYTHONPATH` for `flwr run` and, only inside Flower's own console scripts, wraps `run_migrations` with a thread lock and a file lock so migrations run one at a time (an in-memory database has nothing to migrate and is skipped). The real SuperLink, SuperNodes, ServerApp and ClientApps still run, and nothing about what is verified changes. Note for local runs: `flwr run` leaves its SuperLink running in the background, so repeated local runs reuse it and hide this start-up behaviour. Kill `flower-superlink` before a local cold-start test. Each retry is recorded as `startup_race` in `cli_attempts`. Training errors, failed assertions and timeouts remain final. The gates themselves are unchanged.

## [0.14.2] - 2026-11-22

### Added
- Update-inversion comparison across plaintext, SecAgg+ aggregate, n-1 collusion and Opacus DP views (4 clients, 10 seeds), using the upstream SecAgg+ quantiser arithmetic.
- 300-trial Monte Carlo and an empirical percentile acceptance rule for two-round SecAgg+ error.
- Float32 bitwise replay gate over 7 native Flower runs (7 of 7 pass).
- LiRA at 64 shadows and 16 targets on Digits and full MNIST.
- Static workflow consistency check (`scripts/check_workflows_static.py`), explicitly not a hosted run.

### Changed
- The 24-shadow LiRA result was found to understate leakage at epsilon 6 to 8. The 64-shadow run replaces it in the conclusions and the correction is documented.

### Still failing or open
- The strict float64 parity gate passes 5 of 7 native runs and is unchanged.
- No proven multi-round SecAgg+ error bound. Only a percentile rule exists.
- MNIST licence terms unresolved.

## [0.14.1] - 2026-11-22

### Added
- 4-client native topology, SecAgg+ dropout matrix (5 of 5 matched the share-count rule), fail-fast watchdog, MNIST native matrix, first LiRA study.

### Withdrawn
- The earlier claim that 3-client SecAgg+ dropout always halts. Four of five scenarios were tolerated.

## [0.14.0] - 2026-11-21

### Added
- Full 60,000-image MNIST native Docker and Flower run, MNIST DP study, dashboard, deck and docs.
- Two of three extra full-MNIST native runs failed the 1e-5 float64 parity gate (float32 reply-order aggregation). The gate was not relaxed.

## [0.13.0] - 2026-11-20

### Added
- Three isolated Linux-namespace PyTorch clients trained two FedAvg rounds. This was not Docker-based Flower training.

## [0.12.0] - 2026-11-20

### Added
- Docker Flower deployment files, partition staging, container isolation checker, per-client receipts and a strict parity gate. Not natively executed at this version.

## [0.11.0] - 2026-11-20

### Added
- Retry-safe, idempotent DP query handling and a fault-injection benchmark (deterministic test mechanism, no DP guarantee).

## [0.10.0] - 2026-11-20

### Added
- Demonstration of an experiment-ID reset bypass of the original query-history policy and a persistent principal-wide control.
- 90-population, four-policy matrix and a ten-model, five-seed output-only membership comparison.

## [0.9.0] to [0.4.1] - 2026-11-19

### Added
- Joint aggregate-only inversion, paired uncertainty analysis, offline research explorer, five-seed equal-capacity benchmark, collusion benchmark, multi-record update inversion and FedProx comparison.

See [`docs/history/README-legacy.md`](docs/history/README-legacy.md) for the full per-milestone narrative.

## Hosted CI

On 2026-11-22 every workflow was run on hosted runners for the first time. The first runs failed in several ways, all listed above, and none was hidden or relaxed. After the fixes, the push and pull request workflows (tests on Python 3.11, 3.12 and 3.13, history controls, commit attribution, CodeQL) and the native workflows (Flower smoke, SecAgg+ smoke, public-data research smoke, native v1 gate, Docker certification) passed on the merged `main`. Balanced heterogeneity, Opacus and OpenDP were dispatched on the commit before the merge and passed there. A rare Ray start-up timeout was seen locally but not on hosted runners. The workflows retry it, and the retry is recorded in the evidence.
