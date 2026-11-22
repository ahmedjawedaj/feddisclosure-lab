# Changelog

All notable changes are listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are release tags in this project's own history, not hosted releases. Failed gates and withdrawn claims are listed on purpose.

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
