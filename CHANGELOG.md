# Changelog

All notable changes are listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are release tags in this project's own history, not hosted releases. Failed gates and withdrawn claims are listed on purpose.

## [Unreleased]

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

Before this repository was published on GitHub the workflows had never run on hosted runners. Results of the first hosted runs are recorded in the next entry once they are observed.
