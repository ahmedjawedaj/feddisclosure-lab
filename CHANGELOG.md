# Changelog

All notable changes are listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are release tags in this project's own history, not hosted releases. Failed gates and withdrawn claims are listed on purpose.

## [Unreleased]

### Added
- `commit-attribution` workflow and `scripts/check_commit_attribution.sh`. Pushes and pull requests fail if any commit has an AI assistant as author, committer or co-author.

### Fixed
- `scripts/verify_compose_isolation.py` assumed all four services share one network. The gateway now also joins a separate `edge` network so the host can reach its loopback port. The check now requires the three institutions to share exactly one internal-only network and allows the gateway one extra network. The first hosted `tests` run failed on this and the other five jobs passed.

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
