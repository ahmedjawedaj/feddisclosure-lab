# FedDisclosure Lab

[![tests](https://github.com/ahmedjawedaj/feddisclosure-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedjawedaj/feddisclosure-lab/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ahmedjawedaj/feddisclosure-lab/actions/workflows/codeql.yml/badge.svg)](https://github.com/ahmedjawedaj/feddisclosure-lab/actions/workflows/codeql.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20to%203.13-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-research%20benchmark-orange.svg)](#status-and-honest-limits)

**A reproducible laboratory for measuring how much a federated learning system actually discloses, and how much common defenses really help.**

FedDisclosure Lab trains real federated models with [Flower](https://flower.ai) in Docker, applies real privacy mechanisms ([Opacus](https://opacus.ai) DP-SGD, [OpenDP](https://opendp.org) releases, Flower SecAgg+), and then attacks the result with membership inference and update inversion. Every claim is tied to saved evidence, and negative results are kept next to the positive ones.

> **This is a research benchmark, not a privacy product.** It makes no formal security or privacy certification claim. Read [Status and honest limits](#status-and-honest-limits) before you cite or reuse any number.

## Why this exists

Federated learning is often described as private because raw data never leaves the client. The literature and our own measurements say that is not enough. Updates leak, secure aggregation has failure modes, and differential privacy trades away accuracy that some clients pay for more than others. This repo asks narrow, testable questions and refuses to round the answers in anyone's favor.

## What you can do with it

| Area | What is implemented | Where |
|---|---|---|
| Native federated training | Flower 1.27 Deployment Runtime in Docker Compose (SuperLink, SuperNodes, isolated client containers) on 3 and 4 clients, with per-client execution receipts and a fail-fast watchdog | `compose.flower*.yaml`, `scripts/run_flower_docker.py` |
| Secure aggregation | Upstream Flower SecAgg+ runs, a dropout matrix, an error-propagation study and a multi-round acceptance rule | `scripts/run_secagg_*`, `scripts/verify_secagg_dropout_matrix.py` |
| Differential privacy | Opacus DP-FedAvg with an RDP accountant, OpenDP counting releases with a persistent budget ledger | `src/feddisclosure/defenses/` |
| Membership inference | LiRA shadow-model attack, 64 shadows and 16 targets, on Digits and full MNIST | `scripts/run_lira_study.py` |
| Update inversion | Gradient-matching reconstruction against plaintext, SecAgg+ aggregate, n-1 collusion and DP views | `src/feddisclosure/attacks/view_study.py`, `scripts/run_view_comparison.py` |
| Determinism gates | Bitwise float32 replay of Flower aggregation and a strict float64 parity gate (kept strict even where it fails) | `scripts/reproduce_flower_bitwise.py`, `scripts/verify_flower_replay_gate.py` |
| Federated analytics | Three-site gateway with disclosure controls and a persistent principal-wide query history policy | `services/`, `src/feddisclosure/analytics/` |
| Evidence dashboard | Offline HTML explorer that reads only allowlisted scalar summaries | `dashboard/` |

Only synthetic or public data are used (scikit-learn Digits and the MNIST training split). See [`docs/data-provenance.md`](docs/data-provenance.md).

## Headline results (v0.14.2)

All numbers come from saved runs under `reports/results/`. Context and caveats are in [`docs/v0.14-privacy-secagg-results.md`](docs/v0.14-privacy-secagg-results.md).

**Update inversion, 4 clients, 10 seeds, labels known to the attacker.** Lower error means a better reconstruction. The no-information floor (mean image) is 0.073.

| Attacker view | Victim MSE | Share of images recognisable |
|---|---|---|
| Plaintext update | 0.013 | 94% |
| SecAgg+ aggregate only | 0.057 | 55% |
| SecAgg+ with n-1 colluders | 0.023 | 91% |
| Opacus DP, sigma 0.2 (individual update) | 0.179 | 0% |

Aggregate-only SecAgg+ still leaves a lot of signal in this favorable single-step setup, and collusion removes the protection entirely. The DP row only shows that a naive attacker fails. A DP-aware attacker has not been run yet.

**Membership inference (LiRA AUC, 64 shadows, 16 targets).** 0.5 is chance.

| Setting | No DP | DP, epsilon about 6 to 8 | DP, epsilon about 1.5 to 2 |
|---|---|---|---|
| Digits, overfit regime | 0.742 | 0.543 | 0.518 |
| Digits, moderate regime | 0.605 | 0.537 | 0.508 |
| MNIST, overfit regime | 0.823 | 0.575 | 0.504 |

DP at epsilon near 2 brings the attack to chance, but held-out accuracy collapses in the overfit runs (about 0.2, versus 0.84 to 0.92 without DP). An earlier 24-shadow run understated leakage at epsilon 6 to 8. The 64-shadow run corrected it and the correction is documented.

**SecAgg+ dropout.** Five mid-round dropout scenarios matched the share-count rule: four tolerated, one halted. An earlier claim that three-client dropout halts was wrong and is withdrawn in the docs.

**Determinism.** Bitwise float32 replay reproduces 7 of 7 native Flower runs. The stricter float64 reference parity gate (1e-5) passes only 5 of 7, because Flower averages float32 updates in reply order. The gate was not relaxed.

## Quick start

Requirements: Python 3.11 to 3.13. Docker Engine with Compose v2 is needed only for the native Flower runs.

```bash
git clone https://github.com/ahmedjawedaj/feddisclosure-lab.git
cd feddisclosure-lab
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test,research]'

# fast test suite (153 pass and 5 skip, no Docker needed)
PYTHONPATH=src:. pytest -q -m 'not slow'

# rebuild the offline evidence dashboard
python -m dashboard.build_dashboard
```

The three-site analytics demo and the native Flower runs have their own steps. See [`docs/v0.14-reproduce.md`](docs/v0.14-reproduce.md) for the pinned versions (Flower 1.27.0, torch 2.5.1 CPU, Opacus 1.6.0, OpenDP 0.16.0) and the exact commands behind every table above.

## How an experiment is organised

```mermaid
flowchart LR
    D[Public data<br/>Digits, MNIST] --> P[Partition<br/>IID or label skew]
    P --> T[Flower training<br/>Docker, 3 to 4 clients]
    T --> R[Receipts, hashes,<br/>topology attestation]
    T --> V[Protected views<br/>plaintext, SecAgg+, DP]
    V --> A[Attacks<br/>LiRA, update inversion]
    R --> G[Acceptance gates<br/>parity, replay, fail-fast]
    A --> E[Saved evidence<br/>reports/results]
    G --> E
    E --> B[Dashboard and docs]
```

A run counts as evidence only if its gates pass. Failed gates are recorded and reported, not deleted.

## Repository layout

```
src/feddisclosure/    library: analytics, attacks, defenses, federation, training, evaluation
services/             three-site analytics gateway and client services
scripts/              experiment runners, verifiers, plotting and report builders
configs/              experiment, attack and defense configurations
docker/, compose*.yaml  container images and Compose topologies
tests/                unit, integration, privacy, training and regression tests
reports/results/      saved evidence (JSON, logs, hashes, receipts)
dashboard/            offline evidence explorer
docs/                 protocols, results, threat models, decisions, provenance
```

Project status files at the repo root (`CURRENT_STATE.md`, `NEXT_SPRINT.md`, `PROJECT_CONTEXT.md`, `AGENTS.md`) record what has been verified and what remains open. The pre-public README with the full milestone history is kept in [`docs/history/README-legacy.md`](docs/history/README-legacy.md).

## Status and honest limits

- **Not a privacy product.** No differential privacy, secure aggregation or security certification is claimed. Epsilons are accountant outputs for specific runs.
- **Inversion study is an observation-model study.** The SecAgg+ view uses the upstream quantiser arithmetic on single-step updates, not captured Docker traffic. The attacker is naive and not DP-aware.
- **No proven multi-round SecAgg+ bound.** Two-round error is bimodal because of ReLU flips. Only an empirical percentile rule is offered.
- **Scale.** Docker topologies were tested at 3 and 4 clients on a 2 CPU machine.
- **Data.** Digits and MNIST only. The MNIST originators publish no licence text, so redistribution terms are treated as unverified and the raw files are not in this repo.
- **Hosted CI.** The workflows had not run on GitHub-hosted runners before this repository was published. Their status is shown by the badges above and discussed in [`CHANGELOG.md`](CHANGELOG.md).

## Contributing

Contributions that add experiments, attackers, datasets or fixes are welcome. The one hard rule is that acceptance gates are never weakened to make a result pass and negative results are preserved. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately as described in [`SECURITY.md`](SECURITY.md).

## Citing

If you use this work, please cite it using [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button). There is no DOI or peer-reviewed paper yet.

## License

Code is released under the [Apache License 2.0](LICENSE). Third-party data and software keep their own terms, see [`NOTICE`](NOTICE).
