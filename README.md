# v0.8.0 research milestone

**Executed:** five seeds × individual/aggregate-only/colluding observer roles = 15 actual public-digit update inversion attacks, with a saved multi-seed figure and independent integrity verifier. Mean target MSE: 0.01664 / 0.10848 / 0.01664. The aggregate case is an idealized mean and uses a deliberately weak target-only probe; the all-but-one collusion case algebraically reveals the individual update. **No Flower SecAgg+ was executed.**

**Implemented but native blocked:** local Opacus DP-SGD vs ordinary SGD code, accountant checks, CLI, config, tests and GitHub CI. No real native Opacus output or epsilon guarantee is available from this environment. Docker, Flower and native OpenDP are also unverified.

To verify: `python scripts/verify_aggregate_collusion.py`, `pytest -q -m 'not slow'`. See `docs/v0.8-research-protocol.md`, `docs/v0.8-results.md`, `CURRENT_STATE.md`, `NEXT_SPRINT.md` and the original setup instructions below.

---

# FedDisclosure Lab

**A reproducible privacy leakage and defense benchmark for federated analytics and learning**

Status: **v0.7.0: 20 matched equal-work public-data FedAvg runs executed; OpenDP HTTP/ledger adapter tested with fake mechanism, native OpenDP/Docker/Flower still NOT VERIFIED**. Updated 2026-11-19.

This is an independent research-engineering project using **only synthetic or approved public data**. No thesis novelty, production security, differential privacy or secure aggregation is claimed at this stage.

## v0.7.0 equal-work federation and privacy release controls

- **20 executed experiments**: five matched seeds (241–245) × IID/balanced Dirichlet α 0.1, 0.5 and 10.0, four rounds and identical 192 train/48 validation examples per client. Training uses separate PyTorch worker processes; not native Flower.
- **Mean global accuracies:** 68.89% IID, 71.22% α=0.1, 67.67% α=0.5, 71.56% α=10. **Mean worst-client validation:** 65.00%, 53.33%, 60.00%, 65.42%, respectively. Exploratory, not statistically significant; α differences are not monotonic.
- **Privacy-defense implementation:** site-local OpenDP Laplace count adapter, persistent epsilon ledger, DP-only `/v1/dp-count`, federated aggregate `/v1/federated-dp-count`, protection against raw-route bypass and restarts to/from unsafe database history. Tests use a **fake measurement** to check application behavior. Native OpenDP could not run and no privacy guarantee is claimed for these tests.
- Protocol and results: `docs/v0.7-pre-registered-protocol.md`, `docs/v0.7-research-report.md` and `docs/opendp-runtime.md`. Integrity verification: `python scripts/verify_v070_evidence.py`. Figure: `reports/figures/sprint5_balanced_heterogeneity.png`.
- Native test workflow files for OpenDP and balanced federation are included but have not executed on GitHub. Current release check: 76 fast tests passed, 2 skipped, 2 slow integration tests passed independently, 20 checkpoint hashes verified. Unsplit full-suite timed out and is not counted as passing. Docker, Flower, OpenDP, Opacus, SecAgg+ remain open gates.

### Reproduce v0.7 benchmarks

```bash
python -m pip install -e '.[research,test]'  # public data only
python scripts/run_balanced_heterogeneity.py --seeds 241 242 243 244 245 --rounds 4
python scripts/plot_balanced_heterogeneity.py
python scripts/verify_v070_evidence.py
pytest -q -m 'not slow'
```

### Native OpenDP on a capable host (not executed here)

```bash
python -m pip install -e '.[privacy,test]'
python scripts/run_dp_analytics.py --epsilon 0.25 --total-budget 1.0
```

**Do not deploy both raw and DP query modes to the same analytics user.** Use fresh site volumes, run the real OpenDP map test, and validate assumptions before citing privacy guarantees. See `docs/opendp-runtime.md`.

## v0.6.0 verified research extensions — 19 November 2026

- Measured gradient-matching attacks from **changing disjoint SGD minibatches**, using one observed individual-client model delta, 8 or 32 different public digit examples and privileged labels. The four-step runs have no intermediate-gradient visibility.
- Compared three independently trained FedAvg checkpoints under output-only model access. Three predeclared membership scores × three training seeds produced nine ROC-AUC results, mostly near chance.
- Executed matched eight-round severe label-skew **FedAvg vs FedProx** for seed241, plus an eight-round moderate Dirichlet(alpha=0.5) FedAvg experiment. The moderate run has **unequal site sizes/optimizer steps**; see the research report for confounders.
- Added model-parameter parity verification for the real Flower app, fail-closed config/hash checks, a Flower CI gate, reproducible plots, two additional agent skills for Codex/Claude, and evidence verification.
- **Native Flower, Docker Compose, OpenDP, Opacus, SecAgg+, external MNIST and Colab remain unexecuted/unverified here.**

Read `docs/v0.6-research-report.md`, `docs/v0.6-experiment-protocol.md`, `CURRENT_STATE.md` and `NEXT_SPRINT.md`. To verify the full evidence bundle, run `python scripts/verify_v060_evidence.py` from the extracted repository root.

## Historical v0.4.1 measured update — 19 November 2026

- **Analytics:** live three-institution/four-HTTP-process experiment still reproduces a 3-person synthetic disclosure (`110-107`) and a blocked second query under the history heuristic.
- **Training:** 5 paired seeds × IID/severe label skew (10 real three-client PyTorch reference runs), plus 1/5-client single-seed pilots. These runs are **NOT native Flower** and do not justify a convergence claim.
- **Disclosure:** exact-gradient inversion on public 8×8 digits, five matched exact/noisy examples and one batch-8 known-label case. Successful low-error reconstruction occurs with deliberately privileged gradients. No formally private noise mechanism is present.
- **Local model updates:** measured one-, four-, and sixteen-step actual SGD model-delta inversion on one repeatedly used public image; reconstruction degrades with the longer local-update schedule.
- **Membership:** output-only confidence-based baseline on the frozen public model; five query-seed AUCs are near chance. The attacker never receives membership labels; no safety guarantee follows from this negative result.
- **Verification:** 40 fast tests and both slow independent training integration tests passed as separate invocations; one Flower-dependent test skipped. Real local analytics passed again. Native Docker/Flower gates are still **blocked**, with recorded failed attempts.

The new work and scripts are described in `docs/disclosure-protocol.md`, `docs/fl-experiment-protocol.md`, `CURRENT_STATE.md`, and `NEXT_SPRINT.md`. The release includes actual JSON evidence and generated research figures. For the Python benchmark environment, install `pip install -e '.[research,test]'`; native Flower remains a separate `.[ml,test]` extra.

## Federated analytics runtime

Three separate FastAPI institutional processes, each with its own SQLite database, answer two allowlisted count queries through a federated gateway. An analytics observer sees only gateway-wide totals. A controlled differencing attack combines individually released counts, and a history-aware release policy blocks that particular attack. An independent scorer measures inference and query rejection.

**Expected measured result:** With the naive release policy, the observer sees 110 total records and 107 negative records, inferring a hidden subgroup of 3. With the history rule, the second query is rejected. See `reports/results/sprint1_analytics_verified.json`.

## Run without Docker (fully tested)

Python 3.11–3.13 is supported by the core package.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
python scripts/run_local_stack.py --once
```

The local runner starts **four real processes** (three institutions and one gateway) on dynamically assigned localhost ports, with distinct SQLite databases in temporary storage. It automatically stops the services and writes `reports/results/analytics_local.json`.

## Sprint 2: verified 3-client PyTorch reference and native Flower app

PyTorch + scikit-learn can execute the reproducible CPU-only three-worker baseline even when Flower cannot be installed:

```bash
python -m pip install -e '.[research,test]'  # on a network-enabled machine
python scripts/run_torch_reference.py --config configs/training/fedavg_digits_iid.yaml --data-dir data/training/digits-iid --output reports/results/sprint2/torch_iid_reference.json --references
python scripts/run_torch_reference.py --config configs/training/fedavg_digits_non_iid.yaml --data-dir data/training/digits-non-iid --output reports/results/sprint2/torch_non_iid_reference.json --references
python scripts/plot_training.py
```

This is **actual PyTorch training using 3 separate OS worker processes**, but **not Flower**. The privileged evaluator is a separate process that accesses *public* test data. Local-only and pooled references are explicitly evaluator-only and have different optimizer schedules. Tracked JSON results include measured training metrics, explicit limitations and SHA-256 references to final public-digit `.pt` checkpoints (the checkpoints are Git-ignored and included only in the downloadable release archive).

The native Flower 1.27 `ServerApp` and `ClientApp` are implemented against the official Message API, but could not be executed here because neither Flower nor package-network access was available:

```bash
python scripts/run_flower_experiment.py  # needs flwr[simulation] + its dependencies
```

This launches actual `flwr run` with 3 SuperNodes and writes failure/success logs, checkpoint and validation summary. **Native Flower remains an open runtime acceptance gate.** For real physically isolated institutions use independent SuperNodes on separate mounts/containers; the simulation is a protocol baseline, not strong OS isolation.

See `docs/fl-experiment-protocol.md`, `CURRENT_STATE.md`, and `NEXT_SPRINT.md`.

## Run with Docker Compose (configured, not verified here)

You must have Docker Engine and Compose v2. On the Docker host:

```bash
cp .env.example .env
# Replace BOTH demo placeholder tokens in .env with independent random strings.
docker compose up --build -d --wait
docker compose ps
docker compose logs -f
```

Only the gateway is published, at `127.0.0.1:8080`. Institutional services are reachable only on the private Compose network and hold separate SQLite volumes. Run the benchmark from the host:

```bash
export FDL_RESEARCH_KEY='the-same-key-as-in-.env'
python -m feddisclosure.cli analytics-benchmark --gateway http://127.0.0.1:8080
```

On a Docker host, use `bash scripts/verify_compose.sh` to perform the full network/volume/port inspection and save benchmark/log evidence. This native execution was **not possible** in the build environment.

Stop with `docker compose down`. Add `-v` only when deliberately resetting research datasets and release histories. The `.env` file is gitignored. Do not expose this prototype to the public Internet; the API keys and experimental policy switch are not hardened production authorization.

## Reproducible research

```bash
python -m feddisclosure.cli analytics-benchmark \
  --gateway http://127.0.0.1:8080 \
  --config configs/analytics/differencing.yaml \
  --output reports/results/my_trial.json
```

Config records, attacker observations, outcomes, policies, runtimes, version information and limitations are saved in JSON. The randomized experiment ID ensures that persisted query histories do not contaminate subsequent tests. The evaluator can access synthetic ground truth; the attack module is restricted to published counts.

## Layout

```text
feddisclosure-lab/
├── src/feddisclosure/  # typed protocol, data, policy, attacks, evaluation, CLI
├── services/           # independent site and gateway APIs; future coordinator/evaluator
├── configs/            # versioned, YAML experiment specifications
├── scripts/            # four-process local integration harness
├── tests/              # unit, integration, privacy guards, regression
├── docs/               # research context, methodology, threat models, handoff
├── reports/            # measured JSON evidence, planned figures/report
├── .agents/skills/     # Codex repo skills
├── .claude/skills/     # Claude repo skills
├── AGENTS.md           # universal engineering/research instructions
├── CLAUDE.md           # Claude entry point
├── CURRENT_STATE.md    # verified milestone ledger
└── NEXT_SPRINT.md      # scoped next implementation plan
```

## Development and evidence loop

1. Start from `CURRENT_STATE.md`, `NEXT_SPRINT.md`, and `docs/threat-models.md`.
2. Make one falsifiable research hypothesis with an explicit observer and metric.
3. Implement the smallest baseline; add a failing regression test before fixing bugs.
4. Run `pytest -q` and `python scripts/run_local_stack.py --once`.
5. Save the config, result, version and limitations; mark **observed**, **unverified**, and **planned** separately.
6. Update state files and commit tested work. Never fabricate measurements.

## Seven sprint roadmap

| Sprint | Status | Gate |
|---|---|---|
| 0: Foundation | Implemented | Typed configs, threat models, tests, agent handoff |
| 1: Federated analytics | Core implemented; Docker pending | Three independent sites; count differencing and history rule |
| 2: Flower + PyTorch | Public-digits PyTorch reference executed; native Flower code written but not run | 3-client native Flower FedAvg, IID/non-IID, local/pooled comparisons |
| 3: Leakage attacks | First public-data baselines executed | Gradient inversion and output-only membership; multi-step update leakage remains |
| 4: Defenses | Planned | OpenDP, Opacus, Flower SecAgg+ under matched threat models |
| 5: Benchmarks | Planned | Multi-seed ablations, accuracy/leakage/efficiency figures |
| 6: Research release | Planned | Dashboard, report, clean-install & faculty presentation |

See `docs/roadmap.md` for sprint tasks and acceptance criteria. `training` now includes executed PyTorch reference code and a native Flower implementation pending runtime verification. `federation`, `defenses`, the dashboard and other service modules remain future work.

## Scope boundaries

No external confidential datasets, real hospital connectors, cryptographic protocol reimplementations, public production deployment, novel-method claims or validated digital twin in v1. Prototype demonstrations are not evidence of comprehensive security.

License: Apache-2.0. Research content and experimental code are provided without warranty.

## Existing prototype migration note

The former starter project was documented as a single-process synthetic SQLite demonstration. The earlier starter source archive was not accessible in this working environment, so this repository is a clean implementation that reproduces the documented 110/107/3 benchmark. It should not be described as a line-by-line migration.

## First figure

Run `python scripts/plot_analytics.py` after installing the optional `visual` dependency group. The generated `reports/figures/sprint1_analytics.png` truthfully distinguishes a withheld response from a count of zero.

## v0.5.0 measured extensions

- Multi-record unprotected client-model-delta reconstruction: batches 1, 8, 32; local SGD steps 1 or 4; five scored public-image attack cases with strict evaluator-only image truth and explicit label visibility. See `docs/disclosure-protocol.md`.
- Paired offline multi-process FedAvg versus FedProx (mu=0.3) on severe non-IID public digits, three seeds, four rounds, mean global test accuracy 55.19% versus 49.07% and zero worst-client validation for both. See `docs/fedprox-ablation.md` and `scripts/run_fedprox_ablation.py`.
- Added SHA-256 cross-check of six trained model checkpoints, run resume/manifest guards, two research figures, tests and updated Claude/Codex handoff. Verify with `python scripts/verify_research_evidence.py` after extracting the full evidence archive.
- **Not verified:** real Flower execution, Docker Compose, calibrated OpenDP/Opacus privacy, Flower SecAgg+, external MNIST. The word "federated" in offline references denotes separate training processes with no OS-enforced data isolation.
