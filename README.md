# FedDisclosure Lab

**A reproducible privacy leakage and defense benchmark for federated analytics and learning**

Status: **Sprint 0 built, Sprint 1 core implemented and locally integration-tested, Sprints 2–6 planned**. Updated 2026-11-18.

This is an independent research-engineering project using **only synthetic data**. No thesis novelty, production security, differential privacy or secure aggregation is claimed at this stage.

## What runs today

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
| 2: Flower + PyTorch | Planned | 3-client FedAvg, IID/non-IID, local/pooled comparisons |
| 3: Leakage attacks | Planned | Reproduce gradient inversion + membership inference |
| 4: Defenses | Planned | OpenDP, Opacus, Flower SecAgg+ under matched threat models |
| 5: Benchmarks | Planned | Multi-seed ablations, accuracy/leakage/efficiency figures |
| 6: Research release | Planned | Dashboard, report, clean-install & faculty presentation |

See `docs/roadmap.md` for sprint tasks and acceptance criteria. The `training`, `federation`, `defenses`, `dashboard` and remaining service modules contain planned integration points, not working implementations.

## Scope boundaries

No external confidential datasets, real hospital connectors, cryptographic protocol reimplementations, public production deployment, novel-method claims or validated digital twin in v1. Prototype demonstrations are not evidence of comprehensive security.

License: Apache-2.0. Research content and experimental code are provided without warranty.

## Existing prototype migration note

The former starter project was documented as a single-process synthetic SQLite demonstration. The earlier starter source archive was not accessible in this working environment, so this repository is a clean implementation that reproduces the documented 110/107/3 benchmark. It should not be described as a line-by-line migration.

## First figure

Run `python scripts/plot_analytics.py` after installing the optional `visual` dependency group. The generated `reports/figures/sprint1_analytics.png` truthfully distinguishes a withheld response from a count of zero.
