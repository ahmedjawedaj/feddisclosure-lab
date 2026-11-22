# Contributing to FedDisclosure Lab

Thanks for helping. This is a research benchmark, so the rules protect the evidence as much as the code.

## The rules that matter most

1. **Never weaken an acceptance gate to make a result pass.** If a gate fails, record the failure, explain it, and add a new, clearly named gate if you think the old one is wrong. Do not edit tolerances in place.
2. **Preserve negative results.** Failed runs, withdrawn claims and corrections stay in the history and the docs.
3. **No mock privacy.** Test doubles are fine in unit tests but must never be reported as OpenDP, Opacus or SecAgg+ evidence.
4. **Synthetic or public data only.** No private or personal data, ever.
5. **Say what was and was not run.** State the topology, versions, seeds and whether it ran natively, in simulation or only as a static check.
6. **No secrets, tokens or personal paths** in commits or logs.
7. **No AI assistant as author or co-author on commits.** See the commit authorship rule in `AGENTS.md`. CI enforces it.

## Getting set up

```bash
git clone https://github.com/ahmedjawedaj/feddisclosure-lab.git
cd feddisclosure-lab
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test,research]'
PYTHONPATH=src:. pytest -q -m 'not slow'
```

Docker runs need Docker Engine with Compose v2. Pinned versions for native runs are in [`docs/v0.14-reproduce.md`](docs/v0.14-reproduce.md).

## Workflow

- Fork the repo and branch from `main`. Use a descriptive branch name such as `feature/dp-aware-attacker`.
- Keep changes small and deterministic. Fix seeds and configs, and put new experiments under `configs/`.
- Add or update tests. The fast suite must stay green with `pytest -q -m 'not slow'`.
- New evidence goes under `reports/results/` with hashes, the exact command, and a short note in `docs/`.
- Update `CURRENT_STATE.md` if what is verified changes, and add a line to `CHANGELOG.md`.
- Open a pull request using the template. Explain what you ran and what you did not.
- Do not force-push shared branches.

## Reporting problems

- Bugs and questions: open an issue with the templates.
- Suspected security problems: follow [`SECURITY.md`](SECURITY.md) and do not post details publicly.
- Everyone follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

By contributing you agree that your contributions are licensed under the Apache License 2.0.
