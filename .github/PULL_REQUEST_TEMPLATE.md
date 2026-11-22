## What changed and why

<!-- One or two sentences. Link the issue if there is one. -->

## What I ran

<!-- Exact commands, topology (Docker or simulation), versions and seeds. Say what you did NOT run. -->

## Checklist

- [ ] `PYTHONPATH=src:. pytest -q -m 'not slow'` passes
- [ ] No acceptance gate or tolerance was weakened. Any new gate is named and explained
- [ ] Negative or failed results are recorded, not removed
- [ ] No mock or test-double result is presented as OpenDP, Opacus or SecAgg+ evidence
- [ ] Only synthetic or public data. No secrets or personal paths
- [ ] Docs, `CURRENT_STATE.md` and `CHANGELOG.md` updated where relevant
