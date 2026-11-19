# OpenDP count mode, native execution gate

**Status:** Implementation and budget/HTTP tests exist. The actual OpenDP mechanism has not executed in the current environment (missing package). No formal epsilon guarantee is claimed for an executed release until this gate passes on a Python 3.12 network-enabled machine.

OpenDP construction follows the documented `opendp.prelude` `m.make_laplace` scalar mechanism. The site's allowlisted count has add/remove-one sensitivity 1. The actual OpenDP privacy map is checked against the per-query target epsilon; failure stops service startup. The site uses `dp_budget` SQLite ledger independent of caller-provided experiment IDs and fixed-point accounting, transactionally reserved before native sampling. A protected site refuses the unprotected route and refuses a restart downgrade. The mode must start with *fresh volumes* without past raw releases; an earlier unprotected release already discloses information.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[privacy,test]'
pytest -q tests/privacy/test_dp_http.py tests/privacy/test_dp_ledger.py
python scripts/run_dp_analytics.py --epsilon 0.25 --total-budget 1.0
```

For Docker (unverified), use `docker compose -f compose.yaml -f compose.privacy.yaml up --build -d --wait` **only after provisioning new volumes**; never mix with previous unprotected data. `compose.privacy.yaml` requires native Docker and must be tested for configuration/volume isolation first.

## Interpretation

- An individual's presence can affect an allowlisted count by at most one under add/remove-one adjacency. A release of a single local count with a validated OpenDP Laplace privacy map can have a per-site pure DP bound. For two independent institutions holding disjoint populations, the total can use parallel composition *for that single release*. Repeat queries compose sequentially against the persistent budget; accountant reserves every local release.
- Current sites have a **single shared principal**. This demonstration does not implement real multi-tenant authorization. The operator must not expose a parallel raw endpoint or fresh untracked replicas to the same analytics adversary.
- The query ID is externally chosen among two fixed allowlisted queries, and no raw-count-dependent suppression is applied in DP mode.
- The gateway rejects partial-site results without returning any partial noisy values. A partial failure may still spend site budgets. This is conservative for accounting but not atomic across institutions.
- A noisy difference may still be close to the true subgroup count. A small observed error is not proof of DP failure; the bound limits adjacent-output distributions, not guaranteed per-instance reconstruction error.
- A successful *fake-mechanism* transport test demonstrates only application logic, never DP. The `opendp` import and its actual sampling / privacy-map test must pass.
- DP-SGD via Opacus, client-update protection, and secure aggregation are separate future milestones.

The SQLite ledger pins the total budget cap for each principal on first release and refuses later config-driven increases. All positive accounting tests are performed with an injected fake mechanism, not genuine OpenDP.
