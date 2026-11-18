# NEXT SPRINT — Close Sprint 1 and implement Sprint 2

## Current priority (ordered)

1. **Compose proof:** On Docker-enabled CI or machine, run `docker compose config -q`, `up --build --wait`, trigger CLI benchmark, capture logs and results, `down -v`; assert network/volume isolation. Do not mark complete until witnessed.
2. **Sprint 1 guardrails:** Add site failure and concurrent query stress tests; audit-only metadata for queries without patient IDs; record privacy/utility limitations of rejection side channels.
3. **Document observer variants:** Separate external analytics user vs gateway operator who sees per-site aggregates. Current attack assumes external user only.
4. **Sprint 2 implementation:** Add proper Flower ServerApp/ClientApp with PyTorch, independent site-owned MNIST partitions (download once then split into explicit artifacts), local-only and pooled synthetic/public references, IID and non-IID scenario configs.
5. **Sprint 2 testing:** verify no raw sample crosses site boundary; compare one-round FedAvg numerically with a reference implementation; write seed/accuracy/communication metrics and add a CI smoke run.

## Sprint 2 definition of done

Three genuine independent Flower clients train and aggregate a simple PyTorch model under a versioned config, recording validation accuracy, worst client, rounds and communication. The pooled baseline may use public data only in evaluator and must be identified as a *reference*, not as the deployed federated configuration. Docker execution is validated on a machine supporting containers.

## Discipline

No dashboard-first detours, private datasets, homemade secure aggregation or claiming formal privacy from the history rule. If a new dependency is incompatible with Python 3.13, isolate ML in Python 3.11/3.12 without breaking the core package. Keep sprint evidence and all test outcomes in `CURRENT_STATE.md`.
