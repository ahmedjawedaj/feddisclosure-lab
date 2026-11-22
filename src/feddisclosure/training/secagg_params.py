"""Validation of the SecAgg+ group parameters. Pure Python, no Flower import.

Flower's SecAggPlusWorkflow builds, for every client, a neighbourhood of `num_shares` clients
arranged on a ring (the client itself and `num_shares // 2` neighbours on each side). Each client
secret-shares its masking seed over its neighbourhood, and the server needs `reconstruction_threshold`
live neighbours to unmask a client. With `num_shares == clients` the neighbourhood is everyone, which
is the original behaviour of this repository. With fewer shares than clients the neighbourhoods are
proper subsets, which is the setting SecAgg+ was designed for at larger client counts.

Rules, checked before the workflow is built:

* at least 3 clients
* `2 <= threshold <= shares <= clients`
* shares of 3 or more (upstream rejects smaller integers)
* when `shares < clients`, shares must be odd and `threshold < shares`. Upstream silently raises an
  even share count by one and refuses a threshold equal to the share count, so both would otherwise
  run a different configuration from the one written down.

When `shares == clients` the rule is exactly the one this repository always had, so existing
four-client and three-client runs behave as before.

Security note, not enforced here: for the neighbourhood construction to resist a malicious server,
the threshold should exceed half the neighbourhood size. Small thresholds favour dropout tolerance.
"""
from __future__ import annotations


def validate_secagg_params(clients: int, shares: int, threshold: int) -> None:
    if clients < 3 or not 2 <= threshold <= shares <= clients:
        raise ValueError("SecAgg+ requires >=3 clients and 2 <= threshold <= num-shares <= clients "
                         f"(got clients={clients}, num-shares={shares}, reconstruction-threshold={threshold})")
    if shares < clients:
        if shares < 3:
            raise ValueError(f"num-shares must be at least 3 (got {shares})")
        if shares % 2 == 0:
            raise ValueError(f"num-shares below the client count must be odd (got {shares}), "
                             "because the upstream workflow would silently raise it by one")
        if threshold >= shares:
            raise ValueError(f"reconstruction-threshold must be below num-shares when num-shares < clients "
                             f"(got {threshold} and {shares})")

