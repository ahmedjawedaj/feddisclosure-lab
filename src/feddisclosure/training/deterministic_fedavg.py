"""Order-independent FedAvg for Flower 1.27 (opt-in).

Upstream `flwr.serverapp.strategy.FedAvg` sums float32 `array * weight` in the order the replies
arrive. Node ids are random per registration and replies arrive in completion order, so that sum
is not order stable and floating point addition is not associative. `DeterministicFedAvg` sorts
the replies by the SITE ID that each client reports, then averages with the same arithmetic as the
repository reference `feddisclosure.training.task.fedavg_weighted`: float64 accumulation of
`array * (weight / total)` in site order, cast once back to the array dtype.

Scope and limits, stated plainly:
  * It fails closed. A reply without a site id, a malformed site id or two replies with the same
    site id raise `SiteIdError`. Nothing is guessed from node ids.
  * It only replaces the strategy's own aggregation. The SecAgg+ path does not use the strategy
    aggregation, so this class has no effect there.
  * With this strategy the float64 parity gate no longer tests Flower's aggregation arithmetic.
    It tests transport and client determinism, because the server arithmetic is now the reference
    arithmetic by construction.
  * Reordering by site id is a determinism measure, not a privacy measure.

The client has to report its site id. See `site_record` and docs/v0.15-reply-order.md.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import os

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

AGGREGATION_KEY = "aggregation"
AGGREGATION_ENV = "FDL_AGGREGATION"
UPSTREAM = "upstream"
DETERMINISTIC = "deterministic"
SITE_RECORD_KEY = "site"
SITE_ID_KEY = "site-id"


class SiteIdError(ValueError):
    """A reply carried no usable site id, or two replies claimed the same one."""


def resolve_aggregation_mode(run_config: Mapping | None = None, environ: Mapping[str, str] | None = None) -> str:
    """`upstream` (default) or `deterministic`. The run-config key or the env var can enable it.

    An unknown value is an error, so a typo cannot silently leave the upstream arithmetic in use.
    """
    environ = os.environ if environ is None else environ
    values = []
    if run_config is not None and AGGREGATION_KEY in run_config:
        values.append(("run-config", str(run_config[AGGREGATION_KEY])))
    if environ.get(AGGREGATION_ENV):
        values.append(("environment", environ[AGGREGATION_ENV]))
    for source, value in values:
        if value not in (UPSTREAM, DETERMINISTIC):
            raise ValueError(f"Unknown aggregation mode {value!r} from {source}. Use {UPSTREAM!r} or {DETERMINISTIC!r}")
    return DETERMINISTIC if any(v == DETERMINISTIC for _, v in values) else UPSTREAM


def site_record(site_id: int) -> ConfigRecord:
    """The record a ClientApp adds to its train reply: `RecordDict({..., "site": site_record(id)})`.

    A ConfigRecord is not a MetricRecord, so it is not averaged into the reported train metrics
    and it passes the upstream reply consistency check.
    """
    if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
        raise SiteIdError("site id must be a non-negative integer")
    return ConfigRecord({SITE_ID_KEY: int(site_id)})


def reply_site_id(content: RecordDict) -> int:
    """Read the site id from a train reply. Raises SiteIdError if it is absent or malformed."""
    found = [rec[SITE_ID_KEY] for rec in content.config_records.values() if SITE_ID_KEY in rec]
    if len(found) != 1:
        raise SiteIdError(f"reply must carry exactly one {SITE_ID_KEY!r}, found {len(found)}")
    value = found[0]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SiteIdError(f"{SITE_ID_KEY!r} must be a non-negative integer, got {value!r}")
    return int(value)


def sorted_by_site(site_ids: Sequence[int]) -> list[int]:
    """Indices that put the replies in ascending site id order. Raises on duplicates."""
    if len(set(site_ids)) != len(site_ids):
        raise SiteIdError(f"duplicate site ids in replies: {sorted(site_ids)}")
    return sorted(range(len(site_ids)), key=site_ids.__getitem__)


def deterministic_weighted_average(arrays: Sequence[Mapping[str, np.ndarray]],
                                   weights: Sequence[float]) -> dict[str, np.ndarray]:
    """Reference arithmetic, in the order given: sum of float64(array) * (weight / total), cast once.

    This is the same formula as `task.fedavg_weighted` (float64 accumulation, one cast). Callers
    pass the replies already sorted by site id.
    """
    if not arrays or len(arrays) != len(weights):
        raise ValueError("Nonempty updates and matching weights required")
    if any(not math.isfinite(float(w)) or float(w) <= 0 for w in weights):
        raise ValueError("Weights must be positive and finite")
    keys = list(arrays[0])
    if any(list(a) != keys for a in arrays):
        raise ValueError("Incompatible model keys")
    total = sum(float(w) for w in weights)
    factors = [float(w) / total for w in weights]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        parts = [np.asarray(a[key]) for a in arrays]
        if any(p.shape != parts[0].shape or p.dtype != parts[0].dtype for p in parts):
            raise ValueError("Incompatible tensor shapes or dtypes")
        acc = None
        for part, factor in zip(parts, factors):
            term = part.astype(np.float64) * factor
            acc = term if acc is None else acc + term
        out[key] = np.ascontiguousarray(acc.astype(parts[0].dtype))
    return out


def upstream_weighted_average(arrays: Sequence[Mapping[str, np.ndarray]], weights: Sequence[float],
                              order: Iterable[int] | None = None) -> dict[str, np.ndarray]:
    """Arithmetic of flwr 1.27 `aggregate_arrayrecords`, replies taken in `order` (reply order).

    float32 `array * (weight / total)` accumulated in place. Kept here so tests and the replay
    study can compare against it. A test checks it against the real library function.
    """
    order = list(range(len(arrays))) if order is None else list(order)
    total = sum(weights)
    out: dict[str, np.ndarray] = {}
    for idx in order:
        factor = weights[idx] / total
        for key, value in arrays[idx].items():
            if key not in out:
                out[key] = np.asarray(value) * factor
            else:
                out[key] += np.asarray(value) * factor
    return {k: np.asarray(v) for k, v in out.items()}


class DeterministicFedAvg(FedAvg):
    """FedAvg whose train aggregation does not depend on reply order. See the module docstring."""

    def aggregate_train(self, server_round: int, replies: Iterable[Message]
                        ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid:
            return None, None
        # Fail closed before any arithmetic: every reply needs a distinct site id.
        site_ids = [reply_site_id(msg.content) for msg in valid]
        ordered = [valid[i].content for i in sorted_by_site(site_ids)]
        arrays_in, weights = [], []
        for content in ordered:
            record = next(iter(content.array_records.values()))
            arrays_in.append({k: v.numpy() for k, v in record.items()})
            weights.append(float(next(iter(content.metric_records.values()))[self.weighted_by_key]))
        averaged = deterministic_weighted_average(arrays_in, weights)
        arrays = ArrayRecord({k: Array(v) for k, v in averaged.items()})
        # Metrics are only reported numbers, but sorting them too makes the summary reproducible.
        metrics = self.train_metrics_aggr_fn(ordered, self.weighted_by_key)
        return arrays, metrics
