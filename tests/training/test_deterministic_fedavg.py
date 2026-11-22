"""DeterministicFedAvg: order invariance, reference equality, fail-closed site ids, opt-in default."""
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("flwr")
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from feddisclosure.training import flower_server
from feddisclosure.training.deterministic_fedavg import (
    DETERMINISTIC, UPSTREAM, DeterministicFedAvg, SiteIdError, deterministic_weighted_average,
    reply_site_id, resolve_aggregation_mode, site_record, upstream_weighted_average)
from feddisclosure.training.task import fedavg_weighted

SIZES = [96, 100, 90, 70]          # unequal, so weights are not exact binary fractions


def _states(seed=0, clients=4):
    rng = np.random.default_rng(seed)
    return [{"w": rng.normal(size=(40,)).astype("float32"), "b": rng.normal(size=(5, 3)).astype("float32")}
            for _ in range(clients)]


def _reply(state, size, site_id="absent"):
    parts = {"arrays": ArrayRecord({k: Array(v) for k, v in state.items()}),
             "metrics": MetricRecord({"num-examples": size, "train_loss": 0.5})}
    if site_id != "absent":
        parts["site"] = ConfigRecord({"site-id": site_id})
    return Message(content=RecordDict(parts), dst_node_id=1, message_type="train")


def _aggregate(messages):
    return DeterministicFedAvg().aggregate_train(1, messages)


def test_upstream_arithmetic_is_reply_order_dependent_on_a_constructed_float32_example():
    states = _states()
    results = {upstream_weighted_average(states, SIZES, o)["w"].tobytes() for o in itertools.permutations(range(4))}
    assert len(results) > 1   # float32 addition is not associative, the sum depends on reply order


def test_upstream_replica_matches_the_real_library_function_for_every_order():
    states = _states(3)
    records = [RecordDict({"arrays": ArrayRecord({k: Array(v) for k, v in s.items()}),
                           "metrics": MetricRecord({"num-examples": n})}) for s, n in zip(states, SIZES)]
    for order in itertools.permutations(range(4)):
        library = aggregate_arrayrecords([records[i] for i in order], "num-examples")
        replica = upstream_weighted_average(states, [float(n) for n in SIZES], order)
        assert all(np.array_equal(library[k].numpy(), replica[k]) for k in replica)


def test_deterministic_result_is_bitwise_identical_for_all_arrival_orders():
    states = _states(1)
    replies = [_reply(s, n, site) for site, (s, n) in enumerate(zip(states, SIZES))]
    digests = set()
    for order in itertools.permutations(range(4)):
        arrays, metrics = _aggregate([replies[i] for i in order])
        digests.add(b"".join(arrays[k].numpy().tobytes() for k in sorted(arrays)))
        assert metrics["train_loss"] == pytest.approx(0.5)
    assert len(digests) == 1


def test_deterministic_equals_repository_reference_formula_bitwise_and_float64_mean_closely():
    states = _states(2)
    replies = [_reply(s, n, site) for site, (s, n) in enumerate(zip(states, SIZES))]
    arrays, _ = _aggregate(list(reversed(replies)))
    reference = fedavg_weighted([{k: torch.from_numpy(v) for k, v in s.items()} for s in states], SIZES)
    for key in reference:
        assert np.array_equal(arrays[key].numpy(), reference[key].numpy())          # tolerance 0
        exact = sum(s[key].astype(np.float64) * (n / sum(SIZES)) for s, n in zip(states, SIZES))
        assert np.max(np.abs(arrays[key].numpy() - exact)) <= 1e-6                  # one float32 rounding at most
        assert arrays[key].numpy().dtype == np.float32


def test_site_id_not_arrival_position_decides_the_summation_order():
    states = _states(4)
    # Site ids 3,1,0,2 arrive in that order. Result must equal the sum taken in site order 0,1,2,3.
    sites = [3, 1, 0, 2]
    arrays, _ = _aggregate([_reply(states[i], SIZES[i], sites[i]) for i in range(4)])
    by_site = sorted(range(4), key=sites.__getitem__)
    expected = deterministic_weighted_average([states[i] for i in by_site], [SIZES[i] for i in by_site])
    assert all(np.array_equal(arrays[k].numpy(), expected[k]) for k in expected)


@pytest.mark.parametrize("bad", ["absent", True, -1, "2", 2.0])
def test_fails_closed_on_missing_or_malformed_site_id(bad):
    states = _states(5, clients=3)
    replies = [_reply(states[0], 10, 0), _reply(states[1], 10, 1), _reply(states[2], 10, bad)]
    with pytest.raises(SiteIdError):
        _aggregate(replies)


def test_fails_closed_on_duplicate_site_ids():
    states = _states(6, clients=3)
    with pytest.raises(SiteIdError, match="duplicate"):
        _aggregate([_reply(states[0], 10, 0), _reply(states[1], 10, 1), _reply(states[2], 10, 1)])


def test_site_record_round_trips_and_rejects_bad_ids():
    assert reply_site_id(RecordDict({"site": site_record(3)})) == 3
    for bad in (-1, True, "1"):
        with pytest.raises(SiteIdError):
            site_record(bad)


def test_reply_site_id_rejects_two_site_ids_in_one_reply():
    content = RecordDict({"a": ConfigRecord({"site-id": 1}), "b": ConfigRecord({"site-id": 2})})
    with pytest.raises(SiteIdError):
        reply_site_id(content)


def test_mode_default_is_upstream_and_unknown_values_fail():
    assert resolve_aggregation_mode({}, {}) == UPSTREAM
    assert resolve_aggregation_mode({"aggregation": "upstream"}, {}) == UPSTREAM
    assert resolve_aggregation_mode({"aggregation": "deterministic"}, {}) == DETERMINISTIC
    assert resolve_aggregation_mode({}, {"FDL_AGGREGATION": "deterministic"}) == DETERMINISTIC
    with pytest.raises(ValueError):
        resolve_aggregation_mode({"aggregation": "determinstic"}, {})


class _Recorder:
    created = None

    def __init__(self, **kwargs):
        type(self).created = self

    def start(self, **kwargs):
        state = {"network.0.weight": torch.zeros(1)}
        return SimpleNamespace(arrays=SimpleNamespace(to_torch_state_dict=lambda: state),
                               train_metrics_clientapp={}, evaluate_metrics_clientapp={})


def _run_server(monkeypatch, tmp_path, extra=None, env=None):
    class Upstream(_Recorder):
        pass

    class Det(_Recorder):
        pass
    monkeypatch.setattr(flower_server, "FedAvg", Upstream)
    monkeypatch.setattr(flower_server, "DeterministicFedAvg", Det)
    monkeypatch.delenv("FDL_AGGREGATION", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    cfg = {"seed": 1, "features": 64, "classes": 10, "clients": 3, "learning-rate": 0.1,
           "num-server-rounds": 1, "artifact-dir": str(tmp_path), **(extra or {})}
    flower_server.main(None, SimpleNamespace(run_config=cfg))
    return Upstream, Det, json.loads((tmp_path / "flower_summary.json").read_text())


def test_server_default_uses_upstream_strategy_and_summary_is_unchanged(monkeypatch, tmp_path):
    up, det, summary = _run_server(monkeypatch, tmp_path)
    assert up.created is not None and det.created is None
    assert "aggregation" not in summary


@pytest.mark.parametrize("extra,env", [({"aggregation": "deterministic"}, None), ({}, {"FDL_AGGREGATION": "deterministic"})])
def test_server_opt_in_switches_strategy_and_is_recorded(monkeypatch, tmp_path, extra, env):
    up, det, summary = _run_server(monkeypatch, tmp_path, extra, env)
    assert det.created is not None and up.created is None
    assert "DeterministicFedAvg" in summary["aggregation"]
