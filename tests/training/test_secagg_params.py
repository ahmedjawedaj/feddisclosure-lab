"""SecAgg+ group parameters: fewer shares than clients (v0.15, WP9). No Docker and no network."""
import itertools

import pytest

from feddisclosure.training.secagg_params import validate_secagg_params


def _legacy_ok(clients, shares, threshold):
    """The rule that secagg_server.main enforced before this change."""
    return not (clients < 3 or shares != clients or not 2 <= threshold <= shares)


def _accepts(*args):
    try:
        validate_secagg_params(*args)
    except ValueError:
        return False
    return True


def test_default_behaviour_is_unchanged_when_shares_equal_clients():
    for clients, threshold in itertools.product(range(0, 12), range(0, 14)):
        assert _accepts(clients, clients, threshold) == _legacy_ok(clients, clients, threshold)


def test_everything_the_old_rule_accepted_is_still_accepted():
    for clients, shares, threshold in itertools.product(range(0, 12), range(0, 14), range(0, 14)):
        if _legacy_ok(clients, shares, threshold):
            assert _accepts(clients, shares, threshold)


def test_more_shares_than_clients_is_still_rejected():
    for clients in range(3, 10):
        assert not _accepts(clients, clients + 1, 2)


@pytest.mark.parametrize("clients,shares,threshold", [(8, 7, 4), (8, 5, 3), (8, 3, 2), (16, 7, 4), (9, 5, 2), (8, 8, 5)])
def test_valid_neighbourhood_settings_are_accepted(clients, shares, threshold):
    validate_secagg_params(clients, shares, threshold)


@pytest.mark.parametrize("clients,shares,threshold,why", [
    (8, 4, 2, "odd"),           # even share count, upstream would silently raise it to 5
    (8, 6, 3, "odd"),
    (8, 2, 2, "at least 3"),
    (8, 1, 1, "2 <= threshold"),
    (8, 7, 7, "below num-shares"),   # threshold equal to shares, upstream refuses it
    (8, 5, 5, "below num-shares"),
    (8, 7, 1, "2 <= threshold"),
    (8, 9, 4, "2 <= threshold"),     # more shares than clients
    (2, 2, 2, "requires"),           # fewer than three clients
    (8, 5, 0, "2 <= threshold"),
])
def test_invalid_combinations_are_rejected_with_a_reason(clients, shares, threshold, why):
    with pytest.raises(ValueError, match=why):
        validate_secagg_params(clients, shares, threshold)


def test_validator_agrees_with_the_upstream_constructor_for_fewer_shares_than_clients():
    pytest.importorskip("flwr")
    from flwr.server.workflow import SecAggPlusWorkflow

    def upstream_ok(shares, threshold):
        try:
            SecAggPlusWorkflow(num_shares=shares, reconstruction_threshold=threshold, max_weight=1000.0, timeout=10.0)
        except (ValueError, TypeError):
            return False
        return True

    clients = 8
    for shares, threshold in itertools.product(range(3, clients), range(1, 9)):
        # Upstream accepts an even share count at construction and adjusts it later, so only the
        # cases upstream refuses outright must also be refused here, and every valid one accepted.
        if _accepts(clients, shares, threshold):
            assert upstream_ok(shares, threshold), (shares, threshold)
        if not upstream_ok(shares, threshold):
            assert not _accepts(clients, shares, threshold), (shares, threshold)


def _context(**overrides):
    cfg = {"clients": 8, "num-shares": 7, "reconstruction-threshold": 4, "seed": 0, "features": 64, "classes": 10,
           "num-server-rounds": 1, "max-weight": 1000.0, "timeout": 10.0, "use-secagg": True, "artifact-dir": "unused"}
    cfg.update(overrides)

    from flwr.common import Context, RecordDict
    return Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config=cfg)


def test_server_rejects_bad_parameters_before_touching_the_grid():
    pytest.importorskip("flwr")
    from feddisclosure.training import secagg_server as server
    for bad in ({"num-shares": 4}, {"reconstruction-threshold": 7}, {"num-shares": 9}, {"clients": 2, "num-shares": 2}):
        with pytest.raises(ValueError, match="SecAgg\\+ requires|num-shares|reconstruction-threshold"):
            server.main(object(), _context(**bad))


def test_server_hands_the_smaller_share_count_to_the_upstream_workflow(monkeypatch):
    pytest.importorskip("flwr")
    from feddisclosure.training import secagg_server as server

    class Reached(Exception):
        pass

    seen = {}

    def stop(fit_workflow=None):
        seen["shares"], seen["threshold"] = fit_workflow.num_shares, fit_workflow.reconstruction_threshold
        raise Reached
    monkeypatch.setattr(server, "DefaultWorkflow", stop)
    with pytest.raises(Reached):
        server.main(object(), _context())
    assert seen == {"shares": 7, "threshold": 4}


def test_four_client_default_still_reaches_the_workflow_with_equal_shares(monkeypatch):
    pytest.importorskip("flwr")
    from feddisclosure.training import secagg_server as server

    class Reached(Exception):
        pass

    seen = {}

    def stop(fit_workflow=None):
        seen["shares"] = fit_workflow.num_shares
        raise Reached
    monkeypatch.setattr(server, "DefaultWorkflow", stop)
    with pytest.raises(Reached):
        server.main(object(), _context(clients=4, **{"num-shares": 4, "reconstruction-threshold": 2}))
    assert seen["shares"] == 4
