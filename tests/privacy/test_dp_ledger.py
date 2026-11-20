"""Ledger accounting tests use a fake measurement; they are NOT DP certification."""
import sqlite3
import pytest
from feddisclosure.defenses.dp_release import BudgetExhausted, release_dp_at_site
from feddisclosure.defenses.opendp_count import OpenDPCount, DPLibraryUnavailable, DPConfigurationError
from feddisclosure.data.synthetic import initialize_database, InstitutionSpec


class FakeMeasurement:
    def __init__(self, error=None):
        self.observed = []
        self.error = error
    def release(self, count):
        self.observed.append(count)
        if self.error:
            raise self.error
        return count + 0.75


def init(tmp_path):
    db = str(tmp_path / "site.sqlite")
    initialize_database(db, InstitutionSpec("site", 40, 1))
    return db


def query(db, measure, query_id="all_records", eps=.25, cap=.5, principal="principal"):
    return release_dp_at_site(db, query_id=query_id, principal_scope=principal,
                              epsilon=eps, total_budget=cap, mechanism=measure)


def used(db):
    with sqlite3.connect(db) as c:
        try:
            row = c.execute("SELECT spent_micro FROM dp_budget WHERE principal_scope='principal'").fetchone()
        except sqlite3.OperationalError:
            return 0
        return row[0] if row else 0


def test_budget_persistent_and_experiment_id_cannot_reset_it(tmp_path):
    db = init(tmp_path)
    measure = FakeMeasurement()
    assert query(db, measure).noisy_count == 40.75
    assert query(db, measure, query_id="negative_records").noisy_count == 39.75
    assert used(db) == 500000
    with pytest.raises(BudgetExhausted):
        query(db, measure, query_id="all_records")
    assert len(measure.observed) == 2
    assert used(db) == 500000


def test_failed_measurement_rolls_back_budget(tmp_path):
    db = init(tmp_path)
    with pytest.raises(RuntimeError):
        query(db, FakeMeasurement(RuntimeError("measurement unavailable")))
    assert used(db) == 0
    query(db, FakeMeasurement())
    assert used(db) == 250000


def test_bad_ids_and_budgets_fail_closed(tmp_path):
    db = init(tmp_path)
    m = FakeMeasurement()
    with pytest.raises(ValueError):
        query(db, m, query_id="raw_sql")
    assert used(db) == 0
    for e in (float("nan"), 0, -1, 0.2500003):
        with pytest.raises(ValueError):
            query(db, m, eps=e)
    assert used(db) == 0


def test_opendp_dependency_and_invalid_epsilon():
    with pytest.raises(DPConfigurationError):
        OpenDPCount(0.0)
    with pytest.raises(DPConfigurationError):
        OpenDPCount(float("nan"))
    try:
        import opendp
    except ImportError:
        with pytest.raises(DPLibraryUnavailable):
            OpenDPCount(0.25)


@pytest.mark.skipif(__import__('importlib').util.find_spec('opendp') is None, reason="OpenDP wheel not installed")
def test_actual_opendp_privacy_map_and_sampling():
    mechanism = OpenDPCount(0.5)
    assert mechanism.verified_map_epsilon <= 0.5
    assert isinstance(mechanism.release(100), float)


def test_concurrent_queries_have_atomic_global_budget(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    db = init(tmp_path)
    def attempt(_):
        try:
            query(db, FakeMeasurement(), eps=0.25, cap=0.5)
            return "released"
        except BudgetExhausted:
            return "denied"
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(attempt, range(12)))
    assert results.count("released") == 2
    assert results.count("denied") == 10
    assert used(db) == 500000


def test_privacy_budget_cap_cannot_be_increased_on_restart(tmp_path):
    db = init(tmp_path)
    query(db, FakeMeasurement(), cap=0.5)
    with pytest.raises(ValueError, match="immutable"):
        query(db, FakeMeasurement(), cap=0.75)
    assert used(db) == 250000


def test_retry_key_returns_identical_noise_without_spending_again(tmp_path):
    db = init(tmp_path)
    m = FakeMeasurement()
    key = "research_retry_key_001"
    def release(query_id="all_records", request_id=key):
        return release_dp_at_site(db, query_id=query_id, principal_scope="principal",
                                  epsilon=.25, total_budget=.25, mechanism=m, request_id=request_id)
    first = release()
    assert not first.replayed
    assert release().noisy_count == first.noisy_count
    assert release().replayed
    assert len(m.observed) == 1 and used(db) == 250000
    from feddisclosure.defenses.dp_release import ReleaseIdConflict
    with pytest.raises(ReleaseIdConflict):
        release(query_id="negative_records")
    with pytest.raises(BudgetExhausted):
        release(request_id="new_request_key_002")
    assert used(db) == 250000


def test_retry_key_concurrent_and_restart_safe(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    db = init(tmp_path)
    m = FakeMeasurement()
    def attempt(_):
        return release_dp_at_site(db, query_id="all_records", principal_scope="principal",
                                  epsilon=.25, total_budget=.25, mechanism=m,
                                  request_id="same_request_key_123").noisy_count
    with ThreadPoolExecutor(max_workers=10) as executor:
        values = list(executor.map(attempt, range(20)))
    assert values == [40.75] * 20
    assert m.observed == [40] and used(db) == 250000
    assert attempt(0) == 40.75  # Fresh connection replays previously sampled value.


def test_retry_key_invalid_and_cached_epsilon_mismatch(tmp_path):
    from feddisclosure.defenses.dp_release import ReleaseIdConflict
    db = init(tmp_path)
    for bad in ("short", "../wrong/path", "a" * 101):
        with pytest.raises(ValueError):
            release_dp_at_site(db, query_id="all_records", principal_scope="principal",
                               epsilon=.25, total_budget=.5, mechanism=FakeMeasurement(),
                               request_id=bad)
    release_dp_at_site(db, query_id="all_records", principal_scope="principal",
                       epsilon=.25, total_budget=.5, mechanism=FakeMeasurement(),
                       request_id="conflict_key_123")
    with pytest.raises(ReleaseIdConflict):
        release_dp_at_site(db, query_id="all_records", principal_scope="principal",
                           epsilon=.125, total_budget=.5, mechanism=FakeMeasurement(),
                           request_id="conflict_key_123")
    assert used(db) == 250000
