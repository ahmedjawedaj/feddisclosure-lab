import sqlite3
from feddisclosure.analytics.policy import evaluate_at_site
from feddisclosure.common.models import AggregateRequest
from feddisclosure.data.synthetic import InstitutionSpec, initialize_database


def setup_site(tmp_path, rows=40, positive=1):
    db = str(tmp_path / "one.sqlite")
    initialize_database(db, InstitutionSpec("site-one", rows, positive))
    return db


def query(db, name, policy, experiment="test"):
    return evaluate_at_site(db, AggregateRequest(query_id=name, policy=policy, experiment_id=experiment), 5)


def test_naive_counts_allow_sensitive_complement(tmp_path):
    db = setup_site(tmp_path)
    assert query(db, "all_records", "naive").count == 40
    assert query(db, "negative_records", "naive").count == 39


def test_history_blocks_small_complement(tmp_path):
    db = setup_site(tmp_path)
    assert query(db, "all_records", "history").approved
    second = query(db, "negative_records", "history")
    assert second.approved is False
    assert second.reason == "near_duplicate_complement"
    assert second.count is None


def test_history_is_scoped_to_experiment(tmp_path):
    db = setup_site(tmp_path)
    assert query(db, "all_records", "history", "a").approved
    assert query(db, "negative_records", "history", "b").approved
    assert not query(db, "negative_records", "history", "a").approved


def test_exact_repeat_is_not_blocked(tmp_path):
    db = setup_site(tmp_path)
    assert query(db, "all_records", "history").approved
    assert query(db, "all_records", "history").approved


def test_suppression_of_small_direct_results(tmp_path):
    db = setup_site(tmp_path, rows=6, positive=2)
    assert query(db, "negative_records", "naive").reason == "small_cell"


def test_invalid_query_rejected_at_schema(tmp_path):
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        AggregateRequest(query_id="SELECT * FROM records", policy="naive", experiment_id="x")


def test_database_identity_cannot_be_implicitly_changed(tmp_path):
    import pytest
    db = setup_site(tmp_path)
    with pytest.raises(ValueError, match="Dataset mismatch"):
        initialize_database(db, InstitutionSpec("site-two", 40, 1))
