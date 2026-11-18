from concurrent.futures import ThreadPoolExecutor
import sqlite3
from feddisclosure.common.models import AggregateRequest
from feddisclosure.analytics.policy import evaluate_at_site
from feddisclosure.data.synthetic import InstitutionSpec, initialize_database


def test_concurrent_releases_are_atomic_and_audit_has_no_ids(tmp_path):
    db = tmp_path / "site.sqlite"
    initialize_database(db, InstitutionSpec("test-site", 40, 1))
    requests = [AggregateRequest(experiment_id=f"stress_{i}", query_id="all_records", policy="naive") for i in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        decisions = list(pool.map(lambda r: evaluate_at_site(str(db), r, 5), requests))
    assert all(d.approved and d.count == 40 for d in decisions)
    # Same experiment, different query: release block is captured without membership content.
    evaluate_at_site(str(db), AggregateRequest(experiment_id="history", query_id="all_records", policy="history"), 5)
    denied = evaluate_at_site(str(db), AggregateRequest(experiment_id="history", query_id="negative_records", policy="history"), 5)
    assert not denied.approved
    with sqlite3.connect(db) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(query_audit)")]
        assert "membership_json" not in columns
        assert "record_id" not in columns
        assert conn.execute("SELECT count(*) FROM query_audit").fetchone()[0] == 22
        assert conn.execute("SELECT count(*) FROM query_audit WHERE approved=0").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM release_history").fetchone()[0] == 21
