"""Experimental statistical release controls; not a formal privacy guarantee."""
import json
import sqlite3
from dataclasses import dataclass
from feddisclosure.analytics.engine import matched_record_ids
from feddisclosure.common.models import AggregateRequest


@dataclass(frozen=True)
class ReleaseDecision:
    approved: bool
    count: int | None
    reason: str


def evaluate_release(conn: sqlite3.Connection, request: AggregateRequest, min_cell: int) -> ReleaseDecision:
    """Evaluate and persist atomically within a transaction owned by caller.

    The history rule checks differences in returned membership sets. This is a
    research heuristic, not comprehensive protection from adaptive inference.
    """
    if min_cell < 1:
        raise ValueError("min_cell must be >= 1")
    selected = matched_record_ids(conn, request.query_id)
    count = len(selected)
    if count < min_cell:
        decision = ReleaseDecision(False, None, "small_cell")
        _audit(conn, request, decision)
        return decision
    if request.policy == "history":
        for (prior_json,) in conn.execute(
            "SELECT membership_json FROM release_history WHERE experiment_id=? AND policy=?",
            (request.experiment_id, request.policy),
        ):
            prior = frozenset(json.loads(prior_json))
            difference = selected.symmetric_difference(prior)
            if 0 < len(difference) < min_cell:
                decision = ReleaseDecision(False, None, "near_duplicate_complement")
                _audit(conn, request, decision)
                return decision
    conn.execute(
        "INSERT INTO release_history(experiment_id, query_id, policy, membership_json, count) VALUES (?, ?, ?, ?, ?)",
        (request.experiment_id, request.query_id, request.policy, json.dumps(sorted(selected)), count),
    )
    decision = ReleaseDecision(True, count, "released")
    _audit(conn, request, decision)
    return decision


def _audit(conn: sqlite3.Connection, request: AggregateRequest, decision: ReleaseDecision) -> None:
    """Only metadata. Matched IDs never enter audit rows, logs, or HTTP responses."""
    conn.execute(
        "INSERT INTO query_audit(experiment_id, query_id, policy, approved, reason) VALUES (?, ?, ?, ?, ?)",
        (request.experiment_id, request.query_id, request.policy, int(decision.approved), decision.reason),
    )


def evaluate_at_site(db_path: str, request: AggregateRequest, min_cell: int) -> ReleaseDecision:
    with sqlite3.connect(db_path, timeout=15, isolation_level=None) as conn:
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            decision = evaluate_release(conn, request, min_cell)
            conn.execute("COMMIT")
            return decision
        except Exception:
            conn.execute("ROLLBACK")
            raise
