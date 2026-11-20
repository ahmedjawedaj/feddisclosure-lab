"""Principal-wide privacy accounting for a site's allowlisted count releases.

The ledger stores only budget use and public release metadata. Raw counts and
private record IDs do not appear in audit rows. Execute under BEGIN IMMEDIATE
on the same SQLite database that owns the protected records.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import sqlite3
import re
from typing import Protocol
from feddisclosure.analytics.engine import matched_record_ids


class BudgetExhausted(Exception):
    pass


class ReleaseIdConflict(Exception):
    """A caller is attempting to reuse a retry key for another DP query."""
    pass


class CountMechanism(Protocol):
    def release(self, count: int) -> float: ...


@dataclass(frozen=True)
class PrivateCountRelease:
    noisy_count: float
    epsilon_spent_this_release: float
    delta: float = 0.0
    replayed: bool = False


def release_dp_count(conn: sqlite3.Connection, *, query_id: str, principal_scope: str,
                     epsilon: float, total_budget: float, mechanism: CountMechanism,
                     request_id: str | None = None) -> PrivateCountRelease:
    """Spend once, atomically, independent of attacker-supplied experiment IDs.

    Caller MUST hold BEGIN IMMEDIATE; errors must trigger rollback. A site using
    this policy must not offer an unprotected result through an alternative API
    to the same adversary. A budget error is independent of private row contents.
    """
    if not principal_scope or not all(math.isfinite(float(x)) and x > 0 for x in (epsilon, total_budget)):
        raise ValueError("Invalid principal scope or privacy budgets")
    if epsilon > total_budget:
        raise ValueError("Per-query epsilon exceeds total budget")
    if request_id is not None and (not isinstance(request_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{12,100}", request_id)):
        raise ValueError("Invalid DP request_id")
    conn.execute("""CREATE TABLE IF NOT EXISTS dp_budget (
                     principal_scope TEXT PRIMARY KEY,
                     spent_micro INTEGER NOT NULL CHECK(spent_micro >= 0),
                     cap_micro INTEGER NOT NULL CHECK(cap_micro >= 1))""")
    # Exact fixed-point ledger prevents a 1e-16 floating arithmetic budget bypass.
    micro = round(epsilon * 1_000_000)
    cap = round(total_budget * 1_000_000)
    if abs(micro / 1_000_000 - epsilon) > 1e-10 or micro < 1:
        raise ValueError("epsilon must be representable at 1e-6 precision")
    if abs(cap / 1_000_000 - total_budget) > 1e-10 or cap < 1:
        raise ValueError("budget must be representable at 1e-6 precision")
    # Reuse the same *released* random value for an exact retry, never redraw.
    # The table is created inside the same BEGIN IMMEDIATE transaction.
    conn.execute("""CREATE TABLE IF NOT EXISTS dp_release_cache (
        principal_scope TEXT NOT NULL,
        request_id TEXT NOT NULL,
        query_id TEXT NOT NULL,
        epsilon_micro INTEGER NOT NULL CHECK(epsilon_micro > 0),
        released_noisy_count REAL NOT NULL,
        PRIMARY KEY (principal_scope, request_id))""")
    row = conn.execute("SELECT spent_micro, cap_micro FROM dp_budget WHERE principal_scope=?", (principal_scope,)).fetchone()
    if row is not None and int(row[1]) != cap:
        raise ValueError("Privacy budget cap is immutable once a principal has started releasing")
    if request_id is not None:
        cached = conn.execute(
            "SELECT query_id, epsilon_micro, released_noisy_count FROM dp_release_cache "
            "WHERE principal_scope=? AND request_id=?", (principal_scope, request_id)
        ).fetchone()
        if cached is not None:
            if cached[0] != query_id or int(cached[1]) != micro:
                raise ReleaseIdConflict("Request id already bound to another query or epsilon")
            return PrivateCountRelease(float(cached[2]), epsilon, replayed=True)
    spent = 0 if row is None else int(row[0])
    if spent + micro > cap:
        raise BudgetExhausted("privacy budget exhausted")
    # Increments are private persistent state, not controlled by experiment_id.
    if row is None:
        conn.execute("INSERT INTO dp_budget(principal_scope, spent_micro, cap_micro) VALUES (?, ?, ?)", (principal_scope, micro, cap))
    else:
        conn.execute("UPDATE dp_budget SET spent_micro=? WHERE principal_scope=?", (spent + micro, principal_scope))
    true_count = len(matched_record_ids(conn, query_id))
    released = mechanism.release(true_count)  # established measurement, never add custom noise
    if not isinstance(released, (int, float)) or not math.isfinite(released):
        raise RuntimeError("Mechanism returned invalid noisy count")
    if request_id is not None:
        conn.execute("INSERT INTO dp_release_cache "
                     "(principal_scope, request_id, query_id, epsilon_micro, released_noisy_count) "
                     "VALUES (?, ?, ?, ?, ?)",
                     (principal_scope, request_id, query_id, micro, float(released)))
    return PrivateCountRelease(float(released), epsilon)


def release_dp_at_site(db_path: str, *, query_id: str, principal_scope: str,
                       epsilon: float, total_budget: float, mechanism: CountMechanism,
                     request_id: str | None = None) -> PrivateCountRelease:
    with sqlite3.connect(db_path, timeout=15, isolation_level=None) as conn:
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = release_dp_count(conn, query_id=query_id, principal_scope=principal_scope,
                                      epsilon=epsilon, total_budget=total_budget, mechanism=mechanism,
                                      request_id=request_id)
            conn.execute("COMMIT")
            return result
        except Exception:
            conn.execute("ROLLBACK")
            raise
