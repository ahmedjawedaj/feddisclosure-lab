"""Durable site-wide research release-policy lock (not a formal privacy guarantee)."""
from __future__ import annotations
import sqlite3

VALID_POLICIES = ("naive", "history", "principal_history")


def enforce_release_mode(db_path: str, policy: str | None, *, dp_enabled: bool) -> None:
    """Lock the mode on a fresh database; reject downgrades and conflicting DP modes.

    Only the one authenticated research principal is supported. A distinct
    production identity system would need a stable principal key in the ledger.
    """
    if policy is not None and policy not in VALID_POLICIES:
        raise ValueError("Unknown enforced analytics policy")
    if policy is not None and dp_enabled:
        raise ValueError("DP mode and count release policy cannot both be enabled")
    with sqlite3.connect(db_path, isolation_level=None, timeout=15) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS release_mode_lock (
                         lock_id INTEGER PRIMARY KEY CHECK(lock_id=1),
                         enforced_policy TEXT NOT NULL)""")
            row = conn.execute("SELECT enforced_policy FROM release_mode_lock WHERE lock_id=1").fetchone()
            if row is not None and row[0] != policy:
                raise ValueError("Enforced release mode is immutable; refusing downgrade or change")
            if row is None and policy is not None:
                # Existing releases must not be 'reclassified' as defended.
                old = conn.execute("SELECT COUNT(*) FROM query_audit").fetchone()[0]
                if old:
                    raise ValueError("Protected mode requires a fresh site without prior unprotected queries")
                conn.execute("INSERT INTO release_mode_lock VALUES (1, ?)", (policy,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
