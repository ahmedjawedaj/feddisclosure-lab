"""Private local query execution. SQL identifiers are never supplied by users."""
import sqlite3
from feddisclosure.common.models import QueryId

QUERY_SQL = {
    "all_records": "SELECT id FROM records ORDER BY id",
    "negative_records": "SELECT id FROM records WHERE status = 'negative' ORDER BY id",
}


def matched_record_ids(conn: sqlite3.Connection, query_id: QueryId) -> frozenset[int]:
    try:
        query = QUERY_SQL[query_id]
    except KeyError as exc:
        raise ValueError("Query is not allowlisted") from exc
    return frozenset(row[0] for row in conn.execute(query))
