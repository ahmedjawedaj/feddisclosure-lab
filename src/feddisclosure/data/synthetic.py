"""Deterministic synthetic datasets. Never use real institution data in this demo."""
from dataclasses import dataclass
import sqlite3
from pathlib import Path


@dataclass(frozen=True)
class InstitutionSpec:
    name: str
    rows: int
    positive: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or not 0 <= self.positive <= self.rows:
            raise ValueError("rows must be positive and positive must be in [0, rows]")


DEMO_SPECS = (
    InstitutionSpec("institution-a", 40, 1),
    InstitutionSpec("institution-b", 35, 1),
    InstitutionSpec("institution-c", 35, 1),
)


def initialize_database(db_path: str | Path, spec: InstitutionSpec) -> None:
    """Seed one private dataset once. Refuse to silently reuse different data."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=15) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('negative', 'positive')))")
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("""CREATE TABLE IF NOT EXISTS release_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            policy TEXT NOT NULL,
            membership_json TEXT NOT NULL,
            count INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        current = dict(conn.execute("SELECT key, value FROM metadata"))
        desired = {"name": spec.name, "rows": str(spec.rows), "positive": str(spec.positive)}
        if current and current != desired:
            raise ValueError(f"Dataset mismatch for {spec.name}; use a fresh volume to change data")
        if not current:
            conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("name", spec.name))
            conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("rows", str(spec.rows)))
            conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("positive", str(spec.positive)))
            negative = spec.rows - spec.positive
            records = [(i, "negative" if i <= negative else "positive") for i in range(1, spec.rows + 1)]
            conn.executemany("INSERT INTO records(id, status) VALUES (?, ?)", records)


def ground_truth_positive(specs: tuple[InstitutionSpec, ...] = DEMO_SPECS) -> int:
    """Evaluator-only ground truth. Do not import from attack implementations."""
    return sum(spec.positive for spec in specs)
