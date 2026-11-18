"""Guardrail: attack implementation must not import data or evaluator internals."""
from pathlib import Path
import inspect
import feddisclosure.attacks.aggregate_difference as attack


def test_attacker_has_no_database_or_ground_truth_import():
    source = inspect.getsource(attack)
    for blocked in ("sqlite3", "ground_truth", "initialize_database", "requests", "httpx"):
        assert blocked not in source


def test_no_personal_datasets_in_repo():
    root = Path(__file__).resolve().parents[2]
    assert not list(root.glob("**/*.csv"))
    assert not list(root.glob("**/*.parquet"))
