"""The release gate must flag what it claims to flag and pass clean prose."""
from pathlib import Path
import importlib.util

SPEC = importlib.util.spec_from_file_location("release_gate", Path(__file__).resolve().parents[1] / "scripts" / "release_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_clean_prose_passes():
    assert gate.prose_violations("a.md", "The clip moves by exp(-0.1) per round.") == []


def test_dash_semicolon_and_banned_word_are_flagged():
    assert gate.prose_violations("a.md", "one — two") == ["a.md: dash"]
    assert gate.prose_violations("a.md", "one; two") == ["a.md: semicolon"]
    assert gate.prose_violations("a.md", "This is a flaw.") == ["a.md: 'flaw'"]


def test_banned_word_needs_a_word_boundary():
    assert gate.prose_violations("a.md", "debugging output") == []


def test_commit_msg_hook_rejects_trailers(tmp_path):
    import subprocess
    hook = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "commit-msg"
    ok, bad = tmp_path / "ok", tmp_path / "bad"
    ok.write_text("Add a thing\n")
    bad.write_text("Add a thing\n\nCo-Authored-By: Someone <x@example.com>\n")
    assert subprocess.run(["bash", str(hook), str(ok)]).returncode == 0
    assert subprocess.run(["bash", str(hook), str(bad)]).returncode == 1
