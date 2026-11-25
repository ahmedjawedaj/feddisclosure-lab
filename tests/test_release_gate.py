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


def test_pin_mismatch_is_reported(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    good = gate.hashlib.sha256(b"one").hexdigest()
    assert gate.pin_mismatches({"a.txt": good}, tmp_path, ("a.txt",)) == []
    assert gate.pin_mismatches({"a.txt": "0" * 64}, tmp_path, ("a.txt",)) == ["a.txt"]
    assert gate.pin_mismatches({}, tmp_path, ("a.txt",)) == ["a.txt (no pin)"]
    assert gate.pin_mismatches({"a.txt": good}, tmp_path, ("missing.txt",)) == ["missing.txt (no pin)"]


def test_integrity_fails_without_repin_and_passes_with_it(monkeypatch, capsys):
    monkeypatch.setattr(gate, "pin_mismatches", lambda pins: ["scripts/release_gate.py"])
    monkeypatch.setattr(gate, "gate_diff", lambda base: "DIFF TEXT")
    monkeypatch.delenv("GATE_REPIN", raising=False)
    ok, detail = gate.check_integrity("origin/main")
    assert not ok and "scripts/release_gate.py" in detail
    monkeypatch.setenv("GATE_REPIN", "1")
    ok, detail = gate.check_integrity("origin/main")
    assert ok and "GATE_REPIN=1" in detail
    assert "DIFF TEXT" in capsys.readouterr().out


def test_repo_pins_match_the_files(monkeypatch):
    monkeypatch.delenv("GATE_REPIN", raising=False)
    ok, detail = gate.check_integrity("origin/main")
    assert ok, detail


def test_repin_needs_the_environment_variable(monkeypatch):
    monkeypatch.delenv("GATE_REPIN", raising=False)
    assert gate.repin("origin/main") == 2


def test_prereg_with_predictions_needs_a_power_heading():
    assert gate.prereg_power_missing("# T\n\n## Predictions\n\ntext\n") is True
    assert gate.prereg_power_missing("# T\n\n## Predictions\n\n## Power of each null result\n") is False
    assert gate.prereg_power_missing("# T\n\n## Rows\n\nNo heading for the forecast words here.\n") is False
    assert gate.prereg_power_missing("# T\n\n## Predicted numbers\n") is False
    assert gate.prereg_power_missing("# T\n\nThe word power in text only. Predictions in text only.\n") is False
    assert gate.prereg_power_missing("# T\n\n## Predictions\n\nPower is discussed in a paragraph only.\n") is True


def test_prereg_power_check_applies_only_to_changed_files(monkeypatch, tmp_path):
    (tmp_path / "docs" / "preregistration").mkdir(parents=True)
    (tmp_path / "docs" / "preregistration" / "new.md").write_text("## Predictions\n")
    (tmp_path / "docs" / "preregistration" / "old.md").write_text("## Predictions\n")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "git", lambda *a: "docs/preregistration/new.md\n")
    ok, detail = gate.check_prereg_power("base")
    assert not ok and "new.md" in detail and "old.md" not in detail
    monkeypatch.setattr(gate, "git", lambda *a: "")
    ok, detail = gate.check_prereg_power("base")
    assert ok


def test_template_has_the_five_sections():
    text = (Path(__file__).resolve().parents[1] / "docs" / "preregistration" / "TEMPLATE.md").read_text()
    for h in ("## Question", "## Predictions", "## Power of each null result", "## Stopping rule", "## Deviation log"):
        assert h in text
    assert gate.prereg_power_missing(text) is False
