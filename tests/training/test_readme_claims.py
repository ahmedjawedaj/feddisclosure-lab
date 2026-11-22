"""verify_readme_claims fails closed. Temporary README and evidence fixtures only."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_readme_claims as v  # noqa: E402

README = """# Title
| View | MSE | Share |
|---|---|---|
| Alpha view | 0.013 | 94% |
| Beta view | 0.057 | 67.5% |

```bash
# fast test suite (12 pass and 3 skip, no Docker needed)
```
"""
CLAIMS = [v.Claim("alpha.mse", "Alpha view", 1, "ev/a.json", "views/x/mse/0"),
          v.Claim("alpha.rate", "Alpha view", 2, "ev/a.json", "views/x/rate/0", 100),
          v.Claim("beta.rate", "Beta view", 2, "ev/a.json", "views/y/rate/0", 100)]


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "ev").mkdir()
    (tmp_path / "ev/a.json").write_text(json.dumps(
        {"views": {"x": {"mse": [0.0132035], "rate": [0.9375]}, "y": {"rate": [0.675]}}}))
    (tmp_path / "README.md").write_text(README)
    return tmp_path


def statuses(repo, claims=CLAIMS, suite=False, prose=(), verdicts=(), phrases=()):
    res = v.verify(repo / "README.md", repo, list(claims), suite=suite, prose=list(prose), verdicts=list(verdicts),
                   phrases=list(phrases))
    return {cid: st for cid, st, _ in res}


def test_matching_numbers_pass_within_printed_rounding(repo):
    assert set(statuses(repo).values()) == {"OK"}          # 94% vs 93.75 and 0.013 vs 0.0132 are rounding


def test_mismatch_is_reported_and_exit_code_is_one(repo, monkeypatch):
    (repo / "README.md").write_text(README.replace("| 0.057 | 67.5% |", "| 0.057 | 70.5% |"))
    assert statuses(repo)["beta.rate"] == "MISMATCH"
    monkeypatch.setattr(v, "CLAIMS", CLAIMS)
    assert v.main(["--readme", str(repo / "README.md"), "--root", str(repo)]) == 1


def test_printed_precision_sets_the_tolerance(repo):
    (repo / "README.md").write_text(README.replace("| 0.013 |", "| 0.0140 |"))      # 4 decimals, 0.0132 is off
    assert statuses(repo)["alpha.mse"] == "MISMATCH"


def test_missing_evidence_file_fails_closed(repo):
    (repo / "ev/a.json").unlink()
    assert set(statuses(repo).values()) == {"MISSING"}


def test_missing_json_path_and_missing_readme_row_fail_closed(repo):
    (repo / "ev/a.json").write_text(json.dumps({"views": {"x": {"mse": [0.0132035]}}}))
    s = statuses(repo)
    assert s["alpha.rate"] == "MISSING" and s["beta.rate"] == "MISSING" and s["alpha.mse"] == "OK"
    (repo / "README.md").write_text(README.replace("Beta view", "Gamma view"))
    assert statuses(repo)["beta.rate"] == "MISSING"


def test_missing_readme_fails_closed(tmp_path):
    assert v.verify(tmp_path / "README.md", tmp_path, CLAIMS)[0][1] == "MISSING"


def test_fast_suite_count_line(repo):
    ev = repo / "reports/results/privacy/v0143"
    assert statuses(repo, [], suite=True)["tests.fast_suite_count"] == "MISSING"
    ev.mkdir(parents=True)
    (ev / "fast_suite.txt").write_text("date: 2026-11-23\n12 passed, 3 skipped, 8 deselected in 90.00s\n")
    assert statuses(repo, [], suite=True)["tests.fast_suite_count"] == "OK"
    (ev / "fast_suite.txt").write_text("13 passed, 3 skipped in 90.00s\n")
    assert statuses(repo, [], suite=True)["tests.fast_suite_count"] == "MISMATCH"


def test_fast_suite_line_must_be_a_real_pytest_closing_line_without_failures(repo):
    ev = repo / "reports/results/privacy/v0143"
    ev.mkdir(parents=True)
    suite = lambda: statuses(repo, [], suite=True)["tests.fast_suite_count"]        # noqa: E731
    (ev / "fast_suite.txt").write_text("I ran 12 passed, 3 skipped by hand\n")      # prose is not a pytest line
    assert suite() == "MISSING"
    (ev / "fast_suite.txt").write_text("= 12 passed, 3 skipped in 1.00s =\nnote: 99 passed in 1s later\n")
    assert suite() == "OK"
    (ev / "fast_suite.txt").write_text("1 failed, 11 passed, 3 skipped in 90.00s\n")
    assert suite() == "MISMATCH"                                                  # same counts but a failure
    (ev / "fast_suite.txt").write_text("12 passed, 3 skipped, 1 error in 90.00s\n")
    assert suite() == "MISMATCH"


def test_nan_and_infinity_in_the_evidence_never_pass(repo):
    for bad in ("NaN", "Infinity", "-Infinity"):
        (repo / "ev/a.json").write_text('{"views": {"x": {"mse": [%s], "rate": [%s]}, "y": {"rate": [%s]}}}' % (bad, bad, bad))
        assert set(statuses(repo).values()) == {"MISSING"}, bad


def test_compare_rejects_non_finite_values_directly():
    assert v.compare(0.013, 3, float("nan"))[0] == "MISSING"
    assert v.compare(float("nan"), 3, 0.013)[0] == "MISSING"
    assert v.compare(0.013, 3, float("inf"))[0] == "MISSING"


def test_parse_number_handles_exponents_and_signs():
    assert v.parse_number("1e-3") == (0.001, 3)
    assert v.parse_number("1.5e-3 (note)") == (0.0015, 4)
    assert v.parse_number("2E+2") == (200.0, -2)
    assert v.parse_number("-0.25") == (-0.25, 2)
    assert v.parse_number("94%") == (94.0, 0)
    with pytest.raises(ValueError):
        v.parse_number("none")


def test_exponent_in_the_readme_sets_a_matching_tolerance(repo):
    (repo / "README.md").write_text(README.replace("| 0.013 |", "| 1e-3 |"))
    claim = [v.Claim("alpha.mse", "Alpha view", 1, "ev/a.json", "views/x/mse/0")]
    for value, expected in ((0.00104, "OK"), (0.0016, "MISMATCH")):                  # 1e-3 prints three decimals
        (repo / "ev/a.json").write_text(json.dumps({"views": {"x": {"mse": [value]}}}))
        assert statuses(repo, claim)["alpha.mse"] == expected, value


def test_tolerance_is_half_a_unit_of_the_last_printed_digit_and_no_wider(repo):
    mse = [v.Claim("alpha.mse", "Alpha view", 1, "ev/a.json", "views/x/mse/0")]
    pct = [v.Claim("alpha.rate", "Alpha view", 2, "ev/a.json", "views/x/rate/0", 100)]
    for value, expected in ((0.0134999, "OK"), (0.0135001, "MISMATCH"), (0.0125001, "OK"), (0.0124999, "MISMATCH")):
        (repo / "ev/a.json").write_text(json.dumps({"views": {"x": {"mse": [value], "rate": [0.94]}}}))
        assert statuses(repo, mse)["alpha.mse"] == expected, value
    for value, expected in ((0.944999, "OK"), (0.945001, "MISMATCH")):                 # 94% prints no decimals, so +-0.5 points
        (repo / "ev/a.json").write_text(json.dumps({"views": {"x": {"mse": [0.013], "rate": [value]}}}))
        assert statuses(repo, pct)["alpha.rate"] == expected, value


def test_derived_difference_claim(repo):
    (repo / "README.md").write_text(README.replace("| 94% |", "| 94% | 26 points above |"))
    (repo / "ev/b.json").write_text(json.dumps({"floor": 0.675}))
    claim = v.Claim("alpha.gap", "Alpha view", 3, "ev/a.json", "views/x/rate/0", 100, "ev/b.json", "floor")
    assert statuses(repo, [claim])["alpha.gap"] == "OK"                               # 93.75 - 67.5 = 26.25
    (repo / "ev/b.json").write_text(json.dumps({"floor": 0.60}))
    assert statuses(repo, [claim])["alpha.gap"] == "MISMATCH"


PROSE = [v.ProseClaim("prose.counts", r"\((\d+) of (\d+) images, interval (\d+)% to (\d+)%\)",
                      (v.Ref(1, "ev/a.json", "k"), v.Ref(2, "ev/a.json", "n"),
                       v.Ref(3, "ev/a.json", "lo", 100), v.Ref(4, "ev/a.json", "hi", 100)))]
PROSE_README = README + "\nThe floor is 67.5% (54 of 80 images, interval 57% to 77%).\n"


def prose_repo(repo, **over):
    ev = {"k": 54, "n": 80, "lo": 0.5664, "hi": 0.7676}
    ev.update(over)
    (repo / "ev/a.json").write_text(json.dumps(ev))
    (repo / "README.md").write_text(PROSE_README)


def test_prose_numbers_are_checked(repo):
    prose_repo(repo)
    assert statuses(repo, [], prose=PROSE)["prose.counts"] == "OK"
    prose_repo(repo, k=55)
    assert statuses(repo, [], prose=PROSE)["prose.counts"] == "MISMATCH"
    prose_repo(repo, hi=0.7801)                                                       # 77% printed, 78.01 stored
    assert statuses(repo, [], prose=PROSE)["prose.counts"] == "MISMATCH"
    prose_repo(repo, lo=float("nan"))
    assert statuses(repo, [], prose=PROSE)["prose.counts"] == "MISSING"


def test_prose_sentence_that_disappears_or_changes_shape_fails_closed(repo):
    prose_repo(repo)
    (repo / "README.md").write_text(README)
    assert statuses(repo, [], prose=PROSE)["prose.counts"] == "MISSING"


VERDICT = [v.Verdict("verdict.alpha", "Alpha view", 3, "ev/a.json", "ci/1")]


def verdict_repo(repo, cell, low):
    (repo / "README.md").write_text(README.replace("| 94% |", f"| 94% | {cell} |"))
    (repo / "ev/a.json").write_text(json.dumps({"ci": [-0.1, low, 0.3]}))


def test_verdict_words_follow_the_sign_of_the_evidence_interval(repo):
    verdict_repo(repo, "26 points above", 0.175)
    assert statuses(repo, [], verdicts=VERDICT)["verdict.alpha"] == "OK"
    verdict_repo(repo, "does not beat the floor", -0.2)
    assert statuses(repo, [], verdicts=VERDICT)["verdict.alpha"] == "OK"


def test_stale_verdict_is_caught_when_the_evidence_flips(repo):
    verdict_repo(repo, "does not beat the floor", 0.175)       # evidence now says clearly above
    assert statuses(repo, [], verdicts=VERDICT)["verdict.alpha"] == "MISMATCH"
    verdict_repo(repo, "26 points above", -0.2)                # evidence now says not above
    assert statuses(repo, [], verdicts=VERDICT)["verdict.alpha"] == "MISMATCH"
    verdict_repo(repo, "below, no leakage shown", -0.2)        # the withdrawn wording is neither verdict
    assert statuses(repo, [], verdicts=VERDICT)["verdict.alpha"] == "MISMATCH"


def test_required_phrase_must_be_in_its_cell(repo):
    verdict_repo(repo, "not a protection claim", 0.0)
    ph = [v.Phrase("p", "Alpha view", 3, "not a protection claim")]
    assert statuses(repo, [], phrases=ph)["p"] == "OK"
    verdict_repo(repo, "DP stops the attack", 0.0)
    assert statuses(repo, [], phrases=ph)["p"] == "MISMATCH"


def test_the_real_readme_matches_the_saved_evidence():
    res = v.verify(ROOT / "README.md", ROOT, suite=False)
    assert [(cid, st, d) for cid, st, d in res if st != "OK"] == []
    ids = {cid for cid, _, _ in res}
    assert {"prose.aggregate_against_floor", "verdict.aggregate", "inversion.plaintext_individual.points_above_floor"} <= ids


def test_record_fast_suite_keeps_only_the_closing_line():
    import record_fast_suite as rec
    out = "..F.\n= 2 passed, 1 skipped, 4 deselected, 2 warnings in 20.30s =\n"
    assert rec.closing_line(out) == "2 passed, 1 skipped, 4 deselected, 2 warnings in 20.30s"
    assert rec.closing_line("2 passed by hand\nno summary") is None
