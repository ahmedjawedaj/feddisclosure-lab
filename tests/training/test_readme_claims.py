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
# fast test suite, no Docker needed. The counts depend on what is installed:
#   with flwr and opacus: 12 pass and 3 skip, 4 deselected
#   without flwr and opacus: 9 pass and 6 skip, 4 deselected
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


def statuses(repo, claims=CLAIMS, suite=False, prose=(), verdicts=(), phrases=(), words=()):
    res = v.verify(repo / "README.md", repo, list(claims), suite=suite, prose=list(prose), verdicts=list(verdicts),
                   phrases=list(phrases), words=list(words))
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


FULL_PKGS = "torch 2.15.0+cpu, flwr 1.27.0, opacus 1.6.0, hypothesis 6.170.17"
CI_PKGS = "torch 2.15.0+cpu, flwr absent, opacus absent, hypothesis 6.170.17"


def git_repo(path):
    """A real git repository with one code file. Returns the commit hash. Skips the test if git is unavailable."""
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    run = lambda *a: subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *a],   # noqa: E731
                                    capture_output=True, text=True, check=True).stdout.strip()
    run("init", "-q")
    (path / "src").mkdir(exist_ok=True)
    (path / "src/code.py").write_text("x = 1\n")
    run("add", "src/code.py")
    run("commit", "-q", "-m", "code")
    return run("rev-parse", "HEAD")


def write_record(repo, kind, line, commit, dirty="no", packages=None, tree=None, version="v0150", drop=()):
    name = v.SUITE_FILES[kind]
    d = repo / "reports/results/privacy" / version
    d.mkdir(parents=True, exist_ok=True)
    fields = {"date": "2026-11-23", "commit": commit, "code_dirty": dirty,
              "tree": tree or ("clean checkout of " + commit if kind == "ci" else "working copy"),
              "python": "3.12.3", "packages": packages or (CI_PKGS if kind == "ci" else FULL_PKGS)}
    text = "".join(f"{k}: {x}\n" for k, x in fields.items() if k not in drop) + line + "\n"
    (d / name).write_text(text)


def suite(repo, kind="full"):
    return statuses(repo, [], suite=True)[f"tests.fast_suite_count.{kind}"]


def both(repo):
    return {k: suite(repo, k) for k in ("full", "ci")}


def test_fast_suite_records_are_required_for_both_environments(repo):
    assert both(repo) == {"full": "MISSING", "ci": "MISSING"}


def test_fast_suite_counts_must_match_and_the_record_must_be_current(repo):
    commit = git_repo(repo)
    write_record(repo, "full", "12 passed, 3 skipped, 4 deselected in 90.00s", commit)
    write_record(repo, "ci", "9 passed, 6 skipped, 4 deselected in 9.00s", commit)
    assert both(repo) == {"full": "OK", "ci": "OK"}
    write_record(repo, "full", "13 passed, 3 skipped, 4 deselected in 90.00s", commit)
    assert both(repo) == {"full": "MISMATCH", "ci": "OK"}            # README counts differ from the record


def test_a_record_older_than_the_code_is_flagged(repo):
    commit = git_repo(repo)
    write_record(repo, "full", "12 passed, 3 skipped, 4 deselected in 90.00s", commit)
    write_record(repo, "ci", "9 passed, 6 skipped, 4 deselected in 9.00s", commit)
    (repo / "src/code.py").write_text("x = 2\n")                       # code changed after the record, even uncommitted
    res = statuses(repo, [], suite=True)
    assert res["tests.fast_suite_count.full"] == "MISMATCH" and res["tests.fast_suite_count.ci"] == "MISMATCH"
    detail = {c: d for c, _, d in v.verify(repo / "README.md", repo, [], suite=True, prose=[], verdicts=[], phrases=[])}
    assert "code changed after the record" in detail["tests.fast_suite_count.full"] and "src/code.py" in detail["tests.fast_suite_count.full"]
    (repo / "src/code.py").write_text("x = 1\n")                       # back to the recorded state
    assert both(repo) == {"full": "OK", "ci": "OK"}


def test_changes_outside_the_code_paths_do_not_make_a_record_stale(repo):
    commit = git_repo(repo)
    write_record(repo, "full", "12 passed, 3 skipped, 4 deselected in 90.00s", commit)
    write_record(repo, "ci", "9 passed, 6 skipped, 4 deselected in 9.00s", commit)
    (repo / "README.md").write_text(README + "\nmore prose\n")
    assert both(repo) == {"full": "OK", "ci": "OK"}


def test_a_record_without_a_usable_commit_fails_closed(repo):
    commit = git_repo(repo)
    line = "12 passed, 3 skipped, 4 deselected in 90.00s"
    write_record(repo, "full", line, commit, drop=("commit",))
    assert suite(repo) == "MISSING"                                      # no commit, cannot compare with the code
    write_record(repo, "full", line, "f" * 40)
    assert suite(repo) == "MISSING"                                      # a hash git does not know
    write_record(repo, "full", line, "not-a-hash")
    assert suite(repo) == "MISSING"
    write_record(repo, "full", line, commit, dirty="yes")
    assert suite(repo) == "MISMATCH"                                     # recorded with uncommitted code changes


def test_a_record_outside_a_git_repository_fails_closed(repo):
    write_record(repo, "full", "12 passed, 3 skipped, 4 deselected in 90.00s", "a" * 40)
    assert suite(repo) == "MISSING"


def test_the_newest_record_directory_wins(repo):
    commit = git_repo(repo)
    line = "12 passed, 3 skipped, 4 deselected in 90.00s"
    write_record(repo, "full", "99 passed, 3 skipped, 4 deselected in 90.00s", commit, version="v0143")
    assert suite(repo) == "MISMATCH"                                     # only the old record exists and it disagrees
    write_record(repo, "full", line, commit, version="v0150")
    assert suite(repo) == "OK"                                           # v0150 is newer than v0143
    write_record(repo, "full", "5 passed, 3 skipped, 4 deselected in 90.00s", commit, version="v0160")
    assert suite(repo) == "MISMATCH"                                     # a still newer record takes over
    write_record(repo, "full", "5 passed, 3 skipped, 4 deselected in 90.00s", commit, version="v0.9")
    assert v.newest_record(repo, v.SUITE_FILES["full"]).parent.name == "v0160"


def test_each_record_must_describe_the_environment_the_readme_names(repo):
    commit = git_repo(repo)
    full, ci = "12 passed, 3 skipped, 4 deselected in 90.00s", "9 passed, 6 skipped, 4 deselected in 9.00s"
    write_record(repo, "ci", ci, commit, packages=FULL_PKGS)
    assert suite(repo, "ci") == "MISMATCH"                               # flwr and opacus were installed
    write_record(repo, "ci", ci, commit, tree="working copy")
    assert suite(repo, "ci") == "MISMATCH"                               # not a clean checkout
    write_record(repo, "full", full, commit, packages=CI_PKGS)
    assert suite(repo, "full") == "MISMATCH"                             # the "with flwr and opacus" count needs flwr and opacus


def test_fast_suite_line_must_be_a_real_pytest_closing_line_without_failures(repo):
    commit = git_repo(repo)
    ev = repo / "reports/results/privacy/v0150"

    def put(text):
        write_record(repo, "full", "PLACEHOLDER", commit)
        f = ev / v.SUITE_FILES["full"]
        f.write_text(f.read_text().replace("PLACEHOLDER\n", text))
    put("I ran 12 passed, 3 skipped by hand\n")                          # prose is not a pytest line
    assert suite(repo) == "MISSING"
    put("= 12 passed, 3 skipped in 1.00s =\nnote: 99 passed in 1s later\n")
    assert suite(repo) == "OK"
    put("1 failed, 11 passed, 3 skipped in 90.00s\n")
    assert suite(repo) == "MISMATCH"                                     # same counts but a failure
    put("12 passed, 3 skipped, 1 error in 90.00s\n")
    assert suite(repo) == "MISMATCH"


def test_readme_without_the_two_count_lines_fails_closed(repo):
    (repo / "README.md").write_text(README.replace("with flwr and opacus", "somewhere"))
    assert both(repo)["full"] == "MISSING"


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


def test_record_text_has_the_commit_and_the_environment():
    import record_fast_suite as rec
    text = rec.record_text("5 passed in 1.00s", "a" * 40, False, "working copy", today="2026-11-23")
    fields = v.record_fields(text)
    assert fields["commit"] == "a" * 40 and fields["code_dirty"] == "no" and fields["tree"] == "working copy"
    assert fields["date"] == "2026-11-23" and "torch" in fields["packages"] and "flwr" in fields["packages"]
    assert v.parse_suite_line(text)["passed"] == 5


def test_record_fast_suite_keeps_only_the_closing_line():
    import record_fast_suite as rec
    out = "..F.\n= 2 passed, 1 skipped, 4 deselected, 2 warnings in 20.30s =\n"
    assert rec.closing_line(out) == "2 passed, 1 skipped, 4 deselected, 2 warnings in 20.30s"
    assert rec.closing_line("2 passed by hand\nno summary") is None


# ---------------------------------------------------------------- v0.15 numbers and verdict words in the real README
REAL = (ROOT / "README.md").read_text()
V15_IDS = [c.id for c in v.PROSE if c.id.startswith("v15.")]


def test_every_v15_work_package_has_quoted_numbers_checked():
    assert {"v15.E1", "v15.D3_same_label", "v15.W2", "v15.F1", "v15.B1", "v15.P1", "v15.S3", "v15.X1"} <= set(V15_IDS)
    assert {w.token for w in v.WORDS} >= {"E2", "D3", "B1", "P2", "X1", "S1", "F1"}
    res = {cid: st for cid, st, _ in v.verify(ROOT / "README.md", ROOT, claims=[], suite=False, verdicts=[], phrases=[])}
    assert all(res[i] == "OK" for i in V15_IDS) and all(st == "OK" for cid, st in res.items() if cid.startswith("word."))


@pytest.mark.parametrize("cid", V15_IDS)
def test_a_wrong_number_in_a_v15_sentence_is_a_mismatch_and_a_missing_sentence_fails_closed(tmp_path, cid):
    import re
    claim = next(c for c in v.PROSE if c.id == cid)
    m = re.search(claim.pattern, REAL)
    assert m, f"README no longer matches {cid}"
    for ref in claim.refs:
        a, b = m.span(ref.group)
        wrong = REAL[:a] + "9999" + REAL[b:]
        (tmp_path / "README.md").write_text(wrong)
        res = v.verify(tmp_path / "README.md", ROOT, claims=[], suite=False, prose=[claim], verdicts=[], phrases=[], words=[])
        assert res[0][1] == "MISMATCH", (cid, ref.group, res)
    (tmp_path / "README.md").write_text(REAL[:m.start()] + REAL[m.end():])
    res = v.verify(tmp_path / "README.md", ROOT, claims=[], suite=False, prose=[claim], verdicts=[], phrases=[], words=[])
    assert res[0][1] == "MISSING"


def test_numbers_with_thousands_separators_are_read(repo):
    (repo / "ev/c.json").write_text(json.dumps({"n": 20000}))
    (repo / "README.md").write_text("It ran 20,000 cases.")
    c = v.ProseClaim("t", r"ran ([\d,]+) cases", (v.Ref(1, "ev/c.json", "n"),))
    assert statuses(repo, [], prose=[c])["t"] == "OK"
    (repo / "README.md").write_text("It ran 20,001 cases.")
    assert statuses(repo, [], prose=[c])["t"] == "MISMATCH"


@pytest.mark.parametrize("passed", [True, False])
def test_verdict_words_follow_the_saved_check(repo, passed):
    (repo / "ev/p.json").write_text(json.dumps({"E9": {"pass": passed}, "B9": {"verdict": "PASS" if passed else "FAIL"}}))
    ws = [v.VerdictWord("E9", "ev/p.json", "E9/pass"), v.VerdictWord("B9", "ev/p.json", "B9/verdict")]
    right, wrong = ("PASS", "FAIL") if passed else ("FAIL", "PASS")
    (repo / "README.md").write_text(f"E9 {right} and B9 {right}")
    assert set(statuses(repo, [], words=ws).values()) == {"OK"}
    (repo / "README.md").write_text(f"E9 {wrong} and B9 {right}")
    assert statuses(repo, [], words=ws)["word.E9"] == "MISMATCH"
    (repo / "README.md").write_text(f"E9 {right} and B9 {right} but also E9 {wrong}")
    assert statuses(repo, [], words=ws)["word.E9"] == "MISMATCH"        # contradicting mention
    (repo / "README.md").write_text("E9 is not mentioned. B9 " + right)
    assert statuses(repo, [], words=ws)["word.E9"] == "MISMATCH"        # missing word


def test_verdict_word_fails_closed_on_bad_evidence(repo):
    (repo / "README.md").write_text("E9 PASS")
    w = v.VerdictWord("E9", "ev/nope.json", "E9/pass")
    assert statuses(repo, [], words=[w])["word.E9"] == "MISSING"
    (repo / "ev/p.json").write_text(json.dumps({"E9": {"pass": None}}))
    assert statuses(repo, [], words=[v.VerdictWord("E9", "ev/p.json", "E9/pass")])["word.E9"] == "MISSING"
    assert statuses(repo, [], words=[v.VerdictWord("E9", "ev/p.json", "E9/gone")])["word.E9"] == "MISSING"
