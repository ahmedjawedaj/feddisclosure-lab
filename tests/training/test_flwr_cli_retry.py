import sys

from feddisclosure.training.flwr_cli import UNAVAILABLE, run_flwr_cli


def _script(tmp_path, body):
    path = tmp_path / "fake_cli.py"
    path.write_text(body)
    return [sys.executable, str(path)]


def test_retries_only_the_superlink_unavailable_case(tmp_path):
    marker = tmp_path / "count"
    body = (
        "import pathlib,sys\n"
        f"m=pathlib.Path({str(marker)!r})\n"
        "n=int(m.read_text()) if m.exists() else 0\n"
        "m.write_text(str(n+1))\n"
        f"if n<2: print({UNAVAILABLE!r}, file=sys.stderr); sys.exit(1)\n"
        "print('ok')\n"
    )
    code, out, err, timed_out, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, wait=0.01, reset=lambda: None)
    assert code == 0 and "ok" in out and not timed_out
    assert [a["superlink_unavailable"] for a in log] == [True, True, False]


def test_other_failures_are_final_and_not_retried(tmp_path):
    body = "import sys\nprint('real training error', file=sys.stderr)\nsys.exit(3)\n"
    code, out, err, timed_out, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, wait=0.01, reset=lambda: None)
    assert code == 3 and len(log) == 1 and not log[0]["superlink_unavailable"]


def test_gives_up_after_the_attempt_limit(tmp_path):
    body = f"import sys\nprint({UNAVAILABLE!r}, file=sys.stderr)\nsys.exit(1)\n"
    code, out, err, timed_out, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=3, wait=0.01, reset=lambda: None)
    assert code == 1 and len(log) == 3


def test_reset_runs_between_retries_only(tmp_path):
    body = f"import sys\nprint({UNAVAILABLE!r}, file=sys.stderr)\nsys.exit(1)\n"
    calls = []
    run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=3, wait=0.01, reset=lambda: calls.append(1))
    assert len(calls) == 2  # between attempts 1-2 and 2-3, not after the last


def test_every_attempt_records_its_output_tail(tmp_path):
    body = f"import sys\nprint('boom detail', file=sys.stderr)\nprint({UNAVAILABLE!r}, file=sys.stderr)\nsys.exit(1)\n"
    _, _, _, _, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=2, wait=0.01, reset=lambda: None)
    assert all("boom detail" in a["output_tail"] for a in log)
