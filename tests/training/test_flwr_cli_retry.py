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
    code, out, err, timed_out, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, wait=0.01, reset=lambda: None, listening=lambda: True)
    assert code == 0 and "ok" in out and not timed_out
    assert [a["superlink_unavailable"] for a in log] == [True, True, False]


def test_other_failures_are_final_and_not_retried(tmp_path):
    body = "import sys\nprint('real training error', file=sys.stderr)\nsys.exit(3)\n"
    code, out, err, timed_out, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, wait=0.01, reset=lambda: None, listening=lambda: True)
    assert code == 3 and len(log) == 1 and not log[0]["superlink_unavailable"]


def test_gives_up_after_the_attempt_limit(tmp_path):
    body = f"import sys\nprint({UNAVAILABLE!r}, file=sys.stderr)\nsys.exit(1)\n"
    code, out, err, timed_out, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=3, wait=0.01, reset=lambda: None, listening=lambda: True)
    assert code == 1 and len(log) == 3


def test_reset_runs_between_retries_only(tmp_path):
    body = f"import sys\nprint({UNAVAILABLE!r}, file=sys.stderr)\nsys.exit(1)\n"
    calls = []
    run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=3, wait=0.01, reset=lambda: calls.append(1), listening=lambda: False)
    assert len(calls) == 2  # between attempts 1-2 and 2-3, not after the last


def test_every_attempt_records_its_output_tail(tmp_path):
    body = f"import sys\nprint('boom detail', file=sys.stderr)\nprint({UNAVAILABLE!r}, file=sys.stderr)\nsys.exit(1)\n"
    _, _, _, _, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=2, wait=0.01, reset=lambda: None, listening=lambda: True)
    assert all("boom detail" in a["output_tail"] for a in log)


def test_connection_written_to_a_fresh_config(tmp_path):
    import tomllib
    from feddisclosure.training.flwr_cli import ensure_simulation_connection
    path = ensure_simulation_connection(tmp_path / ".flwr" / "config.toml")
    cfg = tomllib.loads(path.read_text())
    assert cfg["superlink"]["default"] == "local-simulation"
    assert cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] == 3


def test_existing_config_is_extended_not_overwritten(tmp_path):
    import tomllib
    from feddisclosure.training.flwr_cli import ensure_simulation_connection
    path = tmp_path / "config.toml"
    path.write_text('[superlink]\ndefault = "local"\n\n[superlink.local]\noptions.num-supernodes = 10')
    ensure_simulation_connection(path)
    cfg = tomllib.loads(path.read_text())
    assert cfg["superlink"]["default"] == "local"
    assert cfg["superlink"]["local"]["options"]["num-supernodes"] == 10
    assert cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] == 3


def test_conflicting_supernode_count_is_an_error(tmp_path):
    import pytest
    from feddisclosure.training.flwr_cli import ensure_simulation_connection
    path = tmp_path / "config.toml"
    path.write_text('[superlink.local-simulation]\noptions.num-supernodes = 5\n')
    with pytest.raises(RuntimeError, match="different supernode count"):
        ensure_simulation_connection(path)
    assert "= 5" in path.read_text()


def test_wrapped_coloured_sqlite_race_is_recognised():
    from feddisclosure.training.flwr_cli import startup_race
    boxed = ("\x1b[31m\u2502\x1b[0m Exception calling application: (sqlite3.OperationalError) table node already "
             "\x1b[31m\u2502\x1b[0m\n\x1b[31m\u2502\x1b[0m exists \x1b[31m\u2502\x1b[0m")
    assert startup_race(boxed) == "sqlite_table_exists"
    assert startup_race("The current node timed out during startup.") == "ray_startup_timeout"
    assert startup_race(UNAVAILABLE) == "superlink_unavailable"
    assert startup_race("\x1b[31m\u2502\x1b[0m Exception calling application: 'config' \x1b[31m\u2502\x1b[0m") == "superlink_servicer_error"


def test_training_errors_are_not_start_up_races():
    from feddisclosure.training.flwr_cli import startup_race
    assert startup_race("RuntimeError: Only 0 of 1 rounds produced an aggregate") is None
    assert startup_race("AssertionError: loss is nan") is None
    assert startup_race("KeyError: 'config'") is None  # only the SuperLink-side gRPC form is a start-up race
    assert startup_race("Exception in ServerApp thread") is None


def test_sqlite_race_is_retried_then_succeeds(tmp_path):
    marker = tmp_path / "n"
    body = (
        "import pathlib,sys\n"
        f"m=pathlib.Path({str(marker)!r})\n"
        "n=int(m.read_text()) if m.exists() else 0\n"
        "m.write_text(str(n+1))\n"
        "if n<1: print('table node already\\nexists', file=sys.stderr); sys.exit(1)\n"
        "print('ok')\n"
    )
    code, out, _, _, log = run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, wait=0.01, reset=lambda: None, listening=lambda: True)
    assert code == 0 and [a["startup_race"] for a in log] == ["sqlite_table_exists", None]


def test_servicer_error_variants_are_retried(tmp_path):
    from feddisclosure.training.flwr_cli import startup_race
    for msg in ("Exception calling application: 'config'",
                "Exception calling application: invalid interpolation syntax in 'sqlite:///%3Amemory%3A' at position 10"):
        assert startup_race(msg) == "superlink_servicer_error"


def test_booting_superlink_is_reused_not_reset(tmp_path):
    body = f"import sys\nprint({UNAVAILABLE!r}, file=sys.stderr)\nsys.exit(1)\n"
    resets = []
    run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=3, wait=0.01,
                 reset=lambda: resets.append(1), listening=lambda: True)
    assert resets == []  # a listening SuperLink is kept so the next attempt can connect to it


def test_state_database_races_always_reset(tmp_path):
    body = "import sys\nprint('table node already exists', file=sys.stderr)\nsys.exit(1)\n"
    resets = []
    run_flwr_cli(_script(tmp_path, body), cwd=tmp_path, timeout=60, attempts=3, wait=0.01,
                 reset=lambda: resets.append(1), listening=lambda: True)
    assert len(resets) == 2


def test_shim_directory_is_prepended_to_pythonpath():
    from feddisclosure.training.flwr_cli import SHIM_DIR, shim_environment
    assert (SHIM_DIR / "sitecustomize.py").is_file()
    env = shim_environment({"PYTHONPATH": "/somewhere", "HOME": "/h"})
    assert env["PYTHONPATH"].split(":")[0] == str(SHIM_DIR) and env["PYTHONPATH"].endswith("/somewhere")
    assert shim_environment({})["PYTHONPATH"] == str(SHIM_DIR)


def test_shim_only_acts_inside_flower_console_scripts(tmp_path):
    import subprocess
    from feddisclosure.training.flwr_cli import shim_environment
    env = shim_environment()
    env["FEDDISCLOSURE_FLWR_HOME"] = str(tmp_path)
    subprocess.run([sys.executable, "-c", "pass"], env=env, check=True)
    assert not (tmp_path / "migrate.lock.log").exists()  # an ordinary python process is untouched
