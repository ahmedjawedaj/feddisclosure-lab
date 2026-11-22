"""CI harness shim: serialise Flower's state-database migrations.

Flower 1.27 runs Alembic migrations whenever one of its state objects initialises. On
clean hosted runners the SuperLink, its gRPC handler threads and the simulation
subprocess do this at the same moment, and Alembic is not safe to run concurrently
(observed as `'config'`, `'script'`, `table node already exists` and a dropped
connection, a different one on each run). This shim wraps `run_migrations` with a thread
lock and an inter-process file lock so migrations run one at a time. It changes nothing
else: the real SuperLink, SuperNodes, ServerApp and ClientApps still run.

It is only loaded when this directory is on PYTHONPATH (feddisclosure.training.flwr_cli
does that for `flwr run`) and only acts inside Flower's own console scripts.
"""
import os
import sys
import threading
from pathlib import Path

_FLOWER_SCRIPTS = ("flower-superlink", "flwr-simulation", "flwr-serverapp", "flwr-clientapp",
                   "flower-supernode", "flower-superexec")


def _install() -> None:
    import fcntl
    import flwr.supercore.sql_mixin as mixin

    original = mixin.run_migrations
    state_dir = Path(os.environ.get("FEDDISCLOSURE_FLWR_HOME", Path.home() / ".flwr"))
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "migrate.lock"
    thread_lock = threading.RLock()

    def serialised(engine):
        # An in-memory database has no file to share and no migrations to run.
        if engine.url.database in (None, "", ":memory:"):
            return None
        with thread_lock, open(lock_path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                return original(engine)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    mixin.run_migrations = serialised
    with open(state_dir / "migrate.lock.log", "a") as log:
        log.write(f"shim active pid={os.getpid()} script={os.path.basename(sys.argv[0])}\n")


if sys.argv and os.path.basename(sys.argv[0]).startswith(_FLOWER_SCRIPTS):
    try:
        _install()
    except Exception as exc:  # never break Flower start-up because of the shim itself
        print(f"feddisclosure flwr_shim not installed: {exc!r}", file=sys.stderr)
