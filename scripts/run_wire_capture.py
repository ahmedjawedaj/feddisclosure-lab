"""Run the wire-level capture: for each seed one SecAgg+ run and one plaintext control run.

Every run uses four clients with eight public digits each, one round and one local full batch step.
The Docker stack is compose.flower.n4.yaml plus the override compose.wire.yaml (relay in front of
every SuperNode). The ServerApp is the recording one (feddisclosure.training.secagg_server_capture).

The capture fails closed. A run that does not finish, whose capture summary is not marked complete or
whose relay logs are missing is recorded as failed and the script exits non-zero. No partial result is
ever written as a success. Use --dry-run to print the Docker commands without running anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

COMPOSE_BASE = "compose.flower.n4.yaml"
COMPOSE_WIRE = "compose.wire.yaml"
APP_DIR = "docker/wire_relay/app"
CLIENTS = 4
RELAYS = tuple(f"wire-relay-{i}" for i in range(CLIENTS))
MODES = ("secagg", "plaintext")


def run_name(seed: int, mode: str) -> str:
    return f"seed{seed}-{mode}"


def compose_prefix(env_file: Path) -> list[str]:
    return ["docker", "compose", "--env-file", str(env_file), "-f", str(ROOT / COMPOSE_BASE), "-f", str(ROOT / COMPOSE_WIRE)]


def run_config_string(config: dict, manifest: dict, mode: str) -> str:
    from run_flower_docker import _flower_literal
    cfg = {"num-server-rounds": config["rounds"], "local-epochs": config["local_epochs"],
           "batch-size": config["batch_size"], "learning-rate": config["learning_rate"],
           "proximal-mu": config.get("proximal_mu", 0.0), "clients": config["clients"], "seed": config["seed"],
           "features": manifest["num_features"], "classes": manifest["num_classes"],
           "data-root": "/never-mounted-client-data", "artifact-dir": "/artifacts"}
    text = " ".join(f"{k}={_flower_literal(v)}" for k, v in cfg.items())
    extra = {"use-secagg": mode == "secagg", "num-shares": CLIENTS, "reconstruction-threshold": 2,
             "timeout": 60.0, "wire-capture": True}
    return text + " " + " ".join(f"{k}={_flower_literal(v)}" for k, v in extra.items())


def flower_command(prefix: list[str], app_dir: Path, run_config: str) -> list[str]:
    return prefix + ["run", "--rm", "--no-deps", "-v", f"{app_dir.resolve()}:/alt-app:ro", "--entrypoint", "sh",
                     "flwr-cli", "-c",
                     'cp -r /alt-app /tmp/alt-app && exec flwr run /tmp/alt-app fdl-docker --stream --run-config "$0"',
                     run_config]


def plan_commands(prefix: list[str], app_dir: Path, run_config: str) -> list[list[str]]:
    """Every Docker command of one run, in order. Used by --dry-run and by the tests."""
    return [prefix + ["up", "--build", "-d"],
            flower_command(prefix, app_dir, run_config),
            prefix + ["stop", "-t", "30", *RELAYS],
            prefix + ["logs", "--no-color"],
            prefix + ["down", "--remove-orphans"]]


def check_capture_outputs(run_dir: Path) -> None:
    """Raise unless the server capture is complete and every relay wrote a log with traffic."""
    summary = run_dir / "server" / "wire" / "capture_summary.json"
    if not summary.is_file():
        raise RuntimeError("capture summary is missing")
    data = json.loads(summary.read_text())
    if not data.get("completed"):
        raise RuntimeError(f"capture is not complete: {data.get('error')}")
    for i in range(CLIENTS):
        log = run_dir / "relay" / f"site-{i}.connections.jsonl"
        if not log.is_file() or not log.read_text().strip():
            raise RuntimeError(f"relay log for site {i} is missing or empty")
        closes = [json.loads(x) for x in log.read_text().splitlines() if '"event": "close"' in x]
        if not closes or sum(c["c2s"]["bytes"] for c in closes) == 0:
            raise RuntimeError(f"relay for site {i} recorded no traffic")


def execute_run(seed: int, mode: str, root: Path, *, timeout: int, app_dir: Path) -> dict:
    from prepare_flower_deployment import prepare
    from run_flower_docker import run_command, run_flower_cli
    from verify_flower_attestations import verify as verify_receipts
    from verify_flower_topology import live_certify, static_certify
    from feddisclosure.training.wire_capture import prepare_view_partitions, run_config_dict

    name = run_name(seed, mode)
    run_dir, data_dir = root / "runs" / name, root / "data" / f"seed{seed}"
    env_file = root / "env" / f"{name}.env"
    cfg_path = root / "configs" / f"wire-seed{seed}.yaml"
    meta = {"schema_version": 1, "seed": seed, "mode": mode, "ok": False, "data_dir": str(data_dir),
            "relay_dir": "relay", "capture_dir": "server/wire", "started_utc": datetime.now(timezone.utc).isoformat()}
    if (run_dir / "wire_run.json").exists() or (run_dir / "server" / "final_model.pt").exists():
        raise SystemExit(f"Refusing stale output in {run_dir}. Choose a fresh output directory")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        prepare_view_partitions(data_dir, seed)
    if not cfg_path.exists():
        cfg_path.write_text(yaml.safe_dump(run_config_dict(seed)))
    config = yaml.safe_load(cfg_path.read_text())
    manifest = json.loads((data_dir / "manifest.json").read_text())
    prepare(cfg_path, data_dir, run_dir, env_file)
    relay_dir = run_dir / "relay"
    relay_dir.mkdir(exist_ok=True)
    os.chmod(relay_dir, 0o777)   # the relay container runs as a different unprivileged user
    static_certify(ROOT / COMPOSE_BASE)
    prefix = compose_prefix(env_file)
    run_config = run_config_string(config, manifest, mode)
    commands = plan_commands(prefix, app_dir, run_config)
    meta["flower_command"] = commands[1]
    started = False
    try:
        pre = run_command(["docker", "info"], timeout=30)
        if pre.returncode:
            raise RuntimeError("Docker daemon unavailable: " + pre.stderr[-400:])
        built = run_command(commands[0], timeout=1200)
        (run_dir / "compose_start_stdout.log").write_text(built.stdout)
        (run_dir / "compose_start_stderr.log").write_text(built.stderr)
        if built.returncode:
            raise RuntimeError(f"Docker Compose startup failed: exit={built.returncode}")
        started = True
        prep = json.loads((run_dir / "deployment_preparation.json").read_text())
        topology = None
        for _ in range(30):
            try:
                topology = live_certify(ROOT / COMPOSE_BASE, env_file, prep)
                break
            except Exception:
                time.sleep(2)
        if topology is None:
            raise RuntimeError("Flower services did not become running")
        for relay in RELAYS:
            if not run_command(prefix + ["ps", "-q", relay]).stdout.strip():
                raise RuntimeError(f"{relay} is not running")
        (run_dir / "docker_topology.json").write_text(json.dumps(topology, indent=2) + "\n")
        attempts = []
        for attempt in range(1, 7):
            code, out, err, timed, _ = run_flower_cli(commands[1], timeout, prefix, run_dir, None, "before-run",
                                                      watch=topology["container_ids"])
            unreachable = "Connection to the SuperLink is unavailable" in out + err and not list(
                (run_dir / "clients").glob("site-*/events.jsonl"))
            attempts.append({"attempt": attempt, "exit_code": code, "superlink_unreachable": unreachable})
            if not unreachable or timed:
                break
            time.sleep(5)
        for tmp in ("_cli.out", "_cli.err"):
            (run_dir / tmp).unlink(missing_ok=True)
        (run_dir / "flower_cli_stdout.log").write_text(out)
        (run_dir / "flower_cli_stderr.log").write_text(err)
        meta.update(cli_attempts=attempts, exit_code=code, timed_out=timed)
        # SIGTERM lets each relay write its close records. The analysis needs them.
        run_command(commands[2], timeout=120)
        if code:
            raise RuntimeError(f"Flower CLI exit code {code}")
        if not (run_dir / "server" / "final_model.pt").is_file():
            raise RuntimeError("final model missing")
        check_capture_outputs(run_dir)
        verify_receipts(run_dir, manifest, int(config["rounds"]), require_evaluation=False)
        meta["ok"] = True
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if started:
            logs = run_command(commands[3], timeout=60)
            (run_dir / "docker_services.log").write_text(logs.stdout + "\n" + logs.stderr)
            run_command(commands[2], timeout=120)
            run_command(commands[4], timeout=120)
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "wire_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=5, help="seeds 0 to N-1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--timeout", type=int, default=600, help="seconds for each flwr run")
    p.add_argument("--app-dir", type=Path, default=ROOT / APP_DIR)
    p.add_argument("--keep-going", action="store_true", help="continue with later runs after a failure")
    p.add_argument("--dry-run", action="store_true", help="print the Docker commands and run nothing")
    a = p.parse_args(argv)
    if a.seeds < 1:
        p.error("--seeds must be at least 1")
    root = a.output.resolve()
    if a.dry_run:
        for seed in range(a.seeds):
            for mode in a.modes:
                env_file = root / "env" / f"{run_name(seed, mode)}.env"
                cfg = {"rounds": 1, "local_epochs": 1, "batch_size": 8, "learning_rate": 0.15, "clients": CLIENTS,
                       "seed": seed}
                manifest = {"num_features": 64, "num_classes": 10}
                print(f"# run {run_name(seed, mode)} (data {root / 'data' / f'seed{seed}'}, run dir {root / 'runs' / run_name(seed, mode)})")
                for cmd in plan_commands(compose_prefix(env_file), a.app_dir, run_config_string(cfg, manifest, mode)):
                    print(shlex.join(cmd))
        print("# dry run: nothing was executed")
        return 0
    if shutil.which("docker") is None:
        print("Docker CLI missing. Nothing was captured.", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(a.seeds):
        for mode in a.modes:
            meta = execute_run(seed, mode, root, timeout=a.timeout, app_dir=a.app_dir)
            results.append({"seed": seed, "mode": mode, "ok": meta["ok"], "error": meta.get("error")})
            print(json.dumps(results[-1]), flush=True)
            if not meta["ok"] and not a.keep_going:
                break
        else:
            continue
        break
    failed = [r for r in results if not r["ok"]]
    complete = len(results) == a.seeds * len(a.modes) and not failed
    (root / "capture_status.json").write_text(json.dumps({"complete": complete, "runs": results}, indent=2) + "\n")
    if not complete:
        print("CAPTURE INCOMPLETE. No claim can be made from this directory.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
