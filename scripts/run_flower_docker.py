"""Native 3 or 4 SuperNode Flower certification runner (NOT Flower simulation).

One Docker host is sufficient for independent container filesystems. This is
an intentionally insecure local lab network, not a multi-host/TLS deployment.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

from prepare_flower_deployment import prepare
from verify_flower_topology import live_certify, static_certify, SERVICES
from verify_flower_attestations import verify as verify_receipts

ROOT = Path(__file__).resolve().parents[1]


def _flower_literal(value):
    """Flower 1.27 run-config grammar: bool/int/float bare, strings double-quoted."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def run_flower_cli(command, timeout, command_prefix, run_dir, fault_client, fault_when, target="clientexec",
                   watch=None):
    """Run the Flower CLI, optionally stopping one ClientApp container (fault test).

    watch: {service: container_id} for fail-fast. If any watched container stops running while the
    CLI is in progress the CLI is killed at once and the loss is recorded (return code 125).
    """
    victims = [] if fault_client is None else (list(fault_client) if isinstance(fault_client, (list, tuple))
                                               else [fault_client])
    events = {"fault_injected": False, "fault_client": victims[0] if len(victims) == 1 else (victims or None),
              "fault_when": fault_when, "fault_target": target, "participant_lost": None}

    # Resolve container IDs up front so an abrupt crash is one fast `docker kill` per victim.
    victim_ids = {}
    if victims and fault_when == "during-round-2":
        for v in victims:
            victim_ids[v] = run_command(command_prefix + ["ps", "-q", f"{target}-{v}"], timeout=30).stdout.strip()

    def inject(verb):
        if verb == "kill" and victim_ids and all(victim_ids.values()):
            codes = [run_command(["docker", "kill", victim_ids[v]], timeout=30).returncode for v in victims]
        else:
            codes = [run_command(command_prefix + [verb, f"{target}-{v}"], timeout=60).returncode for v in victims]
        events.update(fault_injected=all(c == 0 for c in codes), fault_verb=verb,
                      fault_utc=datetime.now(timezone.utc).isoformat())
    out_path, err_path = run_dir / "_cli.out", run_dir / "_cli.err"
    if victims and fault_when == "before-run":
        inject("stop")
    with out_path.open("w") as out, err_path.open("w") as err:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, text=True)
        deadline = time.monotonic() + timeout
        started_at = last_check = time.monotonic()
        timed_out = lost_early = False
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                timed_out = True
                break
            if (victims and fault_when in ("after-first-train", "during-round-2")
                    and not events["fault_injected"]):
                needed = 1 if fault_when == "after-first-train" else 2
                done = [f for f in (run_dir / "clients").glob("site-*/events.jsonl")
                        if f.read_text().count('"event_type": "train"') >= needed]
                if done:
                    # during-round-2 is an abrupt crash (SIGKILL) after every client already finished
                    # the key-sharing stage of round 2, so the loss happens inside the protocol.
                    inject("kill" if fault_when == "during-round-2" else "stop")
            if watch and time.monotonic() - last_check >= 1.0:
                last_check = time.monotonic()
                lost = _stopped_services(watch)
                if lost:
                    events["participant_lost"] = {"services": lost,
                                                  "utc": datetime.now(timezone.utc).isoformat(),
                                                  "seconds_after_start": round(time.monotonic() - started_at, 2)}
                    proc.kill()
                    lost_early = True
                    break
            waiting_for_fault = bool(victims) and fault_when != "before-run" and not events["fault_injected"]
            time.sleep(0.05 if waiting_for_fault else 0.5)
        proc.wait()
    code = 125 if lost_early else (124 if timed_out else proc.returncode)
    return code, out_path.read_text(), err_path.read_text(), timed_out, events


def _stopped_services(watch):
    """Services whose container is no longer running (docker inspect, one call for all)."""
    names, ids = list(watch), list(watch.values())
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", *ids],
                         capture_output=True, text=True, check=False)
    states = out.stdout.split()
    if len(states) != len(ids):
        return names  # a container vanished entirely
    return [n for n, running in zip(names, states) if running != "true"]


def run_command(command, *, timeout=120, cwd=ROOT):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/fedavg_digits_iid.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/training/digits-docker-iid-v012")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Where the run evidence goes. The default is a new directory "
                             "reports/results/native_flower/local/<UTC time>, never the saved v012 evidence. "
                             "Pass the directory explicitly to write somewhere else.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.flower.generated")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--app-dir", type=Path, default=None,
                        help="Alternative Flower app directory (code only) mounted read-only into the CLI")
    parser.add_argument("--extra-run-config", default="",
                        help="Additional Flower run-config tokens, e.g. 'use-secagg=true'")
    parser.add_argument("--no-fail-fast", action="store_true",
                        help="Keep waiting when a participant container stops (needed only for protocols "
                             "that are meant to tolerate dropouts, e.g. SecAgg+)")
    parser.add_argument("--no-require-evaluation", action="store_true",
                        help="Protocols without client evaluation (SecAgg+); recorded in the evidence")
    parser.add_argument("--fault-stop-client", type=int, choices=[0, 1, 2, 3], action="append", default=None,
                        help="Fault injection: stop this institution's ClientApp container")
    parser.add_argument("--fault-target", choices=["clientexec", "supernode"], default="clientexec",
                        help="Which service of the chosen institution to stop")
    parser.add_argument("--fault-when", choices=["before-run", "after-first-train", "during-round-2"],
                        default="before-run")
    a = parser.parse_args()
    if a.run_dir is None:
        a.run_dir = ROOT / "reports/results/native_flower/local" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        print(f"no --run-dir given, writing to {a.run_dir}")
    run_dir = a.run_dir.resolve()
    prep = prepare(a.config.resolve(), a.data_dir.resolve(), run_dir, a.env_file)
    config = yaml.safe_load(a.config.read_text())
    manifest = json.loads((a.data_dir / "manifest.json").read_text())
    compose_file = ROOT / ("compose.flower.yaml" if int(config["clients"]) == 3
                           else f"compose.flower.n{int(config['clients'])}.yaml")
    static = static_certify(compose_file)
    (run_dir / "static_topology.json").write_text(json.dumps(static, indent=2) + "\n")
    if a.prepare_only:
        print(json.dumps({"preparation_passed": True, "static_topology": True,
                          "native_flower_verified": False, "run_dir": str(run_dir)}, indent=2))
        return

    command_prefix = ["docker", "compose", "--env-file", str(a.env_file.resolve()),
                      "-f", str(compose_file)]
    evidence = {"engine": "Flower ServerApp/ClientApp", "runtime_mode": "deployment",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": {**config, "algorithm": config.get("algorithm", "fedavg")},
                "docker_isolation_verified": False, "client_attestation_verified": False,
                "final_model_exists": False, "summary_exists": False,
                "exit_code": 2, "timed_out": False, "native_flower_verified": False,
                "warning": "Insecure single-host research deployment. No formal privacy protection."}
    folder = run_dir / "server"
    record = folder / "flower_run.json"
    if (folder / "final_model.pt").exists() or record.exists():
        raise SystemExit("Refusing stale Flower output; choose a fresh run directory")
    if shutil.which("docker") is None:
        evidence["error"] = "Docker CLI missing; cannot run native Flower"
        record.write_text(json.dumps(evidence, indent=2) + "\n")
        raise SystemExit(2)
    pre = run_command(["docker", "info"], timeout=20)
    if pre.returncode:
        evidence["error"] = "Docker daemon unavailable: " + pre.stderr[-800:]
        record.write_text(json.dumps(evidence, indent=2) + "\n")
        raise SystemExit(2)

    started = False
    try:
        built = run_command(command_prefix + ["up", "--build", "-d"], timeout=900)
        (run_dir / "compose_start_stdout.log").write_text(built.stdout)
        (run_dir / "compose_start_stderr.log").write_text(built.stderr)
        if built.returncode:
            raise RuntimeError(f"Docker Compose startup failed: exit={built.returncode}")
        started = True
        for attempt in range(30):
            try:
                topology = live_certify(compose_file, a.env_file, prep)
                break
            except (ValueError, OSError, subprocess.CalledProcessError):
                time.sleep(2)
        else:
            raise RuntimeError(f"{len(static['services'])} native Flower services did not become running")
        (run_dir / "docker_topology.json").write_text(json.dumps(topology, indent=2) + "\n")
        evidence["docker_isolation_verified"] = topology["native_verified"]
        cfg = {"num-server-rounds": config["rounds"], "local-epochs": config["local_epochs"],
               "batch-size": config["batch_size"], "learning-rate": config["learning_rate"],
               "proximal-mu": config.get("proximal_mu", 0.0), "clients": config["clients"],
               "seed": config["seed"], "features": manifest["num_features"],
               "classes": manifest["num_classes"], "data-root": "/never-mounted-client-data",
               "artifact-dir": "/artifacts"}
        run_config = " ".join(f"{k}={_flower_literal(v)}" for k, v in cfg.items())
        if a.extra_run_config:
            run_config += " " + a.extra_run_config
        if a.app_dir is not None:
            # `flwr run` migrates pyproject.toml in place. Mount the code read-only and let the
            # CLI container work on a private copy in its own /tmp.
            evidence["app_dir"] = str(a.app_dir)
            flower_cmd = command_prefix + [
                "run", "--rm", "--no-deps", "-v", f"{a.app_dir.resolve()}:/alt-app:ro",
                "--entrypoint", "sh", "flwr-cli", "-c",
                'cp -r /alt-app /tmp/alt-app && exec flwr run /tmp/alt-app fdl-docker --stream --run-config "$0"',
                run_config]
        else:
            flower_cmd = command_prefix + ["run", "--rm", "--no-deps", "flwr-cli",
                                           "run", ".", "fdl-docker", "--stream", "--run-config", run_config]
        evidence["flower_command"] = flower_cmd
        begun = time.monotonic()
        # Containers report "running" before the SuperLink accepts connections. Retry ONLY when the
        # CLI could not reach the SuperLink at all (no run started, nothing trained). Every attempt
        # is recorded. Any other failure, including a timeout, is final.
        attempts = []
        for attempt in range(1, 7):
            code, out, err, timed, fault = run_flower_cli(
                flower_cmd, a.timeout, command_prefix, run_dir, a.fault_stop_client, a.fault_when,
                a.fault_target, watch=None if a.no_fail_fast else topology["container_ids"])
            unreachable = ("Connection to the SuperLink is unavailable" in out + err
                           and not list((run_dir / "clients").glob("site-*/events.jsonl")))
            attempts.append({"attempt": attempt, "exit_code": code, "superlink_unreachable": unreachable})
            if not unreachable or timed:
                break
            time.sleep(5)
        evidence["cli_attempts"] = attempts
        evidence["exit_code"], evidence["timed_out"] = code, timed
        if a.fault_stop_client:
            evidence["fault_injection"] = fault
        evidence["fail_fast"] = not a.no_fail_fast
        if fault.get("participant_lost"):
            evidence["participant_lost"] = fault["participant_lost"]
        for tmp in ("_cli.out", "_cli.err"):
            (run_dir / tmp).unlink(missing_ok=True)
        evidence["seconds"] = round(time.monotonic() - begun, 2)
        (run_dir / "flower_cli_stdout.log").write_text(out)
        (run_dir / "flower_cli_stderr.log").write_text(err)
        evidence["final_model_exists"] = (folder / "final_model.pt").is_file()
        evidence["summary_exists"] = (folder / "flower_summary.json").is_file()
        if fault.get("participant_lost"):
            raise RuntimeError("Participant lost during the run (fail-fast): "
                               + ", ".join(fault["participant_lost"]["services"]))
        if evidence["exit_code"] or not evidence["final_model_exists"] or not evidence["summary_exists"]:
            raise RuntimeError("Native Flower execution incomplete. Inspect real Docker and CLI logs")
        receipts = verify_receipts(run_dir, manifest, int(config["rounds"]),
                                   require_evaluation=not a.no_require_evaluation)
        evidence["evaluation_receipts_required"] = not a.no_require_evaluation
        (run_dir / "client_attestations.json").write_text(json.dumps(receipts, indent=2) + "\n")
        evidence["client_attestation_verified"] = receipts["client_attestations_verified"]
        evidence["participating_clients"] = list(receipts["sites"])
        evidence["native_flower_verified"] = True
    except Exception as exc:
        evidence["error"] = str(exc)
    finally:
        if started:
            logs = run_command(command_prefix + ["logs", "--no-color"], timeout=60)
            (run_dir / "docker_services.log").write_text(logs.stdout + "\n" + logs.stderr)
            if not a.keep_running:
                down = run_command(command_prefix + ["down", "--remove-orphans"], timeout=90)
                (run_dir / "compose_down.log").write_text(down.stdout + "\n" + down.stderr)
        record.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if not evidence["native_flower_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
