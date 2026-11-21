"""Native 3-SuperNode Flower certification runner (NOT Flower simulation).

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


def run_flower_cli(command, timeout, command_prefix, run_dir, fault_client, fault_when, target="clientexec"):
    """Run the Flower CLI, optionally stopping one ClientApp container (fault test)."""
    events = {"fault_injected": False, "fault_client": fault_client, "fault_when": fault_when,
              "fault_target": target}
    out_path, err_path = run_dir / "_cli.out", run_dir / "_cli.err"
    if fault_client is not None and fault_when == "before-run":
        stop = run_command(command_prefix + ["stop", f"{target}-{fault_client}"], timeout=60)
        events.update(fault_injected=stop.returncode == 0, fault_utc=datetime.now(timezone.utc).isoformat())
    with out_path.open("w") as out, err_path.open("w") as err:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, text=True)
        deadline = time.monotonic() + timeout
        timed_out = False
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                timed_out = True
                break
            if fault_client is not None and fault_when == "after-first-train" and not events["fault_injected"]:
                done = [f for f in (run_dir / "clients").glob("site-*/events.jsonl")
                        if '"event_type": "train"' in f.read_text()]
                if done:
                    stop = run_command(command_prefix + ["stop", f"{target}-{fault_client}"], timeout=60)
                    events.update(fault_injected=stop.returncode == 0,
                                  fault_utc=datetime.now(timezone.utc).isoformat())
            time.sleep(0.5)
        proc.wait()
    return (124 if timed_out else proc.returncode), out_path.read_text(), err_path.read_text(), timed_out, events


def run_command(command, *, timeout=120, cwd=ROOT):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/fedavg_digits_iid.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/training/digits-docker-iid-v012")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/results/native_flower/v012")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.flower.generated")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--fault-stop-client", type=int, choices=[0, 1, 2], default=None,
                        help="Fault injection: stop this institution's ClientApp container")
    parser.add_argument("--fault-target", choices=["clientexec", "supernode"], default="clientexec",
                        help="Which service of the chosen institution to stop")
    parser.add_argument("--fault-when", choices=["before-run", "after-first-train"],
                        default="before-run")
    a = parser.parse_args()
    run_dir = a.run_dir.resolve()
    prep = prepare(a.config.resolve(), a.data_dir.resolve(), run_dir, a.env_file)
    config = yaml.safe_load(a.config.read_text())
    manifest = json.loads((a.data_dir / "manifest.json").read_text())
    static = static_certify(ROOT / "compose.flower.yaml")
    (run_dir / "static_topology.json").write_text(json.dumps(static, indent=2) + "\n")
    if a.prepare_only:
        print(json.dumps({"preparation_passed": True, "static_topology": True,
                          "native_flower_verified": False, "run_dir": str(run_dir)}, indent=2))
        return

    command_prefix = ["docker", "compose", "--env-file", str(a.env_file.resolve()),
                      "-f", str(ROOT / "compose.flower.yaml")]
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
                topology = live_certify(ROOT / "compose.flower.yaml", a.env_file, prep)
                break
            except (ValueError, OSError, subprocess.CalledProcessError):
                time.sleep(2)
        else:
            raise RuntimeError("Eight native Flower services did not become running")
        (run_dir / "docker_topology.json").write_text(json.dumps(topology, indent=2) + "\n")
        evidence["docker_isolation_verified"] = topology["native_verified"]
        cfg = {"num-server-rounds": config["rounds"], "local-epochs": config["local_epochs"],
               "batch-size": config["batch_size"], "learning-rate": config["learning_rate"],
               "proximal-mu": config.get("proximal_mu", 0.0), "clients": config["clients"],
               "seed": config["seed"], "features": manifest["num_features"],
               "classes": manifest["num_classes"], "data-root": "/never-mounted-client-data",
               "artifact-dir": "/artifacts"}
        run_config = " ".join(f"{k}={_flower_literal(v)}" for k, v in cfg.items())
        flower_cmd = command_prefix + ["run", "--rm", "--no-deps", "flwr-cli",
                                       "run", ".", "fdl-docker", "--stream", "--run-config", run_config]
        evidence["flower_command"] = flower_cmd
        begun = time.monotonic()
        code, out, err, timed, fault = run_flower_cli(
            flower_cmd, a.timeout, command_prefix, run_dir, a.fault_stop_client, a.fault_when,
            a.fault_target)
        evidence["exit_code"], evidence["timed_out"] = code, timed
        if a.fault_stop_client is not None:
            evidence["fault_injection"] = fault
        for tmp in ("_cli.out", "_cli.err"):
            (run_dir / tmp).unlink(missing_ok=True)
        evidence["seconds"] = round(time.monotonic() - begun, 2)
        (run_dir / "flower_cli_stdout.log").write_text(out)
        (run_dir / "flower_cli_stderr.log").write_text(err)
        evidence["final_model_exists"] = (folder / "final_model.pt").is_file()
        evidence["summary_exists"] = (folder / "flower_summary.json").is_file()
        if evidence["exit_code"] or not evidence["final_model_exists"] or not evidence["summary_exists"]:
            raise RuntimeError("Native Flower execution incomplete. Inspect real Docker and CLI logs")
        receipts = verify_receipts(run_dir, manifest, int(config["rounds"]))
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
