"""Execute the real Flower SuperLink/SuperNodes simulation. Not usable without flwr.

Uses the version-pinned Message API and captures evidence, including failures.
"""
import argparse
from datetime import datetime, timezone
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml
from feddisclosure.training.datasets import prepare_partitions

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/fedavg_digits_iid.yaml")
    parser.add_argument("--data-dir", default="data/training/digits-iid")
    parser.add_argument("--output-dir", default="reports/results/sprint2/flower")
    args = parser.parse_args()
    if find_spec("flwr") is None:
        raise SystemExit("Flower is missing: install Python 3.12 and pip install -e '.[ml,test]' on a network-enabled host")
    config = yaml.safe_load((ROOT / args.config).read_text())
    data_dir = (ROOT / args.data_dir).resolve()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in ("final_model.pt", "flower_summary.json", "flower_run.json"):
        if (output / stale).exists():
            raise SystemExit(f"Refusing to reuse existing Flower evidence: {output / stale}")
    if not (data_dir / "manifest.json").exists():
        prepare_partitions(data_dir, dataset=config["dataset"], clients=config["clients"],
                           partition=config["partition"], seed=config["seed"], limit=config.get("source_limit"))
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if any(manifest[k] != config[k] for k in ("clients", "seed", "partition", "source_limit")) or manifest["source"] != config["dataset"]:
        raise SystemExit("Prepared data does not match the experiment configuration")
    cfg = {"num-server-rounds": config["rounds"], "local-epochs": config["local_epochs"],
           "batch-size": config["batch_size"], "learning-rate": config["learning_rate"],
           "proximal-mu": float(config.get("proximal_mu", 0.0)),
           "clients": config["clients"], "seed": config["seed"],
           "features": manifest["num_features"], "classes": manifest["num_classes"],
           "data-root": str(data_dir), "artifact-dir": str(output)}
    # Flower 1.27 run-config grammar: bool, int and float bare, strings double-quoted.
    run_config = " ".join(f"{k}={json.dumps(v) if isinstance(v, str) else v}" for k, v in cfg.items())
    command = ["flwr", "run", ".", "--stream", "--run-config", run_config]
    start = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1200)
        stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        exit_code = 124
    elapsed = time.perf_counter() - start
    # Keep complete runtime evidence separate; do not claim success from an app that failed.
    (output / "flower_stdout.log").write_text(stdout)
    (output / "flower_stderr.log").write_text(stderr)
    evidence = {"engine": "Flower ServerApp/ClientApp", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": command, "config": config, "exit_code": exit_code, "timed_out": timed_out,
                "duration_seconds": round(elapsed, 4),
                "final_model_exists": (output / "final_model.pt").is_file(),
                "summary_exists": (output / "flower_summary.json").is_file()}
    (output / "flower_run.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if exit_code or not evidence["final_model_exists"] or not evidence["summary_exists"]:
        # Show the tail of Flower's own output in the log. Without it a hosted failure
        # is invisible because the full logs only exist in the uploaded artifact.
        for name, text in (("stderr", stderr), ("stdout", stdout)):
            tail = "\n".join(text.strip().splitlines()[-40:])
            if tail:
                print(f"--- last lines of Flower {name} ---\n{tail}\n--- end {name} ---", file=sys.stderr)
        raise SystemExit("Flower experiment failed. Inspect evidence logs. It is not a verified run.")


if __name__ == "__main__":
    main()
