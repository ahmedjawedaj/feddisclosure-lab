"""Execute the output-only attack followed by privileged ROC scoring."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from feddisclosure.evaluation.model_output_observer import create_output_observations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--per-class", type=int, default=6)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    data = Path(args.data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    sites = [data / part["file"] for part in manifest["client_partitions"]]
    test_file = data / manifest["evaluator_test"]["file"]
    out = Path(args.output_dir).resolve()
    contract = create_output_observations(args.checkpoint, sites, test_file, out,
                                          per_class=args.per_class, seed=args.seed)
    subprocess.run([sys.executable, "-m", "feddisclosure.attacks.membership_cli", "--observation",
                    str(out / "attacker" / "model_predictions.npz"), "--output-dir", str(out / "attacker")],
                   check=True, timeout=40)
    subprocess.run([sys.executable, "-m", "feddisclosure.evaluation.membership_cli", "--scores",
                    str(out / "attacker" / "scores.npz"), "--truth",
                    str(out / "evaluator_only" / "membership_labels.npz"),
                    "--output", str(out / "evaluation.json")], check=True, timeout=40)
    print(json.dumps({"contract": contract, "evaluation": json.loads((out / "evaluation.json").read_text())}, indent=2))


if __name__ == "__main__":
    main()
