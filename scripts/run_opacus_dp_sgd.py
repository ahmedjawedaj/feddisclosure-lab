"""Native Opacus vs ordinary local SGD. Fail closed if Opacus is missing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.task import DigitMLP, train_local
from feddisclosure.defenses.opacus_training import run_local_dp_sgd, OpacusUnavailable
import torch


def run(site_file: Path, data_dir: Path, out_dir: Path, *, seed=241, epochs=2,
        batch_size=32, epsilon=5.0, delta=1e-5, max_grad_norm=1.0):
    # Run MUST fail before writing artifacts when native Opacus isn't installed.
    try:
        import opacus  # noqa: F401
    except ImportError as exc:
        raise OpacusUnavailable("Native Opacus not installed; NO DP benchmark was executed") from exc
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite existing experiment")
    x, y = load_site_partition(site_file)
    torch.manual_seed(seed)
    initial = DigitMLP().state_dict()
    baseline = DigitMLP()
    baseline.load_state_dict(initial)
    baseline_metrics = train_local(baseline, x, y, epochs=epochs, batch_size=batch_size,
                                   lr=.15, seed=seed)
    dp_state, dp_metrics = run_local_dp_sgd(x, y, seed=seed, epochs=epochs,
                 batch_size=batch_size, learning_rate=.15, target_epsilon=epsilon,
                 target_delta=delta, max_grad_norm=max_grad_norm, initial_state=initial)
    out_dir.mkdir(parents=True)
    states = {"baseline": {k: v.detach().clone() for k, v in baseline.state_dict().items()},
              "opacus_dp": dp_state}
    evaluations = {}
    for name, state in states.items():
        checkpoint = out_dir / f"{name}.pt"
        torch.save(state, checkpoint)
        output = out_dir / f"{name}_evaluation.json"
        env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}")
        subprocess.run([sys.executable, "-m", "feddisclosure.training.evaluator_process", "test",
                        "--data-dir", str(data_dir), "--weights", str(checkpoint),
                        "--output", str(output), "--features", "64", "--classes", "10"],
                       env=env, cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
        evaluations[name] = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             "public_test": json.loads(output.read_text())}
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "experiment": "single_silo_local_dp_sgd_native_opacus_vs_sgd",
              "backend": "NATIVE Opacus only", "seed": seed, "n_train": len(y),
              "baseline": baseline_metrics, "dp_accounting": dp_metrics,
              "evaluations": evaluations,
              "limitations": ["This benchmark has not composed privacy costs across FL rounds or multiple institutions",
                              "No claim about Flower SecAgg+ or a deployed DP system",
                              "DP Poisson sampling and standard baseline SGD differ in realized minibatches"]}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/training/digits-iid")
    p.add_argument("--client-id", type=int, default=0)
    p.add_argument("--output-dir", default="reports/results/sprint4/opacus_native")
    p.add_argument("--epsilon", type=float, default=5.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=241)
    args = p.parse_args()
    if args.client_id not in (0, 1, 2):
        p.error("client-id must be 0..2")
    try:
        result = run(Path(args.data_root) / f"institution-{args.client_id:02d}" / "train.npz",
                     Path(args.data_root), Path(args.output_dir),
                     epsilon=args.epsilon, delta=args.delta, epochs=args.epochs, seed=args.seed)
    except OpacusUnavailable as exc:
        print(f"NOT VERIFIED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps({"dp_accounting": result["dp_accounting"],
                      "evaluations": result["evaluations"]}, indent=2))

if __name__ == "__main__":
    main()
