"""CPU-only independent-process training reference (not a Flower run)."""
import argparse
import json
from pathlib import Path
from feddisclosure.training.offline_runner import run_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/fedavg_digits_iid.yaml")
    parser.add_argument("--data-dir", default="data/training/digits-iid")
    parser.add_argument("--output", default="reports/results/sprint2/torch_iid_reference.json")
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    result = run_reference(args.config, args.data_dir, args.output, include_references=args.references)
    print(json.dumps({"engine": result["engine"], "rounds": len(result["rounds"]),
                      "last_round": result["rounds"][-1]["public_test"], "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
