"""Controlled public dataset preparation. Does not run in clients or gateway."""
import argparse
import json
from feddisclosure.training.datasets import prepare_partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/training/digits-iid")
    parser.add_argument("--dataset", choices=["sklearn_digits", "mnist"], default="sklearn_digits")
    parser.add_argument("--partition", choices=["iid", "label_skew", "dirichlet_0.5", "dirichlet_balanced_0.1", "dirichlet_balanced_0.5", "dirichlet_balanced_10"], default="iid")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()
    manifest = prepare_partitions(args.output, dataset=args.dataset, partition=args.partition,
                                  clients=args.clients, seed=args.seed, limit=args.limit)
    print(json.dumps({"clients": len(manifest["client_partitions"]), "source": manifest["source"],
                      "partition": manifest["partition"], "manifest": f"{args.output}/manifest.json"}, indent=2))


if __name__ == "__main__":
    main()
