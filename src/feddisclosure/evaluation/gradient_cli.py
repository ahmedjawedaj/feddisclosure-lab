import argparse
import json
from feddisclosure.evaluation.gradient import evaluate_attack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--recovered", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(evaluate_attack(args.ground_truth, args.recovered, args.output), indent=2))


if __name__ == "__main__":
    main()
