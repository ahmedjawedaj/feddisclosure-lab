import argparse
import json
from feddisclosure.evaluation.membership import evaluate_membership


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(evaluate_membership(args.scores, args.truth, args.output), indent=2))


if __name__ == "__main__":
    main()
