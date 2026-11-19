import argparse
import json
from feddisclosure.attacks.membership import attack_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", choices=["max_confidence", "negative_entropy", "top2_margin"], default="max_confidence")
    args = p.parse_args()
    print(json.dumps(attack_outputs(args.observation, args.output_dir, method=args.method), indent=2))


if __name__ == "__main__":
    main()
