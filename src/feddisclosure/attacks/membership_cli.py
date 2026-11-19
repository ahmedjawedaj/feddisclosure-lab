import argparse
import json
from feddisclosure.attacks.membership import attack_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observation", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    print(json.dumps(attack_outputs(args.observation, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
