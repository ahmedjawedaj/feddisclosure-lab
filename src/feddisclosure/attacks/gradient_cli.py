import argparse
import json
from feddisclosure.attacks.gradient_inversion import run_attack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--iterations", type=int, default=350)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    print(json.dumps(run_attack(args.observation, args.output_dir,
                                iterations=args.iterations, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
