"""CLI for auditable experiment entry points."""
import argparse
import json
import os
from feddisclosure.evaluation.runner import run_analytics_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="FedDisclosure Lab research runner")
    sub = parser.add_subparsers(dest="command", required=True)
    benchmark = sub.add_parser("analytics-benchmark")
    benchmark.add_argument("--gateway", default="http://127.0.0.1:8080")
    benchmark.add_argument("--config", default="configs/analytics/differencing.yaml")
    benchmark.add_argument("--output", default="reports/results/analytics_latest.json")
    args = parser.parse_args()
    if args.command == "analytics-benchmark":
        result = run_analytics_benchmark(args.gateway, os.getenv("FDL_RESEARCH_KEY", "local-dev-research-key"), args.config, args.output)
        print(json.dumps({"output": args.output, "results": {p: s["metrics"] for p, s in result["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
