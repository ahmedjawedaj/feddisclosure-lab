"""Run predefined experiments through the actual network gateway."""
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import time
import uuid
import httpx
import yaml
from feddisclosure.attacks.aggregate_difference import ObservedCount
from feddisclosure.evaluation.analytics import evaluate


def run_analytics_benchmark(
    gateway_url: str, research_key: str, config_path: str | Path, output_path: str | Path,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())
    truth = sum(site["positive"] for site in config["institutions"].values())
    results = {}
    with httpx.Client(timeout=12.0) as client:
        for policy in config["policies"]:
            experiment_id = f"{config['id']}-{policy}-{uuid.uuid4().hex[:12]}"
            observations = []
            rejected = 0
            responses = []
            start = time.perf_counter()
            for query_id in config["queries"]:
                resp = client.post(
                    f"{gateway_url.rstrip('/')}/v1/federated-count",
                    json={"experiment_id": experiment_id, "query_id": query_id, "policy": policy},
                    headers={"X-Research-Key": research_key},
                )
                if resp.status_code == 200:
                    count = resp.json()["count"]
                    observations.append(ObservedCount(query_id, count))
                    responses.append({"query_id": query_id, "released": True, "count": count})
                elif resp.status_code == 409:
                    rejected += 1
                    responses.append({"query_id": query_id, "released": False, "count": None})
                else:
                    raise RuntimeError(f"Gateway HTTP {resp.status_code}: {resp.text}")
            measurement = evaluate(observations, truth, len(config["queries"]), rejected, config["min_cell"])
            results[policy] = {
                "experiment_id": experiment_id,
                "releases": responses,
                "metrics": measurement.to_dict(),
                "elapsed_seconds": round(time.perf_counter() - start, 6),
            }
    result = {
        "experiment": config["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "attacker_observation": "gateway_aggregate_counts_only",
        "versions": {"feddisclosure-lab": _version("feddisclosure-lab"), "httpx": _version("httpx")},
        "results": results,
        "limitations": [
            "Synthetic counts; this is a controlled existence proof, not a population risk estimate.",
            "History near-duplicate control is a heuristic and provides no formal privacy guarantee.",
            "Ground truth is evaluator-only. Attacker code receives published counts only.",
            "No differential privacy or secure aggregation is implemented in this experiment.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "uninstalled (source tree)"
