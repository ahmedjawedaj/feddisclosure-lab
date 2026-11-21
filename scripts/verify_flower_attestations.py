"""Check actual per-site ClientApp activity before accepting a Flower run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def verify(run_dir: Path, manifest: dict, rounds: int, *, require_evaluation: bool = True) -> dict:
    if len(manifest["client_partitions"]) < 3 or rounds < 1:
        raise ValueError("Expected at least three institutions and positive training rounds")
    events_by_site = {}
    run_ids = set()
    for site in manifest["client_partitions"]:
        i = site["client_id"]
        file = run_dir / "clients" / f"site-{i}" / "events.jsonl"
        if not file.is_file():
            raise ValueError(f"Missing actual ClientApp receipt for institution {i}")
        events = [json.loads(line) for line in file.read_text().splitlines() if line]
        train = [e for e in events if e.get("event_type") == "train"]
        evals = [e for e in events if e.get("event_type") == "evaluate"]
        if len(train) != rounds or (require_evaluation and len(evals) < 1):
            raise ValueError(f"Institution {i} did not complete each expected training round")
        for e in events:
            if e.get("site_id") != i or not e.get("run_id"):
                raise ValueError("Receipt identity missing or mismatched")
            run_ids.add(e["run_id"])
            expected = (site["sha256"] if e["event_type"] == "train" else site["val_sha256"])
            n = (site["samples"] if e["event_type"] == "train" else site["val_samples"])
            if e["dataset_sha256"] != expected or e["samples"] != n:
                raise ValueError("Receipt does not match prepared source partition")
            if e["event_type"] == "train" and not e.get("weights_sha256"):
                raise ValueError("Missing trained-model digest")
        events_by_site[str(i)] = {"train_receipts": len(train),
                                  "evaluation_receipts": len(evals),
                                  "trained_weight_sha256": [e["weights_sha256"] for e in train]}
    if len(run_ids) != 1:
        raise ValueError("Client receipts do not belong to one Flower run")
    return {"client_attestations_verified": True, "run_id": next(iter(run_ids)),
            "rounds": rounds, "sites": events_by_site,
            "limitation": "Receipts attest client processes; independent Docker inspect is required for filesystem isolation"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = verify(a.run_dir, json.loads(a.manifest.read_text()), a.rounds)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
