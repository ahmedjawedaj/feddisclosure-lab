"""Three-site FedAvg with *actual Linux namespace-isolated* PyTorch workers.

Experimental fallback only. It does NOT certify Docker, Flower, or SecAgg+.
Unlike the earlier reference, each local worker has its own user/mount/PID
namespace and can read only its own read-only /site bind mount. The coordinator
still runs on the trusted host, so this is not hostile-host containment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from feddisclosure.training.datasets import prepare_partitions
from feddisclosure.training.offline_runner import load_config, _call_evaluator
from feddisclosure.training.task import DigitMLP, fedavg_weighted

ROOT = Path(__file__).resolve().parents[1]


def run(config_path: Path, data_root: Path, output_dir: Path) -> dict:
    if sys.platform != "linux" or not shutil.which("unshare") or not shutil.which("mount"):
        raise RuntimeError("Linux user/mount/PID namespaces unavailable")
    cfg = load_config(config_path)
    if cfg["clients"] != 3:
        raise ValueError("Require exactly three isolated institutions")
    data_root, output_dir = data_root.resolve(), output_dir.resolve()
    if data_root == output_dir or data_root in output_dir.parents or output_dir in data_root.parents:
        raise ValueError("Dataset and evidence roots must be disjoint")
    if not data_root.exists():
        prepare_partitions(data_root, dataset=cfg["dataset"], clients=3,
                           partition=cfg["partition"], seed=cfg["seed"],
                           limit=cfg.get("source_limit"))
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for k in ("clients", "seed", "partition", "source_limit"):
        if manifest.get(k) != cfg.get(k):
            raise ValueError(f"Invalid or mismatched data manifest: {k}")
    if manifest["source"] != cfg["dataset"] or manifest.get("schema_version") != 2:
        raise ValueError("Invalid source or unsafe manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    model = DigitMLP(manifest["num_features"], manifest["num_classes"])
    report = {"engine": "linux_namespace_isolated_torch_NOT_DOCKER_NOT_FLOWER",
              "native_docker_verified": False, "native_flower_verified": False,
              "config": cfg, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "client_count": 3, "rounds": [], "model_parity": None}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fdl-ns-reference-") as td:
        scratch = Path(td)
        site_targets = []
        for site_id in range(3):
            path = scratch / f"site-mount-{site_id}"
            path.mkdir()
            site_targets.append(path)
        for round_number in range(1, cfg["rounds"] + 1):
            incoming = scratch / f"round-{round_number}-incoming.pt"
            torch.save(model.state_dict(), incoming)
            jobs = []
            for site in manifest["client_partitions"]:
                i = site["client_id"]
                source = data_root / f"institution-{i:02d}"
                target = site_targets[i]
                weights_out = scratch / f"round-{round_number}-site-{i}-weights.pt"
                metrics_out = scratch / f"round-{round_number}-site-{i}-metrics.json"
                proof = output_dir / f"round-{round_number}-site-{i}-namespace.json"
                worker_args = ["--site-file", str(target / "train.npz"),
                               "--validation-file", str(target / "val.npz"),
                               "--train-sha256", site["sha256"], "--validation-sha256", site["val_sha256"],
                               "--weights-in", str(incoming), "--weights-out", str(weights_out),
                               "--metrics-out", str(metrics_out),
                               "--features", str(manifest["num_features"]),
                               "--classes", str(manifest["num_classes"]),
                               "--epochs", str(cfg["local_epochs"]),
                               "--batch-size", str(cfg["batch_size"]),
                               "--lr", str(cfg["learning_rate"]),
                               "--proximal-mu", str(cfg.get("proximal_mu", 0.0)),
                               "--seed", str(cfg["seed"] + i)]
                command = ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork",
                           sys.executable, str(ROOT / "scripts/linuxns_worker.py"),
                           "--data-root", str(data_root), "--site-source", str(source),
                           "--site-target", str(target), "--site-id", str(i),
                           "--proof", str(proof), "--", *worker_args]
                jobs.append((site, command, proof, weights_out, metrics_out))
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            def execute(job):
                result = subprocess.run(job[1], cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=120, check=False)
                if result.returncode != 0:
                    (output_dir / f"round-{round_number}-site-{job[0]['client_id']}-failure.log").write_text(
                        result.stdout + "\n" + result.stderr)
                    raise RuntimeError(f"Namespace worker {job[0]['client_id']} failed: {result.stderr[-300:]}")
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(execute, jobs))
            proofs = [json.loads(job[2].read_text()) for job in jobs]
            if len({p["mount_namespace"] for p in proofs}) != 3 or len({p["pid_namespace"] for p in proofs}) != 3:
                raise RuntimeError("Clients did not execute in independent mount and PID namespaces")
            if any(p["own_train_sha256"] != site["sha256"] or not p["all_source_partitions_masked"]
                   or not p["own_mount_read_only"] for p, (site, *_) in zip(proofs, jobs)):
                raise RuntimeError("Isolated client filesystem boundaries not verified")
            weights = [torch.load(job[3], map_location="cpu", weights_only=True) for job in jobs]
            metrics = [json.loads(job[4].read_text()) for job in jobs]
            sizes = [site["samples"] for site, *_ in jobs]
            if any(m["num-examples"] != n for m,n in zip(metrics, sizes)):
                raise RuntimeError("Unexpected training sample counts")
            model.load_state_dict(fedavg_weighted(weights, sizes))
            aggregate = scratch / f"round-{round_number}-global.pt"
            torch.save(model.state_dict(), aggregate)
            evaluation = _call_evaluator("test", data_dir=data_root, weights=aggregate,
                                        output=scratch / f"round-{round_number}-public.json",
                                        features=manifest["num_features"], classes=manifest["num_classes"])
            report["rounds"].append({"round": round_number, "global_accuracy": evaluation["accuracy"],
                                     "global_loss": evaluation["loss"], "participants": 3,
                                     "namespace_ids_distinct": True})
    final = output_dir / "namespace_final_model.pt"
    torch.save(model.state_dict(), final)
    report["checkpoint_sha256"] = hashlib.sha256(final.read_bytes()).hexdigest()
    report["execution_seconds"] = round(time.perf_counter() - start, 3)
    report["all_site_mounts_isolated"] = True
    report["all_three_clients_trained"] = True
    report["limitations"] = ["Not Docker", "Not Flower", "Not formal process attestation",
                             "Trusted host orchestrator can read host data", "No DP or secure aggregation"]
    (output_dir / "linux_namespace_training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/training/fedavg_digits_iid.yaml")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/training/digits-namespace-iid")
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports/results/linux_namespace/isolated_iid")
    a = p.parse_args()
    result = run(a.config, a.data_dir, a.output_dir)
    print(json.dumps({"engine": result["engine"], "rounds": result["rounds"],
                      "all_site_mounts_isolated": result["all_site_mounts_isolated"],
                      "native_flower_verified": False}, indent=2))


if __name__ == "__main__":
    main()
