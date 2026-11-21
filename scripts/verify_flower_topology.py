"""Inspect the intended and, optionally, RUNNING Flower Docker trust boundaries.

Static YAML validation is not equivalent to OS-enforced isolation. Only
--native can set native_verified=True after live docker inspect calls succeed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ("superlink", "serverexec", "supernode-0", "supernode-1", "supernode-2",
            "clientexec-0", "clientexec-1", "clientexec-2")


def client_count(services: dict) -> int:
    """Institutions in a compose file. Indices must be contiguous from 0 and at least three."""
    n = sum(1 for name in services if name.startswith("clientexec-"))
    require(n >= 3 and all(f"clientexec-{i}" in services and f"supernode-{i}" in services
                           for i in range(n)), "Institution services are not contiguous")
    require(not any(name.startswith(("clientexec-", "supernode-")) and int(name.rsplit("-", 1)[1]) >= n
                    for name in services), "Unexpected extra institution service")
    return n


def service_names(n: int) -> tuple:
    return ("superlink", "serverexec", *(f"supernode-{i}" for i in range(n)),
            *(f"clientexec-{i}" for i in range(n)))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def static_certify(compose_path: Path) -> dict:
    data = yaml.safe_load(compose_path.read_text())
    services = data["services"]
    n = client_count(services)
    names = service_names(n)
    require(all(s in services for s in names) and "flwr-cli" in services,
            "Deployment topology incomplete")
    require(services["superlink"]["image"] == "flwr/superlink:1.27.0", "Unpinned SuperLink")
    require(services["superlink"]["command"][-2:] == ["--isolation", "process"],
            "Not a separate ServerApp runtime")
    require("9093" not in json.dumps(services["superlink"].get("ports", [])),
            "Control API must not be exposed to host")
    require("flwr/superexec" not in str(services["serverexec"].get("image", "")),
            "ServerApp executor must be custom project image")
    require("serverapp" in services["serverexec"]["command"], "Missing ServerApp executor")
    for role in ("superlink", "serverexec", "flwr-cli", *(f"supernode-{i}" for i in range(n))):
        mount_def = json.dumps(services[role].get("volumes", []))
        require("/site" not in mount_def and "FDL_SITE_" not in mount_def,
                f"Privileged service {role} has private client data")
    seen_site_mounts = set()
    for i in range(n):
        node = services[f"supernode-{i}"]
        exec_app = services[f"clientexec-{i}"]
        require(node["image"] == "flwr/supernode:1.27.0", "Unpinned SuperNode")
        require(f"partition-id={i} num-partitions={n}" in node["command"], "SuperNode ID mismatch")
        require("--isolation" in node["command"] and "process" in node["command"],
                "SuperNode is not separate-process")
        require(f"supernode-{i}:9094" in exec_app["command"], "ClientApp endpoint mismatch")
        require(exec_app["environment"]["FDL_SITE_ID"] == str(i), "Client identity mismatch")
        require(exec_app["environment"]["FDL_SITE_DATA_DIR"] == "/site", "No site-specific root")
        volumes = exec_app["volumes"]
        sites = [v for v in volumes if v.get("target") == "/site"]
        require(len(sites) == 1 and sites[0].get("read_only") is True,
                "Each client must have exactly one read-only /site bind mount")
        mount = sites[0]["source"]
        require(f"FDL_SITE_{i}_PATH" in mount and mount not in seen_site_mounts,
                "Client site mount is not unique")
        seen_site_mounts.add(mount)
        require(len(volumes) == 2 and volumes[1]["target"] == "/attestation"
                and f"site-{i}" in volumes[1]["source"],
                "Client may write only its own attestation path")
        require(not exec_app.get("ports") and not node.get("ports"),
                "Private institution must not publish host ports")
    return {"static_verified": True, "native_verified": False,
            "services": list(names), "client_readonly_mounts": n, "clients": n,
            "claim": "Static source topology only; Docker not yet executed"}


def execute(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=45)
    return completed.stdout.strip()


def live_certify(compose: Path, env: Path, preparation: dict) -> dict:
    static = static_certify(compose)
    n = static["clients"]
    require(len(preparation["sites"]) == n, "Prepared site count differs from the Compose topology")
    names = service_names(n)
    args = ["docker", "compose", "--env-file", str(env), "-f", str(compose)]
    containers = {}
    for service in names:
        container_id = execute(args + ["ps", "-q", service])
        require(container_id and "\n" not in container_id, f"Missing Docker container for {service}")
        containers[service] = container_id
    require(len(set(containers.values())) == len(containers), "Container identities are not independent")
    details = json.loads(execute(["docker", "inspect", *containers.values()]))
    require(len(details) == len(containers), "Incomplete Docker inspect output")
    info = dict(zip(names, details))
    for name, item in info.items():
        require(item["State"]["Running"], f"{name} is not running")
        require(len(item["NetworkSettings"]["Networks"]) == 1, "Unexpected extra network")
        require(not item["HostConfig"]["PortBindings"], f"{name} exposes host ports")
    networks = {next(iter(item["NetworkSettings"]["Networks"])) for item in details}
    require(len(networks) == 1, "Flower services are not on one dedicated network")
    for name in ("superlink", "serverexec", *(f"supernode-{i}" for i in range(n))):
        require(not any(m["Destination"] == "/site" for m in info[name]["Mounts"]),
                f"{name} can see a client dataset")
    for i in range(n):
        mounts = info[f"clientexec-{i}"]["Mounts"]
        sites = [m for m in mounts if m["Destination"] == "/site"]
        require(len(sites) == 1 and not sites[0]["RW"], f"Client {i} data mount is not read-only")
        require(Path(sites[0]["Source"]).resolve() == Path(preparation["sites"][i]["path"]),
                f"Client {i} received the wrong site directory")
        require(not any(m["Destination"] == "/site" for j in range(n) if j != i
                        for m in info[f"clientexec-{j}"]["Mounts"]
                        if m["Source"] == sites[0]["Source"]),
                "Two clients share a dataset path")
    static.update({"native_verified": True, "container_ids": containers,
                   "network_name": next(iter(networks)),
                   "claim": "Observed running Docker containers and distinct read-only site mounts"})
    return static


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compose", type=Path, default=ROOT / "compose.flower.yaml")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env.flower.generated")
    p.add_argument("--preparation", type=Path, default=ROOT / "reports/results/native_flower/v012/deployment_preparation.json")
    p.add_argument("--output", type=Path, default=ROOT / "reports/results/native_flower/v012/topology.json")
    p.add_argument("--native", action="store_true")
    a = p.parse_args()
    try:
        result = (live_certify(a.compose, a.env_file, json.loads(a.preparation.read_text()))
                  if a.native else static_certify(a.compose))
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
        result = {"static_verified": False, "native_verified": False, "error": str(exc)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if a.native and not result.get("native_verified") or not a.native and not result.get("static_verified"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
