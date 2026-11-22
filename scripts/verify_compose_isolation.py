"""Native Docker inspection; assert distinct volumes, internal network, loopback port."""
import argparse
import json
import subprocess
from pathlib import Path


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/results/sprint1/docker_isolation.json")
    args = p.parse_args()
    names = ["institution-a", "institution-b", "institution-c", "gateway"]
    ids = [docker("compose", "ps", "-q", name) for name in names]
    if not all(ids):
        raise RuntimeError("Four Compose services required")
    containers = json.loads(docker("inspect", *ids))
    volume_names = []
    networks = []
    for name, item in zip(names, containers):
        mounts = [x["Name"] for x in item["Mounts"] if x["Type"] == "volume"]
        networks.append(set(item["NetworkSettings"]["Networks"]))
        published = item["NetworkSettings"]["Ports"]
        if name != "gateway":
            assert len(mounts) == 1, f"{name} must have one separate private DB volume"
            assert not any(v for v in published.values() if v is not None), f"{name} is unexpectedly exposed"
            volume_names += mounts
        else:
            host_ports = [b["HostIp"] for vals in published.values() if vals for b in vals]
            assert host_ports == ["127.0.0.1"], f"Gateway published off loopback: {host_ports}"
    assert len(set(volume_names)) == 3, "Shared institution volume would violate isolation"
    # Institutions must share exactly one network, and it must be internal-only.
    institution_networks = networks[:3]
    gateway_networks = networks[3]
    assert all(n == institution_networks[0] for n in institution_networks), "Institutions differ in networks"
    assert len(institution_networks[0]) == 1, "Institutions must be on exactly one network"
    network = next(iter(institution_networks[0]))
    network_config = json.loads(docker("network", "inspect", network))[0]
    assert network_config["Internal"] is True, "Institution network must be internal-only"
    # The gateway joins that network and may add a separate edge network so the host
    # can reach its loopback port. Institutions are never attached to the edge network.
    assert network in gateway_networks, "Gateway must be on the institution network"
    extra = sorted(gateway_networks - {network})
    assert len(extra) <= 1, f"Gateway has unexpected extra networks: {extra}"
    evidence = {"observed": True, "containers": names, "separate_data_volumes": True,
                "all_services_on_one_internal_network": True,
                "institutions_on_internal_network_only": True,
                "gateway_extra_networks": len(extra),
                "only_gateway_bound_to_host_loopback": True}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
