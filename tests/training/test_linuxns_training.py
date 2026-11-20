"""Contract tests: do not mistake Linux namespace training for Flower."""
from pathlib import Path
import json
import pytest
from scripts.linuxns_worker import validate_mapping
from scripts.run_linuxns_reference import run
from scripts.verify_linuxns_reference import verify


def test_namespace_site_mapping_requires_own_institution(tmp_path):
    root = tmp_path / "data"
    target = tmp_path / "mount"
    for i in range(3):
        (root / f"institution-{i:02d}").mkdir(parents=True)
    target.mkdir()
    validate_mapping(root, root / "institution-00", target, 0)
    with pytest.raises(ValueError, match="Unsafe site mapping"):
        validate_mapping(root, root / "institution-01", target, 0)
    with pytest.raises(ValueError, match="Unsafe site mapping"):
        validate_mapping(root, root / "institution-00", root / "other", 0)
    with pytest.raises(ValueError, match="three-site"):
        validate_mapping(root, root / "institution-00", target, 4)


def test_namespace_trust_boundary_documented():
    import inspect
    source = inspect.getsource(run)
    assert "linux_namespace_isolated_torch_NOT_DOCKER_NOT_FLOWER" in source
    assert '"native_docker_verified": False' in source
    assert '"native_flower_verified": False' in source


@pytest.mark.slow
def test_namespace_training_integration(tmp_path):
    """Runs only on a Linux host permitting unprivileged user/mount/PID namespaces."""
    import subprocess, shutil, sys
    if sys.platform != "linux" or not shutil.which("unshare"):
        pytest.skip("Linux unshare unavailable")
    pre = subprocess.run(["unshare", "-Urmpf", "true"], capture_output=True, timeout=5)
    if pre.returncode:
        pytest.skip("User/mount/PID namespaces not allowed by kernel policy")
    import yaml
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"id":"ns-integration", "seed":43, "dataset":"sklearn_digits",
                                     "partition":"iid", "clients":3, "rounds":1,
                                     "local_epochs":1, "batch_size":16, "learning_rate":0.15,
                                     "source_limit":120}))
    result = run(config, tmp_path / "data", tmp_path / "evidence")
    assert result["all_site_mounts_isolated"] and result["all_three_clients_trained"]
    assert result["native_flower_verified"] is False
