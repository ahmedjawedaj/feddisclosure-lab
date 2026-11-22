"""Static checks of compose.wire.yaml and the wire-capture workflow. NOT a Docker execution test."""
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

BASE = yaml.safe_load((ROOT / "compose.flower.n4.yaml").read_text())
WIRE_TEXT = (ROOT / "compose.wire.yaml").read_text()
WIRE = yaml.safe_load(WIRE_TEXT)
N = 4


def test_override_defines_one_relay_per_supernode_and_nothing_else_new():
    new = set(WIRE["services"]) - set(BASE["services"])
    assert new == {f"wire-relay-{i}" for i in range(N)}
    changed = set(WIRE["services"]) & set(BASE["services"])
    assert changed == {f"supernode-{i}" for i in range(N)}


def test_each_supernode_goes_through_its_own_relay_and_is_otherwise_unchanged():
    for i in range(N):
        cmd = WIRE["services"][f"supernode-{i}"]["command"]
        pos = cmd.index("--superlink")
        assert cmd[pos + 1] == f"wire-relay-{i}:9092"
        restored = list(cmd)
        restored[pos + 1] = "superlink:9092"
        assert restored == BASE["services"][f"supernode-{i}"]["command"]     # no drift from the base file
        assert f"wire-relay-{i}" in WIRE["services"][f"supernode-{i}"]["depends_on"]


def test_relays_sit_between_supernodes_and_superlink_on_the_same_network():
    for i in range(N):
        relay = WIRE["services"][f"wire-relay-{i}"]
        cmd = relay["command"]
        assert cmd[cmd.index("--upstream") + 1] == "superlink:9092"
        assert cmd[cmd.index("--name") + 1] == f"site-{i}"
        assert relay["networks"] == ["flower-lab"] and "superlink" in relay["depends_on"]
        assert relay["entrypoint"] == ["python", "/opt/wire_relay/relay.py"]
        assert "no-new-privileges:true" in relay["security_opt"]
    assert BASE["networks"]["flower-lab"]["driver"] == "bridge" and "networks" not in WIRE   # no new network


def test_no_port_is_published_to_the_host():
    assert all("ports" not in svc for svc in WIRE["services"].values())
    assert "ports:" not in WIRE_TEXT and "expose:" not in WIRE_TEXT


def test_relay_image_is_the_pinned_flower_image_and_the_script_is_mounted_read_only():
    for i in range(N):
        relay = WIRE["services"][f"wire-relay-{i}"]
        assert relay["image"] == "flwr/superexec:1.27.0" and "build" not in relay
        script = [v for v in relay["volumes"] if v["target"] == "/opt/wire_relay/relay.py"]
        assert len(script) == 1 and script[0]["read_only"] is True
        assert (ROOT / "docker/wire_relay/relay.py").is_file()


def test_relay_volume_is_the_run_directory_relay_folder_only():
    for i in range(N):
        mounts = [v for v in WIRE["services"][f"wire-relay-{i}"]["volumes"] if v["target"] == "/relay-out"]
        assert len(mounts) == 1 and mounts[0]["type"] == "bind"
        assert mounts[0]["source"].endswith("/relay") and "FDL_RUN_DIR" in mounts[0]["source"]
        assert not any("/site" in v["target"] or "FDL_SITE" in v["source"] for v in WIRE["services"][f"wire-relay-{i}"]["volumes"])


def test_relays_get_no_private_client_data_and_no_secrets():
    text = WIRE_TEXT.lower()
    assert "fdl_site_" not in text and "/site" not in text and "secret" not in text


def test_the_capture_app_uses_the_recording_serverapp_and_the_unchanged_clientapp():
    import tomllib
    app = tomllib.loads((ROOT / "docker/wire_relay/app/pyproject.toml").read_text())["tool"]["flwr"]["app"]
    assert app["components"]["serverapp"] == "feddisclosure.training.secagg_server_capture:app"
    assert app["components"]["clientapp"] == "feddisclosure.training.secagg_client:app"
    cfg = app["config"]
    assert (cfg["clients"], cfg["num-shares"], cfg["num-server-rounds"], cfg["local-epochs"], cfg["batch-size"]) == (4, 4, 1, 1, 8)
    assert cfg["learning-rate"] == 0.15 and cfg["wire-capture"] is True


def test_default_application_is_not_pointed_at_the_recording_serverapp():
    import tomllib
    default = tomllib.loads((ROOT / "experiments/flower_secagg/pyproject.toml").read_text())["tool"]["flwr"]["app"]
    assert default["components"]["serverapp"] == "feddisclosure.training.secagg_server:app"


# ---- workflow ----------------------------------------------------------------------------------

WORKFLOW = ROOT / ".github/workflows/wire-capture.yml"
DOC = yaml.safe_load(WORKFLOW.read_text())


def test_workflow_is_dispatch_only_with_minimal_permissions():
    triggers = DOC.get(True, DOC.get("on"))
    assert list(triggers) == ["workflow_dispatch"]
    assert DOC["permissions"] == {"contents": "read"}
    assert set(triggers["workflow_dispatch"]["inputs"]) == {"seeds", "iterations"}
    assert triggers["workflow_dispatch"]["inputs"]["seeds"]["default"] == "5"


def test_workflow_runs_on_ubuntu_latest_within_the_time_budget_and_pins_versions():
    job = DOC["jobs"]["wire-capture"]
    assert job["runs-on"] == "ubuntu-latest" and job["timeout-minutes"] <= 60
    text = WORKFLOW.read_text()
    assert "'torch==2.5.1'" in text and "'flwr==1.27.0'" in text
    uses = re.findall(r"uses: (\S+)", text)
    assert uses and all(re.fullmatch(r"[\w./-]+@v\d+", u) for u in uses)
    capture = next(s for s in job["steps"] if "run_wire_capture" in s.get("run", ""))
    assert capture["timeout-minutes"] <= 45


def test_workflow_passes_inputs_through_the_environment_not_inline_in_shell():
    run_text = "\n".join(s.get("run", "") for s in DOC["jobs"]["wire-capture"]["steps"])
    assert "${{" not in run_text
    assert DOC["jobs"]["wire-capture"]["env"]["WIRE_SEEDS"] == "${{ inputs.seeds }}"


def test_workflow_uploads_evidence_even_on_failure_without_model_weights():
    step = next(s for s in DOC["jobs"]["wire-capture"]["steps"] if s.get("uses", "").startswith("actions/upload-artifact"))
    assert step["if"] == "always()"
    assert "!reports/results/privacy/v0150/wire/hosted/**/final_model.pt" in step["with"]["path"]


def test_workflow_paths_exist_and_follow_the_repository_static_checker():
    import check_workflows_static as chk
    report = chk.check()["wire-capture.yml"]
    assert report["ok"], report
    assert report["paths_referenced"] >= 4
