"""Static checks of the 8-client Compose file, staging and workflow. NOT a Docker execution test."""
import json
from pathlib import Path
import re
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_compose_clients import generate  # noqa: E402
from verify_flower_topology import static_certify  # noqa: E402
import run_scaling_n8 as n8  # noqa: E402

COMPOSE = ROOT / "compose.flower.n8.yaml"
WORKFLOW = ROOT / ".github/workflows/scaling-n8.yml"
DOC = yaml.safe_load(WORKFLOW.read_text())


def test_compose_file_is_the_generated_one_and_passes_the_isolation_checks():
    assert COMPOSE.read_text() == generate((ROOT / "compose.flower.yaml").read_text(), 8)
    result = static_certify(COMPOSE)
    assert result["static_verified"] and result["clients"] == 8 and len(result["services"]) == 2 + 2 * 8


def test_every_institution_has_its_own_node_executor_and_read_only_site_mount():
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    sources = set()
    for i in range(8):
        assert f"partition-id={i} num-partitions=8" in services[f"supernode-{i}"]["command"]
        site = [v for v in services[f"clientexec-{i}"]["volumes"] if v["target"] == "/site"]
        assert len(site) == 1 and site[0]["read_only"] is True
        sources.add(site[0]["source"])
    assert len(sources) == 8
    assert not any("ports" in svc for svc in services.values())
    assert yaml.safe_load(COMPOSE.read_text())["name"] == "feddisclosure-flower"     # the sampler filters on this prefix


def test_config_and_runner_use_eight_clients_and_one_round():
    cfg = yaml.safe_load(n8.CONFIG.read_text())
    assert (cfg["clients"], cfg["rounds"], cfg["dataset"]) == (8, 1, "sklearn_digits")
    assert n8.COMPOSE == COMPOSE


def test_the_shared_runner_picks_this_compose_file_from_the_client_count():
    text = (ROOT / "scripts/run_flower_docker.py").read_text()
    assert 'f"compose.flower.n{int(config[\'clients\'])}.yaml"' in text


def test_staging_writes_eight_read_only_sites_and_eight_attestation_folders(tmp_path):
    report = n8.stage_clients(n8.CONFIG, tmp_path / "data", tmp_path / "run", tmp_path / "run.env")
    assert report["client_count"] == 8 and len(report["sites"]) == 8
    env = (tmp_path / "run.env").read_text()
    for i in range(8):
        assert f"FDL_SITE_{i}_PATH=" in env and f"FDL_SITE_{i}_TRAIN_SHA256=" in env
        assert (tmp_path / "run" / "clients" / f"site-{i}").is_dir()
    assert "FDL_SITE_8_PATH" not in env
    assert not (tmp_path / "run" / "evaluator_only").exists()
    assert json.loads((tmp_path / "run" / "deployment_preparation.json").read_text())["evaluator_not_mounted"] is True


def test_staging_refuses_overlapping_roots_and_a_changed_partition(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        n8.stage_clients(n8.CONFIG, tmp_path, tmp_path / "run", tmp_path / "run.env")
    n8.stage_clients(n8.CONFIG, tmp_path / "data", tmp_path / "run", tmp_path / "run.env")
    site = tmp_path / "data" / "institution-03" / "train.npz"
    site.write_bytes(site.read_bytes() + b"x")
    with pytest.raises(ValueError, match="integrity"):
        n8.stage_clients(n8.CONFIG, tmp_path / "data", tmp_path / "run2", tmp_path / "run2.env")


def test_runner_arguments_carry_the_secagg_settings_and_skip_evaluation_receipts(tmp_path):
    class A:
        num_shares, threshold, round_timeout, timeout, app_dir = 7, 4, 120.0, 900, n8.APP_DIR
    argv = n8.runner_argv(A, tmp_path / "d", tmp_path / "r", tmp_path / "e.env")
    extra = argv[argv.index("--extra-run-config") + 1]
    assert "use-secagg=true" in extra and "num-shares=7" in extra and "reconstruction-threshold=4" in extra
    assert "--no-require-evaluation" in argv and argv[argv.index("--config") + 1] == str(n8.CONFIG)


def test_dry_run_prints_the_plan_and_bad_secagg_settings_are_refused(tmp_path, capsys):
    assert n8.main(["--output", str(tmp_path), "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out
    for bad in (["--num-shares", "4", "--threshold", "2"], ["--num-shares", "7", "--threshold", "7"]):
        with pytest.raises(SystemExit):
            n8.main(["--output", str(tmp_path), "--dry-run", *bad])
    assert n8.LIMIT_GB == 6.0


# ---- workflow ----------------------------------------------------------------------------------

def test_workflow_is_dispatch_only_with_minimal_permissions():
    triggers = DOC.get(True, DOC.get("on"))
    assert list(triggers) == ["workflow_dispatch"]
    assert DOC["permissions"] == {"contents": "read"}
    assert set(triggers["workflow_dispatch"]["inputs"]) == {"num_shares", "threshold"}
    assert triggers["workflow_dispatch"]["inputs"]["num_shares"]["default"] == "8"


def test_workflow_runs_on_a_hosted_runner_pins_versions_and_keeps_evidence_on_failure():
    job = DOC["jobs"]["scaling-n8"]
    assert job["runs-on"] == "ubuntu-latest" and job["timeout-minutes"] <= 60
    text = WORKFLOW.read_text()
    assert "'torch==2.5.1'" in text and "'flwr==1.27.0'" in text
    uses = re.findall(r"uses: (\S+)", text)
    assert uses and all(re.fullmatch(r"[\w./-]+@v\d+", u) for u in uses)
    steps = job["steps"]
    run_step = next(s for s in steps if "run_scaling_n8.py" in s.get("run", ""))
    assert "--num-shares" in run_step["run"] and "--threshold" in run_step["run"]
    upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert upload["if"] == "always()"
    assert "docker_n8/data/**" in upload["with"]["path"]          # staged client data is not uploaded


def test_workflow_passes_the_static_workflow_check():
    from check_workflows_static import check
    report = check()["scaling-n8.yml"]
    assert report["ok"], report
