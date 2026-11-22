"""Static/native-preflight tests are NOT cryptographic proof of SecAgg+."""
import ast
from importlib.util import find_spec
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_secagg_project_uses_official_modules_and_threshold():
    cfg = tomllib.loads((ROOT / "experiments/flower_secagg/pyproject.toml").read_text())
    fl = cfg["tool"]["flwr"]
    params = fl["app"]["config"]
    assert params["clients"] == 3
    assert params["num-shares"] == params["clients"]
    assert 2 <= params["reconstruction-threshold"] <= params["num-shares"]
    # The connection is written to the Flower config by the runner, never kept as a legacy table.
    assert "federations" not in fl
    assert "ensure_simulation_connection()" in (ROOT / "scripts/run_secagg_native.py").read_text()
    client = (ROOT / "src/feddisclosure/training/secagg_client.py").read_text()
    server = (ROOT / "src/feddisclosure/training/secagg_server.py").read_text()
    ast.parse(client)
    ast.parse(server)
    assert "from flwr.client.mod import secaggplus_mod" in client
    assert "from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow" in server
    # The mod is wrapped only to allow a matched plaintext control (use-secagg=false).
    # The wrapper must delegate to the upstream mod and default to secure aggregation on.
    assert "mods=[conditional_secaggplus_mod]" in client
    assert "return secaggplus_mod(msg, ctxt, call_next)" in client
    assert 'ctxt.run_config.get("use-secagg", True)' in client
    assert params["use-secagg"] is True
    assert "SecAggPlusWorkflow(" in server
    assert "fraction_evaluate=0.0" in server
    assert "num_shares=shares, reconstruction_threshold=threshold" in server


def test_native_runner_rejects_missing_runtime(tmp_path):
    from scripts.run_secagg_native import execute
    if find_spec("flwr") is None:
        with pytest.raises(RuntimeError, match="Native Flower"):
            execute(data_root=tmp_path, output=tmp_path / "no-evidence")
        assert not (tmp_path / "no-evidence").exists()


def test_native_ci_flows_generate_inputs_in_clean_checkout():
    for workflow, required in (
        ("flower-smoke.yml", ["prepare_fl_data.py", "run_flower_experiment.py", "verify_flower_parity.py"]),
        ("research-smoke.yml", ["prepare_fl_data.py", "run_torch_reference.py", "run_gradient_inversion.py"]),
        ("secagg-native.yml", ["prepare_fl_data.py", "run_secagg_native.py"]),
    ):
        source = (ROOT / ".github/workflows" / workflow).read_text()
        steps = source[source.index("    steps:"):]
        positions = [steps.index(s) for s in required]
        assert positions == sorted(positions), f"Inputs must be prepared before jobs in {workflow}"
