from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_history_overlay_enforces_policy_all_sites_and_gateway():
    overlay = yaml.safe_load((ROOT/'compose.history.yaml').read_text())
    services = overlay['services']
    assert set(services) == {'institution-a','institution-b','institution-c','gateway'}
    assert all(v['environment']['FDL_RELEASE_POLICY'] == 'principal_history'
               for v in services.values())


def test_native_verifier_is_explicitly_gated():
    source = (ROOT/'scripts/verify_compose_history.sh').read_text()
    assert 'command -v docker' in source
    assert 'exit 2' in source
    assert 'up --build -d --wait' in source
    assert 'down --volumes --remove-orphans' in source
