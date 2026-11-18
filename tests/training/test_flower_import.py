"""Native Flower import gate: runs in ML CI, skipped when dependency unavailable."""
import pytest
pytest.importorskip("torch")
flwr = pytest.importorskip("flwr")
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from feddisclosure.training.flower_client import app as client_app
from feddisclosure.training.flower_server import app as server_app


def test_native_flower_apps_exist():
    assert isinstance(client_app, ClientApp)
    assert isinstance(server_app, ServerApp)
