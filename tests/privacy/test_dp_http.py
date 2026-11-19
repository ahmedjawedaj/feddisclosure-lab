"""DP HTTP tests use a fake measurement and make NO epsilon guarantee."""
import asyncio
import httpx
import pytest
from feddisclosure.defenses.opendp_count import DPLibraryUnavailable
from services.client.app import ClientSettings, create_app
from services.gateway.app import GatewaySettings, create_app as make_gateway
import services.gateway.app as gateway_module


class FakeMeasurement:
    def release(self, count):
        return count + 0.25


def site(tmp_path):
    return create_app(ClientSettings("site-a", str(tmp_path / "site.db"), 40, 1, 5, "secret",
                                     dp_enabled=True, dp_epsilon=0.25, dp_total_budget=0.5),
                      _test_mechanism=FakeMeasurement())


def test_dp_mode_disables_raw_route_and_cannot_reset_budget(tmp_path):
    app = site(tmp_path)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as c:
            headers = {"X-Internal-Token": "secret"}
            raw = {"experiment_id": "a", "query_id": "all_records", "policy": "naive"}
            assert (await c.post("/v1/aggregate", json=raw, headers=headers)).status_code == 403
            payload = {"experiment_id": "a", "query_id": "all_records"}
            assert (await c.post("/v1/dp-count", json=payload)).status_code == 401
            first = await c.post("/v1/dp-count", json=payload, headers=headers)
            assert first.status_code == 200 and first.json()["noisy_count"] == 40.25
            assert "count" not in first.json()
            payload["experiment_id"] = "different_experiment"
            assert (await c.post("/v1/dp-count", json=payload, headers=headers)).status_code == 200
            third = await c.post("/v1/dp-count", json=payload, headers=headers)
            assert third.status_code == 409 and "noisy_count" not in third.json()
    asyncio.run(check())
    # A privacy-enabled site cannot be restarted to silently expose raw counts.
    with pytest.raises(ValueError, match="downgrade"):
        create_app(ClientSettings("site-a", str(tmp_path / "site.db"), 40, 1, 5, "secret"))


def test_cannot_activate_dp_on_existing_raw_release_history(tmp_path):
    path = str(tmp_path / "history.sqlite")
    raw = create_app(ClientSettings("site-a", path, 40, 1, 5, "secret"))
    async def query():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=raw), base_url="http://x") as c:
            return await c.post("/v1/aggregate", json={"experiment_id": "p", "query_id": "all_records", "policy": "naive"}, headers={"X-Internal-Token": "secret"})
    assert asyncio.run(query()).status_code == 200
    with pytest.raises(ValueError, match="fresh database"):
        create_app(ClientSettings("site-a", path, 40, 1, 5, "secret", dp_enabled=True), _test_mechanism=FakeMeasurement())


def test_dp_gateway_aggregate_only_and_fail_closed(monkeypatch):
    real_client = httpx.AsyncClient
    def make_reply(request):
        host = request.url.host
        if host == "site-c":
            return httpx.Response(409, json={"detail": "budget exhausted", "noisy_count": 999})
        return httpx.Response(200, json={"institution": host, "query_id": "all_records",
                            "noisy_count": 40.25, "epsilon_this_release": .25, "delta": 0})
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(make_reply), **kwargs))
    gateway = make_gateway(GatewaySettings(("http://site-a", "http://site-b", "http://site-c"),
                                            "secret", "research", dp_enabled=True, dp_epsilon=.25))
    async def query():
        async with real_client(transport=httpx.ASGITransport(app=gateway), base_url="http://local") as c:
            data = {"experiment_id": "a", "query_id": "all_records"}
            assert (await c.post("/v1/federated-dp-count", json=data)).status_code == 401
            deny = await c.post("/v1/federated-dp-count", json=data, headers={"X-Research-Key": "research"})
            assert deny.status_code == 409 and "999" not in str(deny.json()) and "noisy_count" not in str(deny.json())
            assert (await c.post("/v1/federated-count", json=dict(data, policy="naive"),
                                 headers={"X-Research-Key": "research"})).status_code == 403
    asyncio.run(query())


def test_dp_gateway_returns_only_sum(monkeypatch):
    real_client = httpx.AsyncClient
    def mock(request):
        counts = {"site-a": 40.25, "site-b": 35.25, "site-c": 35.25}
        return httpx.Response(200, json={"institution": request.url.host, "query_id": "all_records",
                            "noisy_count": counts[request.url.host], "epsilon_this_release": .25, "delta": 0})
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(mock), **kwargs))
    gateway = make_gateway(GatewaySettings(("http://site-a", "http://site-b", "http://site-c"),
                                            "secret", "research", dp_enabled=True, dp_epsilon=.25))
    async def query():
        async with real_client(transport=httpx.ASGITransport(app=gateway), base_url="http://local") as c:
            response = await c.post("/v1/federated-dp-count", json={"experiment_id": "a", "query_id": "all_records"}, headers={"X-Research-Key": "research"})
            assert response.status_code == 200
            d = response.json()
            assert d["noisy_count"] == 110.75 and d["participating_sites"] == 3
            assert d["epsilon_per_site_this_release"] == 0.25
            assert "site-a" not in str(d)
    asyncio.run(query())


def test_missing_real_dp_package_fails_startup_not_open(tmp_path):
    try:
        import opendp
    except ImportError:
        with pytest.raises(DPLibraryUnavailable):
            create_app(ClientSettings("site-a", str(tmp_path / "protected.sqlite"), 40, 1, 5, "secret", dp_enabled=True))
