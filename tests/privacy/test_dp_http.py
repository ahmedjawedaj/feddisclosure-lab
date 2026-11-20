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


def test_dp_http_request_id_replay_and_mismatch(tmp_path):
    import sqlite3
    app = site(tmp_path)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://site") as c:
            headers = {"X-Internal-Token": "secret"}
            payload = {"experiment_id": "any", "query_id": "all_records",
                       "request_id": "reliable_request_123"}
            first = await c.post("/v1/dp-count", json=payload, headers=headers)
            second = await c.post("/v1/dp-count", json=payload, headers=headers)
            assert first.status_code == second.status_code == 200
            assert first.json() == second.json()
            conflict = await c.post("/v1/dp-count", json={**payload, "query_id": "negative_records"}, headers=headers)
            assert conflict.status_code == 409
            assert "noisy_count" not in conflict.text
    asyncio.run(check())
    with sqlite3.connect(tmp_path / "site.db") as conn:
        assert conn.execute("SELECT SUM(spent_micro) FROM dp_budget").fetchone()[0] == 250000
        assert conn.execute("SELECT COUNT(*) FROM dp_release_cache").fetchone()[0] == 1


def test_gateway_retry_after_partial_site_failure_does_not_double_spend(tmp_path, monkeypatch):
    """Network failure consumes partial budget, but a shared retry ID reuses it.

    This is a FAKE measurement. It tests transactional HTTP orchestration,
    not a native OpenDP privacy guarantee.
    """
    import sqlite3
    real_client = httpx.AsyncClient
    sites = {
        name: create_app(ClientSettings(name, str(tmp_path / f"{name}.db"), 40, 1, 5,
                                        "secret", dp_enabled=True, dp_epsilon=.25,
                                        dp_total_budget=.5), _test_mechanism=FakeMeasurement())
        for name in ("site-a", "site-b", "site-c")
    }
    calls = {name: 0 for name in sites}
    class RoutedTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            site_name = request.url.host
            calls[site_name] += 1
            if site_name == "site-c" and calls[site_name] == 1:
                return httpx.Response(503, json={"detail": "temporary site outage"})
            inner = httpx.ASGITransport(app=sites[site_name])
            return await inner.handle_async_request(request)
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda **kwargs:
                        real_client(transport=RoutedTransport(), **kwargs))
    gateway = make_gateway(GatewaySettings(tuple(f"http://{n}" for n in sites),
                                           "secret", "research", dp_enabled=True, dp_epsilon=.25))
    async def exercise():
        async with real_client(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway") as c:
            req = {"experiment_id": "science", "query_id": "all_records",
                   "request_id": "stable_retry_identifier_123"}
            h = {"X-Research-Key": "research"}
            first = await c.post("/v1/federated-dp-count", json=req, headers=h)
            assert first.status_code == 409 and "noisy_count" not in first.text
            again = await c.post("/v1/federated-dp-count", json=req, headers=h)
            assert again.status_code == 200
            identical = await c.post("/v1/federated-dp-count", json=req, headers=h)
            assert identical.json() == again.json()
            mismatch = await c.post("/v1/federated-dp-count",
                                    json={**req, "query_id": "negative_records"}, headers=h)
            assert mismatch.status_code == 409
    asyncio.run(exercise())
    for name in sites:
        with sqlite3.connect(tmp_path / f"{name}.db") as conn:
            assert conn.execute("SELECT SUM(spent_micro) FROM dp_budget").fetchone()[0] == 250000
            assert conn.execute("SELECT COUNT(*) FROM dp_release_cache").fetchone()[0] == 1


def test_gateway_rejects_mismatched_site_request_id(monkeypatch):
    real_client = httpx.AsyncClient
    def respond(request):
        request_id = "stale_other_request_123" if request.url.host == "site-b" else "expected_request_123"
        return httpx.Response(200, json={"institution": request.url.host,
             "query_id": "all_records", "request_id": request_id,
             "noisy_count": 7.25, "epsilon_this_release": .25, "delta": 0})
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda **kwargs:
                        real_client(transport=httpx.MockTransport(respond), **kwargs))
    gateway = make_gateway(GatewaySettings(("http://site-a", "http://site-b", "http://site-c"),
                                           "secret", "research", dp_enabled=True, dp_epsilon=.25))
    async def query():
        async with real_client(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway") as c:
            result = await c.post("/v1/federated-dp-count", json={
                "experiment_id": "test", "query_id": "all_records",
                "request_id": "expected_request_123"}, headers={"X-Research-Key": "research"})
            assert result.status_code == 502
            assert "noisy_count" not in result.text
    asyncio.run(query())
