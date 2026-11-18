"""Gateway response shape and fail-closed semantics are security-critical."""
import asyncio
import httpx
from services.gateway.app import GatewaySettings, create_app
import services.gateway.app as gateway_module


def test_gateway_hides_partial_results_and_requires_key(monkeypatch):
    real_async_client = httpx.AsyncClient

    def fake_site(request):
        if request.url.host == "site-c":
            return httpx.Response(403, json={"detail": "small cell"})
        return httpx.Response(200, json={
            "institution": request.url.host,
            "query_id": "all_records", "policy": "naive", "count": 40,
        })

    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda **kwargs: real_async_client(
        transport=httpx.MockTransport(fake_site), **kwargs,
    ))
    app = create_app(GatewaySettings(("http://site-a", "http://site-b", "http://site-c"), "secret", "research"))

    async def test_call():
        async with real_async_client(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            req = {"experiment_id": "abc", "query_id": "all_records", "policy": "naive"}
            assert (await client.post("/v1/federated-count", json=req)).status_code == 401
            res = await client.post("/v1/federated-count", json=req, headers={"X-Research-Key": "research"})
            assert res.status_code == 409
            assert "count" not in res.json()
            assert "40" not in str(res.json())
    asyncio.run(test_call())


def test_gateway_sums_three_sites_without_per_site_breakdown(monkeypatch):
    real_async_client = httpx.AsyncClient
    sample_counts = {"site-a": 40, "site-b": 35, "site-c": 35}

    def fake_site(request):
        return httpx.Response(200, json={
            "institution": request.url.host, "query_id": "all_records",
            "policy": "naive", "count": sample_counts[request.url.host],
        })

    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda **kwargs: real_async_client(
        transport=httpx.MockTransport(fake_site), **kwargs,
    ))
    app = create_app(GatewaySettings(tuple(f"http://{name}" for name in sample_counts), "secret", "research"))

    async def test_call():
        async with real_async_client(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            res = await client.post("/v1/federated-count", json={
                "experiment_id": "abc", "query_id": "all_records", "policy": "naive",
            }, headers={"X-Research-Key": "research"})
            assert res.status_code == 200
            assert res.json()["count"] == 110
            assert res.json()["participating_sites"] == 3
            assert "institutions" not in res.json()
            assert "site-a" not in str(res.json())
    asyncio.run(test_call())


def test_gateway_fails_closed_if_site_unreachable(monkeypatch):
    real_async_client = httpx.AsyncClient

    def fake_site(request):
        if request.url.host == "site-c":
            raise httpx.ConnectError("connection failed")
        return httpx.Response(200, json={
            "institution": request.url.host, "query_id": "all_records", "policy": "naive", "count": 40,
        })
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda **kwargs: real_async_client(
        transport=httpx.MockTransport(fake_site), **kwargs,
    ))
    app = create_app(GatewaySettings(("http://site-a", "http://site-b", "http://site-c"), "secret", "research"))

    async def run():
        async with real_async_client(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            res = await client.post("/v1/federated-count", json={
                "experiment_id": "abc", "query_id": "all_records", "policy": "naive",
            }, headers={"X-Research-Key": "research"})
            assert res.status_code == 503
            assert "count" not in res.json()
            assert "40" not in str(res.json())
    asyncio.run(run())
