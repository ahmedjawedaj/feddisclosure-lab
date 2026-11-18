import asyncio
import httpx
from services.client.app import ClientSettings, create_app


def test_site_token_and_release_policy(tmp_path):
    app = create_app(ClientSettings("test-site", str(tmp_path / "site.sqlite"), 40, 1, 5, "secret"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = {"query_id": "all_records", "policy": "history", "experiment_id": "z"}
            assert (await client.post("/v1/aggregate", json=body)).status_code == 401
            assert (await client.post("/v1/aggregate", json=body, headers={"X-Internal-Token": "bad"})).status_code == 401
            first = await client.post("/v1/aggregate", json=body, headers={"X-Internal-Token": "secret"})
            assert first.status_code == 200
            assert first.json()["count"] == 40
            body["query_id"] = "negative_records"
            second = await client.post("/v1/aggregate", json=body, headers={"X-Internal-Token": "secret"})
            assert second.status_code == 403
            assert "count" not in second.json()
            body["query_id"] = "DROP TABLE"
            assert (await client.post("/v1/aggregate", json=body, headers={"X-Internal-Token": "secret"})).status_code == 422

    asyncio.run(run())
