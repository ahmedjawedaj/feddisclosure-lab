"""Repeatable three-site HTTP retry-cost benchmark with an injected FAKE mechanism.

The test mechanism is deterministic and NOT differentially private. This
benchmark validates retry/accounting behavior only, never OpenDP guarantees.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import time

import httpx
import services.gateway.app as gateway_module
from services.client.app import ClientSettings, create_app
from services.gateway.app import GatewaySettings, create_app as gateway_app


class TestOnlyMeasurement:
    def release(self, count: int) -> float:
        return float(count) + .25


async def scenario(folder: Path, *, idempotent: bool) -> dict:
    names = ("institution-a", "institution-b", "institution-c")
    apps = {name: create_app(
        ClientSettings(name, str(folder / f"{name}.db"), rows=40, positive=1,
                       min_cell=5, internal_token="test-local-secret", dp_enabled=True,
                       dp_epsilon=.25, dp_total_budget=.5),
        _test_mechanism=TestOnlyMeasurement(),
    ) for name in names}
    calls = {n: 0 for n in names}
    class Network(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            institution = request.url.host
            calls[institution] += 1
            if institution == "institution-c" and calls[institution] == 1:
                return httpx.Response(503, json={"detail": "simulated unavailable institution"})
            return await httpx.ASGITransport(app=apps[institution]).handle_async_request(request)
    original = httpx.AsyncClient
    gateway_module.httpx.AsyncClient = lambda **kwargs: original(transport=Network(), **kwargs)
    try:
        app = gateway_app(GatewaySettings(tuple(f"http://{name}" for name in names),
                                          "test-local-secret", "test-research-key",
                                          dp_enabled=True, dp_epsilon=.25))
        payload = {"experiment_id": "controlled_retry", "query_id": "all_records"}
        if idempotent:
            payload["request_id"] = "retry_same_randomized_answer_001"
        statuses, counts = [], []
        async with original(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            for _ in range(3):
                response = await client.post("/v1/federated-dp-count", json=payload,
                                             headers={"X-Research-Key": "test-research-key"})
                statuses.append(response.status_code)
                counts.append(response.json().get("noisy_count"))
    finally:
        gateway_module.httpx.AsyncClient = original
    budgets = {}
    for name in names:
        with sqlite3.connect(folder / f"{name}.db") as conn:
            budget = conn.execute("SELECT SUM(spent_micro) FROM dp_budget").fetchone()[0]
            cache = conn.execute("SELECT COUNT(*) FROM dp_release_cache").fetchone()[0]
            budgets[name] = {"spent_epsilon": budget / 1_000_000, "cached_releases": cache}
    assert statuses == ([409, 200, 200] if idempotent else [409, 200, 409])
    assert counts[0] is None
    if idempotent:
        assert counts[1] == counts[2]
        assert {n: b["spent_epsilon"] for n, b in budgets.items()} == dict.fromkeys(names, .25)
    else:
        assert budgets["institution-a"]["spent_epsilon"] == .5
        assert budgets["institution-b"]["spent_epsilon"] == .5
    return {"request_id_present": idempotent, "http_statuses": statuses,
            "gateway_noisy_values": counts, "site_budgets": budgets}


def run(output: Path) -> dict:
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fdl-idempotency-") as tmp:
        root = Path(tmp)
        (root / "legacy").mkdir()
        (root / "retry_safe").mkdir()
        legacy = asyncio.run(scenario(root / "legacy", idempotent=False))
        retry_safe = asyncio.run(scenario(root / "retry_safe", idempotent=True))
    result = {"schema_version": 1,
              "time_utc": datetime.now(timezone.utc).isoformat(),
              "engine": "three_site_http_ASGI_fake_measurement_NOT_DP",
              "legacy": legacy, "retry_safe": retry_safe,
              "elapsed_seconds": round(time.perf_counter() - start, 4),
              "limitations": [
                  "The deterministic test measurement offers NO DP guarantee",
                  "No native OpenDP, Docker, or SecAgg+ execution is involved",
                  "Only deterministic one-site outage and three repeat requests are tested",
                  "A retry ID binds one previous noisy answer, not a new composition budget",
              ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("reports/results/sprint4/retry_benchmark.json"))
    a = p.parse_args()
    out = run(a.output)
    print(json.dumps({"legacy": out["legacy"], "retry_safe": out["retry_safe"]}, indent=2))
