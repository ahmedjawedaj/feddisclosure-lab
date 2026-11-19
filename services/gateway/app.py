"""Research gateway. Only aggregated count leaves this service to the user."""
from dataclasses import dataclass
import math
import asyncio
import hmac
import os
import httpx
from fastapi import FastAPI, Header, HTTPException
from feddisclosure.common.models import (AggregateRequest, FederatedRelease, AggregateRelease,
                                         DPCountRequest, DPSiteCountRelease, DPFederatedCountRelease)


@dataclass(frozen=True)
class GatewaySettings:
    urls: tuple[str, ...]
    internal_token: str
    research_key: str
    dp_enabled: bool = False
    dp_epsilon: float = 0.25

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            urls=tuple(x.strip() for x in os.getenv("FDL_CLIENT_URLS", "http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103").split(",") if x.strip()),
            internal_token=os.getenv("FDL_INTERNAL_TOKEN", "local-dev-internal-token"),
            research_key=os.getenv("FDL_RESEARCH_KEY", "local-dev-research-key"),
            dp_enabled=os.getenv("FDL_DP_ENABLED", "0") == "1",
            dp_epsilon=float(os.getenv("FDL_DP_EPSILON", "0.25")),
        )


def create_app(settings: GatewaySettings) -> FastAPI:
    if len(settings.urls) < 2 or not settings.internal_token or not settings.research_key:
        raise ValueError("Research gateway requires >=2 sites and both keys")
    if settings.dp_enabled and (not math.isfinite(settings.dp_epsilon) or settings.dp_epsilon <= 0):
        raise ValueError("Invalid DP epsilon")
    api = FastAPI(title="FedDisclosure Research Gateway", docs_url=None, redoc_url=None)

    @api.get("/health")
    def health() -> dict:
        return {"ok": True}

    @api.post("/v1/federated-count", response_model=FederatedRelease)
    async def count(payload: AggregateRequest, x_research_key: str = Header(default="")) -> FederatedRelease:
        if not hmac.compare_digest(x_research_key, settings.research_key):
            raise HTTPException(401, "missing or invalid research key")
        if settings.dp_enabled:
            raise HTTPException(403, "unprotected query route is disabled in DP-only mode")
        async with httpx.AsyncClient(timeout=10.0) as http:
            async def request_site(url: str) -> httpx.Response:
                return await http.post(
                    f"{url}/v1/aggregate", json=payload.model_dump(),
                    headers={"X-Internal-Token": settings.internal_token},
                )
            try:
                responses = await asyncio.gather(*(request_site(url) for url in settings.urls))
            except httpx.RequestError as exc:
                raise HTTPException(503, "institution unreachable; no aggregate released") from exc
        if any(resp.status_code != 200 for resp in responses):
            # All-or-nothing: do not leak partial-site aggregates in errors.
            raise HTTPException(409, "federated query rejected; no aggregate released")
        try:
            releases = [AggregateRelease.model_validate(resp.json()) for resp in responses]
        except Exception as exc:
            raise HTTPException(502, "malformed institution response") from exc
        if any(r.query_id != payload.query_id or r.policy != payload.policy for r in releases):
            raise HTTPException(502, "institution response mismatch")
        if len({r.institution for r in releases}) != len(settings.urls):
            raise HTTPException(502, "duplicate institution identity")
        return FederatedRelease(
            experiment_id=payload.experiment_id, query_id=payload.query_id,
            policy=payload.policy, count=sum(r.count for r in releases),
            participating_sites=len(releases),
        )

    @api.post("/v1/federated-dp-count", response_model=DPFederatedCountRelease)
    async def federated_dp_count(payload: DPCountRequest, x_research_key: str = Header(default="")) -> DPFederatedCountRelease:
        if not hmac.compare_digest(x_research_key, settings.research_key):
            raise HTTPException(401, "missing or invalid research key")
        if not settings.dp_enabled:
            raise HTTPException(503, "DP mode not enabled")
        async with httpx.AsyncClient(timeout=10.0) as http:
            async def request_site(url: str) -> httpx.Response:
                return await http.post(f"{url}/v1/dp-count", json=payload.model_dump(),
                    headers={"X-Internal-Token": settings.internal_token})
            try:
                replies = await asyncio.gather(*(request_site(url) for url in settings.urls))
            except httpx.RequestError as exc:
                raise HTTPException(503, "institution unreachable; no federated release") from exc
        if any(response.status_code != 200 for response in replies):
            # Some successful sites can have spent budget. No partial counts exposed.
            raise HTTPException(409, "federated DP release not available")
        try:
            releases = [DPSiteCountRelease.model_validate(reply.json()) for reply in replies]
        except Exception as exc:
            raise HTTPException(502, "invalid DP institution release") from exc
        if len({r.institution for r in releases}) != len(settings.urls):
            raise HTTPException(502, "duplicate institution")
        if any(r.query_id != payload.query_id or abs(r.epsilon_this_release - settings.dp_epsilon) > 1e-12
               or r.delta != 0 for r in releases):
            raise HTTPException(502, "DP release metadata mismatch")
        return DPFederatedCountRelease(experiment_id=payload.experiment_id,
            query_id=payload.query_id, noisy_count=sum(r.noisy_count for r in releases),
            participating_sites=len(releases), epsilon_per_site_this_release=settings.dp_epsilon,
            delta_per_site=0.0)

    return api


app = create_app(GatewaySettings.from_env())
