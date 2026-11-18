"""Research gateway. Only aggregated count leaves this service to the user."""
from dataclasses import dataclass
import asyncio
import hmac
import os
import httpx
from fastapi import FastAPI, Header, HTTPException
from feddisclosure.common.models import AggregateRequest, FederatedRelease, AggregateRelease


@dataclass(frozen=True)
class GatewaySettings:
    urls: tuple[str, ...]
    internal_token: str
    research_key: str

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            urls=tuple(x.strip() for x in os.getenv("FDL_CLIENT_URLS", "http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103").split(",") if x.strip()),
            internal_token=os.getenv("FDL_INTERNAL_TOKEN", "local-dev-internal-token"),
            research_key=os.getenv("FDL_RESEARCH_KEY", "local-dev-research-key"),
        )


def create_app(settings: GatewaySettings) -> FastAPI:
    if len(settings.urls) < 2 or not settings.internal_token or not settings.research_key:
        raise ValueError("Research gateway requires >=2 sites and both keys")
    api = FastAPI(title="FedDisclosure Research Gateway", docs_url=None, redoc_url=None)

    @api.get("/health")
    def health() -> dict:
        return {"ok": True}

    @api.post("/v1/federated-count", response_model=FederatedRelease)
    async def count(payload: AggregateRequest, x_research_key: str = Header(default="")) -> FederatedRelease:
        if not hmac.compare_digest(x_research_key, settings.research_key):
            raise HTTPException(401, "missing or invalid research key")
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

    return api


app = create_app(GatewaySettings.from_env())
