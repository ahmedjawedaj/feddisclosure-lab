"""Each Docker instance loads only its own isolated synthetic SQLite volume."""
from dataclasses import dataclass
import hmac
import os
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from feddisclosure.analytics.policy import evaluate_at_site
from feddisclosure.common.models import AggregateRelease, AggregateRequest
from feddisclosure.data.synthetic import InstitutionSpec, initialize_database


@dataclass(frozen=True)
class ClientSettings:
    institution: str
    db_path: str
    rows: int
    positive: int
    min_cell: int
    internal_token: str

    @classmethod
    def from_env(cls) -> "ClientSettings":
        return cls(
            institution=os.getenv("FDL_INSTITUTION", "institution-a"),
            db_path=os.getenv("FDL_DB_PATH", f"/tmp/fdl-{os.getenv('FDL_INSTITUTION', 'institution-a')}.sqlite"),
            rows=int(os.getenv("FDL_ROWS", "40")),
            positive=int(os.getenv("FDL_CASES", "1")),
            min_cell=int(os.getenv("FDL_MIN_CELL", "5")),
            internal_token=os.getenv("FDL_INTERNAL_TOKEN", "local-dev-internal-token"),
        )


def create_app(settings: ClientSettings) -> FastAPI:
    if not settings.internal_token:
        raise ValueError("internal token is required")
    initialize_database(settings.db_path, InstitutionSpec(settings.institution, settings.rows, settings.positive))
    api = FastAPI(title=f"FedDisclosure {settings.institution}", docs_url=None, redoc_url=None)

    @api.get("/health")
    def health() -> dict:
        return {"ok": True}

    @api.post("/v1/aggregate", response_model=AggregateRelease)
    def aggregate(payload: AggregateRequest, x_internal_token: str = Header(default="")) -> AggregateRelease:
        if not hmac.compare_digest(x_internal_token, settings.internal_token):
            raise HTTPException(401, "missing or invalid internal token")
        decision = evaluate_at_site(settings.db_path, payload, settings.min_cell)
        if not decision.approved:
            raise HTTPException(403, {"code": "release_blocked", "reason": decision.reason})
        return AggregateRelease(
            institution=settings.institution, query_id=payload.query_id,
            policy=payload.policy, count=decision.count,
        )

    return api


app = create_app(ClientSettings.from_env())
