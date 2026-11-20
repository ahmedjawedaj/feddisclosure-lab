"""Each Docker instance loads only its own isolated synthetic SQLite volume."""
from dataclasses import dataclass
import math
import sqlite3
import hmac
import os
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from feddisclosure.analytics.policy import evaluate_at_site
from feddisclosure.common.models import AggregateRelease, AggregateRequest, DPCountRequest, DPSiteCountRelease
from feddisclosure.defenses.dp_release import BudgetExhausted, release_dp_at_site
from feddisclosure.defenses.opendp_count import OpenDPCount
from feddisclosure.data.synthetic import InstitutionSpec, initialize_database
from feddisclosure.analytics.enforcement import enforce_release_mode


@dataclass(frozen=True)
class ClientSettings:
    institution: str
    db_path: str
    rows: int
    positive: int
    min_cell: int
    internal_token: str
    dp_enabled: bool = False
    dp_epsilon: float = 0.25
    dp_total_budget: float = 1.0
    enforced_policy: str | None = None

    @classmethod
    def from_env(cls) -> "ClientSettings":
        return cls(
            institution=os.getenv("FDL_INSTITUTION", "institution-a"),
            db_path=os.getenv("FDL_DB_PATH", f"/tmp/fdl-{os.getenv('FDL_INSTITUTION', 'institution-a')}.sqlite"),
            rows=int(os.getenv("FDL_ROWS", "40")),
            positive=int(os.getenv("FDL_CASES", "1")),
            min_cell=int(os.getenv("FDL_MIN_CELL", "5")),
            internal_token=os.getenv("FDL_INTERNAL_TOKEN", "local-dev-internal-token"),
            dp_enabled=os.getenv("FDL_DP_ENABLED", "0") == "1",
            dp_epsilon=float(os.getenv("FDL_DP_EPSILON", "0.25")),
            dp_total_budget=float(os.getenv("FDL_DP_TOTAL_BUDGET", "1.0")),
            enforced_policy=os.getenv("FDL_RELEASE_POLICY") or None,
        )


def create_app(settings: ClientSettings, *, _test_mechanism=None) -> FastAPI:
    if not settings.internal_token:
        raise ValueError("internal token is required")
    initialize_database(settings.db_path, InstitutionSpec(settings.institution, settings.rows, settings.positive))
    enforce_release_mode(settings.db_path, settings.enforced_policy, dp_enabled=settings.dp_enabled)
    if settings.dp_enabled:
        if not (math.isfinite(settings.dp_epsilon) and math.isfinite(settings.dp_total_budget)
                and 0 < settings.dp_epsilon <= settings.dp_total_budget):
            raise ValueError("Invalid DP configuration")
        with sqlite3.connect(settings.db_path) as conn:
            previous = conn.execute("SELECT COUNT(*) FROM query_audit").fetchone()[0]
        if previous:
            raise ValueError("DP mode requires a fresh database without earlier unprotected queries")
        # A production DP-enabled service will NOT start without OpenDP.
        mechanism = _test_mechanism if _test_mechanism is not None else OpenDPCount(settings.dp_epsilon)
    else:
        if _test_mechanism is not None:
            raise ValueError("Test measurement cannot be injected unless DP mode enabled")
        # A restart must not silently downgrade a protected site to an
        # unprotected analytics API while keeping the old privacy ledger.
        with sqlite3.connect(settings.db_path) as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dp_budget'").fetchone()
            spent = conn.execute("SELECT COALESCE(SUM(spent_micro),0) FROM dp_budget").fetchone()[0] if exists else 0
        if spent:
            raise ValueError("Database already has DP releases; refusing unprotected-mode downgrade")
        mechanism = None
    api = FastAPI(title=f"FedDisclosure {settings.institution}", docs_url=None, redoc_url=None)

    @api.get("/health")
    def health() -> dict:
        return {"ok": True}

    @api.post("/v1/aggregate", response_model=AggregateRelease)
    def aggregate(payload: AggregateRequest, x_internal_token: str = Header(default="")) -> AggregateRelease:
        if not hmac.compare_digest(x_internal_token, settings.internal_token):
            raise HTTPException(401, "missing or invalid internal token")
        if settings.dp_enabled:
            raise HTTPException(403, "unprotected query route is disabled in DP-only mode")
        if settings.enforced_policy and payload.policy != settings.enforced_policy:
            raise HTTPException(403, "request policy differs from enforced site release policy")
        decision = evaluate_at_site(settings.db_path, payload, settings.min_cell)
        if not decision.approved:
            raise HTTPException(403, {"code": "release_blocked", "reason": decision.reason})
        return AggregateRelease(
            institution=settings.institution, query_id=payload.query_id,
            policy=payload.policy, count=decision.count,
        )

    @api.post("/v1/dp-count", response_model=DPSiteCountRelease)
    def dp_count(payload: DPCountRequest, x_internal_token: str = Header(default="")) -> DPSiteCountRelease:
        if not hmac.compare_digest(x_internal_token, settings.internal_token):
            raise HTTPException(401, "missing or invalid internal token")
        if not settings.dp_enabled:
            raise HTTPException(503, "OpenDP count mode is not enabled")
        try:
            release = release_dp_at_site(
                settings.db_path, query_id=payload.query_id,
                principal_scope="all_research_access", epsilon=settings.dp_epsilon,
                total_budget=settings.dp_total_budget, mechanism=mechanism,
            )
        except BudgetExhausted as exc:
            raise HTTPException(409, "privacy budget exhausted") from exc
        return DPSiteCountRelease(institution=settings.institution,
            query_id=payload.query_id, noisy_count=release.noisy_count,
            epsilon_this_release=release.epsilon_spent_this_release, delta=release.delta)

    return api


app = create_app(ClientSettings.from_env())
