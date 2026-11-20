"""Minimal, typed research-only protocol. There is no raw SQL endpoint."""
from typing import Literal
from pydantic import BaseModel, Field

QueryId = Literal["all_records", "negative_records"]
Policy = Literal["naive", "history", "principal_history"]


class AggregateRequest(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    query_id: QueryId
    policy: Policy


class AggregateRelease(BaseModel):
    institution: str
    query_id: QueryId
    policy: Policy
    count: int = Field(ge=0)


class FederatedRelease(BaseModel):
    experiment_id: str
    query_id: QueryId
    policy: Policy
    count: int = Field(ge=0)
    participating_sites: int = Field(ge=1)


class DPCountRequest(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    query_id: QueryId
    # The explicit request ID is a retry key, NOT a fresh privacy budget.
    # It is optional for backward compatibility. A new request ID costs epsilon.
    request_id: str | None = Field(default=None, min_length=12, max_length=100,
                                   pattern=r"^[a-zA-Z0-9_-]+$")


class DPSiteCountRelease(BaseModel):
    institution: str
    query_id: QueryId
    request_id: str | None = None
    noisy_count: float = Field(allow_inf_nan=False)
    epsilon_this_release: float = Field(gt=0)
    delta: float = Field(ge=0)


class DPFederatedCountRelease(BaseModel):
    experiment_id: str
    query_id: QueryId
    request_id: str | None = None
    noisy_count: float = Field(allow_inf_nan=False)
    participating_sites: int = Field(ge=2)
    epsilon_per_site_this_release: float = Field(gt=0)
    delta_per_site: float = Field(ge=0)
