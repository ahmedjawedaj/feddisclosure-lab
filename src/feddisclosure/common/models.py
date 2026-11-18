"""Minimal, typed research-only protocol. There is no raw SQL endpoint."""
from typing import Literal
from pydantic import BaseModel, Field

QueryId = Literal["all_records", "negative_records"]
Policy = Literal["naive", "history"]


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
