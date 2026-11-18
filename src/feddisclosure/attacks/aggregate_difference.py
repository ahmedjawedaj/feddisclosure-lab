"""An analytics-user attack. Input is ONLY already published aggregate results."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ObservedCount:
    query_id: str
    count: int


def infer_restricted_count(observations: list[ObservedCount]) -> int | None:
    """Infer a hidden subgroup from all minus the publicly released complement."""
    available = {item.query_id: item.count for item in observations}
    if "all_records" not in available or "negative_records" not in available:
        return None
    inferred = available["all_records"] - available["negative_records"]
    if inferred < 0:
        return None
    return inferred
