"""Authorized scorer: never pass evaluator-only truth to an attack module."""
from dataclasses import dataclass, asdict
from feddisclosure.attacks.aggregate_difference import ObservedCount, infer_restricted_count


@dataclass(frozen=True)
class Evaluation:
    inferred_positive_count: int | None
    restricted_ground_truth: int
    absolute_error: int | None
    exact_inference: bool
    successful_small_group_disclosure: bool
    rejected_queries: int
    legitimate_query_rejection_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(observed: list[ObservedCount], ground_truth: int, attempted: int, rejected: int, min_cell: int) -> Evaluation:
    if attempted <= 0 or rejected > attempted:
        raise ValueError("invalid attempted/rejected counts")
    # Attack sees only `observed`. Ground truth is consumed below, in evaluator.
    guess = infer_restricted_count(observed)
    accurate = guess is not None and guess == ground_truth
    return Evaluation(
        inferred_positive_count=guess,
        restricted_ground_truth=ground_truth,
        absolute_error=abs(guess - ground_truth) if guess is not None else None,
        exact_inference=accurate,
        successful_small_group_disclosure=accurate and ground_truth < min_cell,
        rejected_queries=rejected,
        legitimate_query_rejection_rate=rejected / attempted,
    )
