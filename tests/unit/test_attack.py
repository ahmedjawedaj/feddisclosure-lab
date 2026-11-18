from feddisclosure.attacks.aggregate_difference import ObservedCount, infer_restricted_count
from feddisclosure.evaluation.analytics import evaluate


def test_infer_only_from_published_counts():
    observed = [ObservedCount("all_records", 110), ObservedCount("negative_records", 107)]
    assert infer_restricted_count(observed) == 3
    metrics = evaluate(observed, ground_truth=3, attempted=2, rejected=0, min_cell=5)
    assert metrics.successful_small_group_disclosure
    assert metrics.legitimate_query_rejection_rate == 0


def test_missing_query_produces_no_guess():
    observed = [ObservedCount("all_records", 110)]
    assert infer_restricted_count(observed) is None
    metrics = evaluate(observed, ground_truth=3, attempted=2, rejected=1, min_cell=5)
    assert not metrics.successful_small_group_disclosure
    assert metrics.legitimate_query_rejection_rate == 0.5


def test_truth_does_not_change_attack_guess():
    observed = [ObservedCount("all_records", 110), ObservedCount("negative_records", 107)]
    assert evaluate(observed, 3, 2, 0, 5).inferred_positive_count == 3
    assert evaluate(observed, 4, 2, 0, 5).inferred_positive_count == 3
