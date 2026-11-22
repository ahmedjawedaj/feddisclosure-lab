"""Peak memory bookkeeping for the 8-client Docker run (v0.15, row S3). No Docker is used here."""
import time

import pytest

from feddisclosure.evaluation import scaling_memory as sm

GB = sm.GB


@pytest.mark.parametrize("text,expected", [("512B", 512), ("1KiB", 1024), ("1.5MiB", 1572864), ("2GiB", 2 * 2 ** 30),
                                           ("3MB", 3_000_000), ("0.5GB", 500_000_000), (" 10 MiB ", 10 * 2 ** 20)])
def test_parse_size_reads_binary_and_decimal_units(text, expected):
    assert sm.parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "12", "MiB", "1.2XiB", "-3MiB"])
def test_parse_size_refuses_unreadable_text(bad):
    with pytest.raises(ValueError):
        sm.parse_size(bad)


def test_parse_stats_keeps_only_project_containers_and_uses_the_usage_column():
    out = ("feddisclosure-flower-superlink-1\t120.5MiB / 15.6GiB\n"
           "feddisclosure-flower-clientexec-0-1\t1.2GiB / 15.6GiB\n"
           "some-other-container\t9GiB / 15.6GiB\n\n")
    parsed = sm.parse_stats(out)
    assert set(parsed) == {"feddisclosure-flower-superlink-1", "feddisclosure-flower-clientexec-0-1"}
    assert parsed["feddisclosure-flower-clientexec-0-1"] == round(1.2 * 2 ** 30)


def test_parse_stats_also_accepts_space_separated_columns():
    assert sm.parse_stats("feddisclosure-flower-serverexec-1   2GiB / 7GiB") == {"feddisclosure-flower-serverexec-1": 2 * 2 ** 30}
    with pytest.raises(ValueError):
        sm.parse_stats("feddisclosure-flower-x-1\tgarbage / 7GiB")


def test_peak_is_the_largest_sum_at_one_poll_and_not_the_sum_of_individual_peaks():
    t = sm.PeakTracker()
    t.add({"a": 3 * GB, "b": 1 * GB})        # total 4 GB
    t.add({"a": 1 * GB, "b": 3 * GB})        # total 4 GB, b peaks later than a
    assert t.peak_total == 4 * GB
    assert t.per_container == {"a": 3 * GB, "b": 3 * GB}      # summing these would wrongly give 6 GB
    t.add({})                                 # an empty poll is not a sample
    assert t.samples == 2


def _tracker(total_gb, containers=18):
    t = sm.PeakTracker()
    t.add({f"c{i}": int(total_gb * GB / containers) for i in range(containers)})
    return t


def test_pass_needs_a_peak_strictly_below_the_limit():
    assert _tracker(5.9).report(run_completed=True, expected_containers=18, poll_seconds=2)["pass"] is True
    exact = sm.PeakTracker()
    exact.add({"a": 6 * GB})
    r = exact.report(run_completed=True, expected_containers=1, poll_seconds=2)
    assert r["pass"] is False and r["outcome"] == "fail_memory"
    assert _tracker(7.5).report(run_completed=True, expected_containers=18, poll_seconds=2)["outcome"] == "fail_memory"


def test_low_memory_alone_is_not_a_pass():
    low = _tracker(1.0)
    assert low.report(run_completed=False, expected_containers=18, poll_seconds=2)["outcome"] == "run_incomplete"
    assert low.report(run_completed=True, expected_containers=20, poll_seconds=2)["outcome"] == "containers_missing"
    empty = sm.PeakTracker()
    r = empty.report(run_completed=True, expected_containers=18, poll_seconds=2)
    assert r["outcome"] == "no_samples" and r["pass"] is False


def test_report_gives_gigabytes_as_decimal_and_the_limit():
    r = _tracker(4.5).report(run_completed=True, expected_containers=18, poll_seconds=2)
    assert r["peak_total_gb"] == pytest.approx(4.5, abs=0.001) and r["limit_gb"] == 6.0
    assert r["gb_means"] == "10^9 bytes" and r["containers_seen"] == 18
    assert sm.LIMIT_GB == 6.0                                  # pre-registered value, never changed


def test_sampler_collects_polls_and_records_a_failed_poll_without_stopping():
    calls = {"n": 0}

    def read():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("docker busy")
        return "feddisclosure-flower-superlink-1\t100MiB / 8GiB\n"
    tracker = sm.PeakTracker()
    s = sm.StatsSampler(read, tracker, interval=0.01)
    s.start()
    deadline = time.time() + 5
    while tracker.samples < 3 and time.time() < deadline:
        time.sleep(0.01)
    s.stop()
    assert tracker.samples >= 3 and any("docker busy" in e for e in s.errors)
    assert not s.is_alive()
