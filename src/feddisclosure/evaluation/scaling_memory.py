"""Peak container memory for the 8-client SecAgg+ run (pre-registered row S3).

`docker stats --no-stream` is polled while the run is in progress. Every poll gives the memory in use
by each project container. The peak that matters for a runner with limited RAM is the largest SUM over
the containers at one poll, so that is the pass criterion. The largest single container is reported as
well. Docker reports cgroup usage without the page cache, and polling can miss a spike shorter than the
poll interval, so the number is a lower bound on the true peak. Say so wherever it is quoted.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable, Iterable

GB = 10 ** 9
LIMIT_GB = 6.0          # pre-registered, S3
PROJECT_PREFIX = "feddisclosure-flower-"
STATS_FORMAT = "{{.Name}}\t{{.MemUsage}}"

_UNITS = {"b": 1, "kb": 10 ** 3, "mb": 10 ** 6, "gb": 10 ** 9, "tb": 10 ** 12,
          "kib": 2 ** 10, "mib": 2 ** 20, "gib": 2 ** 30, "tib": 2 ** 40}
_SIZE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*$")


def expected_container_names(clients: int, prefix: str = PROJECT_PREFIX) -> set[str]:
    """The 2 + 2 * clients containers of the SecAgg+ compose project: a ClientApp executor and a SuperNode per
    client, plus the server executor and the SuperLink. Other containers (such as the Flower CLI one) are extra."""
    names = {f"{prefix}serverexec-1", f"{prefix}superlink-1"}
    for i in range(clients):
        names |= {f"{prefix}clientexec-{i}-1", f"{prefix}supernode-{i}-1"}
    return names


def parse_size(text: str) -> int:
    """'123.4MiB' -> bytes. Accepts the binary and decimal units that Docker prints."""
    m = _SIZE.match(text)
    if not m or m.group(2).lower() not in _UNITS:
        raise ValueError(f"cannot read a memory size from {text!r}")
    return int(round(float(m.group(1)) * _UNITS[m.group(2).lower()]))


def parse_stats(output: str, prefix: str = PROJECT_PREFIX) -> dict[str, int]:
    """Memory in use per container, from lines 'name<TAB>used / limit'. Other containers are ignored."""
    out: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(None, 1)             # a tab or any run of spaces separates name and usage
        if len(parts) != 2 or not parts[0].startswith(prefix):
            continue
        out[parts[0]] = parse_size(parts[1].split("/")[0])
    return out


class PeakTracker:
    def __init__(self) -> None:
        self.samples = 0
        self.peak_total = 0
        self.per_container: dict[str, int] = {}
        self.max_containers = 0
        self.names_seen: set[str] = set()
        self.timeline: list[list[float]] = []        # [seconds since start, total bytes]
        self._t0 = time.monotonic()

    def add(self, sample: dict[str, int]) -> None:
        if not sample:
            return
        self.samples += 1
        total = sum(sample.values())
        self.peak_total = max(self.peak_total, total)
        self.max_containers = max(self.max_containers, len(sample))
        self.names_seen |= set(sample)
        for name, used in sample.items():
            self.per_container[name] = max(self.per_container.get(name, 0), used)
        self.timeline.append([round(time.monotonic() - self._t0, 1), total])

    def report(self, *, limit_gb: float = LIMIT_GB, run_completed: bool, expected_containers: int | None = None,
               poll_seconds: float, expected_names: Iterable[str] | None = None) -> dict:
        """The S3 record. pass needs a finished run, every expected container seen and a peak below the limit.

        With `expected_names` the exact set of expected containers must have been seen, and extra containers
        (such as the CLI one) are allowed. A count alone cannot tell an extra container from a missing one, so it
        is used only when no names are given."""
        peak_single = max(self.per_container.values(), default=0)
        if expected_names is not None:
            names = set(expected_names)
            expected_containers = len(names)
            missing = sorted(names - self.names_seen)
            extra = sorted(self.names_seen - names)
            complete_view = not missing
        else:
            if expected_containers is None:
                raise ValueError("give expected_names or expected_containers")
            missing, extra = None, None
            complete_view = self.max_containers >= expected_containers
        within = self.samples > 0 and self.peak_total < limit_gb * GB
        outcome = ("run_incomplete" if not run_completed else "no_samples" if self.samples == 0
                   else "containers_missing" if not complete_view else "pass" if within else "fail_memory")
        return {"schema_version": 1, "samples": self.samples, "poll_interval_seconds": poll_seconds,
                "containers_expected": expected_containers, "containers_seen": self.max_containers,
                "expected_containers_missing": missing, "extra_containers_seen": extra,
                "peak_total_bytes": self.peak_total, "peak_total_gb": round(self.peak_total / GB, 3),
                "peak_single_container_gb": round(peak_single / GB, 3),
                "per_container_peak_gb": {k: round(v / GB, 3) for k, v in sorted(self.per_container.items())},
                "limit_gb": limit_gb, "gb_means": "10^9 bytes", "run_completed": run_completed,
                "outcome": outcome, "pass": outcome == "pass",
                "memory_measure": "docker stats MemUsage per container, summed at each poll, maximum over polls",
                "caveat": "Docker reports usage without page cache. A spike shorter than the poll interval can be "
                          "missed, so the peak is a lower bound.",
                "timeline_seconds_total_bytes": self.timeline[:: max(1, len(self.timeline) // 200)]}


class StatsSampler(threading.Thread):
    """Polls memory in a background thread. `read` returns the raw output of docker stats."""

    def __init__(self, read: Callable[[], str], tracker: PeakTracker, interval: float = 2.0):
        super().__init__(daemon=True)
        self._read, self.tracker, self.interval = read, tracker, interval
        self._stop_event = threading.Event()
        self.errors: list[str] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tracker.add(parse_stats(self._read()))
            except Exception as exc:                  # a failed poll is recorded, it never stops the run
                self.errors.append(f"{type(exc).__name__}: {exc}"[:200])
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=30)
