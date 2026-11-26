"""ServerApp for the P2 probe (v0.16.13, row S4).

It is the recording ServerApp of v0.15 (`secagg_server_capture`) with one more observation point. The
application strategy class is swapped for a subclass that writes to the capture directory, as
`fit_results_seen.json`, the metrics, the sample count and the node identifier of every result that
`aggregate_fit` receives. It changes nothing that the strategy returns.
"""
from __future__ import annotations

import json
from pathlib import Path

from flwr.common import Context
from flwr.server import Grid, ServerApp

from feddisclosure.training import secagg_server as base
from feddisclosure.training import secagg_server_capture as capture

app = ServerApp()
SEEN_NAME = "fit_results_seen.json"


def _plain(value):
    return value if isinstance(value, (bool, int, float, str)) or value is None else repr(value)


def make_metrics_strategy(directory: Path, parent):
    class MetricsSeenFedAvg(parent):
        def aggregate_fit(self, server_round, results, failures):
            seen = [{"node_id": str(getattr(proxy, "node_id", getattr(proxy, "cid", ""))),
                     "cid": str(getattr(proxy, "cid", "")), "num_examples": int(fit_res.num_examples),
                     "metrics": {k: _plain(v) for k, v in dict(fit_res.metrics).items()}}
                    for proxy, fit_res in results]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / SEEN_NAME).write_text(json.dumps(
                {"server_round": int(server_round), "results": seen, "failures": len(failures)},
                indent=2, sort_keys=True) + "\n")
            return super().aggregate_fit(server_round, results, failures)

    return MetricsSeenFedAvg


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    directory = Path(str(cfg["artifact-dir"])) / str(cfg.get("wire-capture-dir", "wire"))
    original = base.CountingFedAvg
    base.CountingFedAvg = make_metrics_strategy(directory, original)
    try:
        capture.run_captured(grid, context)
    finally:
        base.CountingFedAvg = original
