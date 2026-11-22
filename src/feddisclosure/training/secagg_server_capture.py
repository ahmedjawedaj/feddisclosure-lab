"""ServerApp for the wire-level validation (v0.15, WP3).

It runs the unchanged `secagg_server.main` with two additions that only observe.

* The grid is wrapped in `RecordingGrid`. Every message of the four SecAgg+ stages (or of the plain
  fit round in the control run) passes through it. The reply that goes back to the workflow is the
  very object the grid returned.
* The strategy class is swapped for a subclass that saves the aggregate the workflow hands to
  `aggregate_fit`. In a SecAgg+ run that is the dequantised aggregate.

`secagg_server.py` itself is not changed, so the default application behaves exactly as before.
Any failure while recording stops the run. A capture that is not complete is never reported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from flwr.common import Context, parameters_to_ndarrays
from flwr.server import Grid, ServerApp

from feddisclosure.training import secagg_server as base
from feddisclosure.training.wire_capture import WireCaptureError, WireRecorder

app = ServerApp()


class RecordingGrid(Grid):
    """Transparent recording proxy around a Flower grid."""

    def __init__(self, inner: Grid, recorder: WireRecorder):
        self._inner, self._recorder = inner, recorder

    def set_run(self, run_id: int) -> None:
        self._inner.set_run(run_id)

    @property
    def run(self):
        return self._inner.run

    def create_message(self, content, message_type, dst_node_id, group_id, ttl=None):
        return self._inner.create_message(content, message_type, dst_node_id, group_id, ttl)

    def get_node_ids(self):
        return self._inner.get_node_ids()

    def push_messages(self, messages):
        out = list(messages)
        call, stage = self._recorder.record_out(out)
        ids = list(self._inner.push_messages(out))
        self._recorder.remember_pushed(out, ids, call, stage)
        return ids

    def pull_messages(self, message_ids):
        replies = list(self._inner.pull_messages(message_ids))
        for msg in replies:
            call, stage = self._recorder.lookup_pending(msg)
            self._recorder.record_in(call, stage, [msg])
        return replies

    def send_and_receive(self, messages, *, timeout=None):
        out = list(messages)
        call, stage = self._recorder.record_out(out)
        replies = list(self._inner.send_and_receive(out, timeout=timeout))
        self._recorder.record_in(call, stage, replies)
        return replies

    def __getattr__(self, name):
        # Anything else (for example close) goes straight to the real grid.
        return getattr(self._inner, name)


def make_capturing_strategy(recorder: WireRecorder, mode: str, parent=None):
    """Subclass of the application strategy that saves what aggregate_fit receives and returns."""
    parent = parent or base.CountingFedAvg

    class CapturingFedAvg(parent):
        def aggregate_fit(self, server_round, results, failures):
            if mode == "secagg" and results:
                # In a SecAgg+ run every result carries the same dequantised aggregate.
                first = [np.array(a) for a in parameters_to_ndarrays(results[0][1].parameters)]
                for _, fit_res in results[1:]:
                    other = parameters_to_ndarrays(fit_res.parameters)
                    if len(other) != len(first) or any(not np.array_equal(a, b) for a, b in zip(first, other)):
                        raise WireCaptureError("SecAgg+ results disagree about the aggregate")
                recorder.record_aggregate("dequantised_aggregate", first)
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            if parameters is not None:
                recorder.record_aggregate("strategy_output", [np.array(a) for a in parameters_to_ndarrays(parameters)])
            return parameters, metrics

    return CapturingFedAvg


def run_captured(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    mode = "secagg" if bool(cfg.get("use-secagg", True)) else "plaintext"
    if not bool(cfg.get("wire-capture", True)):
        base.main(grid, context)
        return
    directory = Path(str(cfg["artifact-dir"])) / str(cfg.get("wire-capture-dir", "wire"))
    meta = {"seed": int(cfg["seed"]), "clients": int(cfg["clients"]), "max_weight": float(cfg["max-weight"]),
            "num_shares": int(cfg["num-shares"]), "reconstruction_threshold": int(cfg["reconstruction-threshold"]),
            "rounds": int(cfg["num-server-rounds"]), "learning_rate": float(cfg["learning-rate"]),
            "batch_size": int(cfg["batch-size"]), "local_epochs": int(cfg["local-epochs"])}
    recorder = WireRecorder(directory, mode=mode, meta=meta)
    original = base.CountingFedAvg
    base.CountingFedAvg = make_capturing_strategy(recorder, mode, original)
    try:
        base.main(RecordingGrid(grid, recorder), context)
        if "strategy_output" not in recorder._aggregates or (mode == "secagg" and "dequantised_aggregate" not in recorder._aggregates):
            raise WireCaptureError("the aggregate was not captured, the strategy hook did not run")
        recorder.finish(completed=True)
    except BaseException as exc:
        recorder.finish(completed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        base.CountingFedAvg = original


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_captured(grid, context)
