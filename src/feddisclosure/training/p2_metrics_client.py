"""ClientApp for the P2 probe (v0.16.13, row S4): the SecAgg+ client of this repository, with fit metrics.

The repository client returns an empty metrics dictionary, so the metrics half of W5 in v0.15 was not
exercised. This client returns three metrics so that a run can show whether they reach the server.

* probe_const: a float equal to 1234.5 plus the site index.
* train_loss: a float, the mean cross entropy of the final local model on the client's own records.
* n_local: an integer, the local record count.

The exact dictionary that was sent is appended to `fit_metrics_sent.jsonl` in the client's own
attestation mount. That file is a host side receipt and is never sent to the server. The metrics are
probe values for a visibility check. No claim about what they reveal is made.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from flwr.client import ClientApp

from feddisclosure.training.secagg_client import SiteClient, conditional_secaggplus_mod
from feddisclosure.training.task import evaluate_local

RECEIPT_NAME = "fit_metrics_sent.jsonl"
PROBE_BASE = 1234.5


def probe_metrics(site_id: int, local_loss: float, n_local: int) -> dict:
    return {"probe_const": PROBE_BASE + float(site_id), "train_loss": float(local_loss), "n_local": int(n_local)}


class MetricsSiteClient(SiteClient):
    def fit(self, parameters, config):
        params, n, _ = super().fit(parameters, config)
        loss = float(evaluate_local(self.model, self.x, self.y)["loss"])
        metrics = probe_metrics(self.site_id, loss, len(self.y))
        directory = os.environ.get("FDL_CLIENT_ATTEST_DIR")
        if directory and Path(directory).is_dir():
            line = json.dumps({"site_id": self.site_id, "run_id": str(self.run_id), "metrics": metrics}, sort_keys=True)
            fd = os.open(Path(directory) / RECEIPT_NAME, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.fchmod(fd, 0o644)
                os.write(fd, (line + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        return params, n, metrics


def create_client(context):
    return MetricsSiteClient(context).to_client()


app = ClientApp(client_fn=create_client, mods=[conditional_secaggplus_mod])
