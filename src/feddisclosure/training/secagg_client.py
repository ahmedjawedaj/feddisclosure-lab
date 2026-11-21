"""Flower 1.27 legacy ClientApp + upstream SecAgg+ modifier.

This intentionally does not implement encryption, masking, or key sharing.
Flower's secaggplus_mod owns those operations. Only a private site dataset is
opened by the client, and individual model tensors are never logged here.
"""
from __future__ import annotations
import os

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from feddisclosure.training.datasets import load_site_partition
from feddisclosure.training.flower_client import _site_files
from feddisclosure.training.deployment_contract import record_client_event
import hashlib
from feddisclosure.training.task import DigitMLP, evaluate_local, train_local


class SiteClient(NumPyClient):
    def __init__(self, context: Context):
        self.config = context.run_config
        self.run_id = context.run_id
        site_train, site_val = _site_files(context)
        self.x, self.y = load_site_partition(site_train)
        self.val_x, self.val_y = load_site_partition(site_val)
        self.site_id = int(os.getenv("FDL_SITE_ID", context.node_config.get("partition-id", 0)))
        self.model = DigitMLP(int(self.config["features"]), int(self.config["classes"]))

    def get_parameters(self, config):
        return [v.detach().cpu().numpy().copy() for v in self.model.state_dict().values()]

    def fit(self, parameters, config):
        # Flower SecAgg+ modifier masks the returned parameters before network upload.
        state = self.model.state_dict()
        if len(parameters) != len(state):
            raise ValueError("Mismatch in client parameter count")
        for (name, value), arr in zip(state.items(), parameters):
            if arr.shape != tuple(value.shape) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid received parameter tensor: {name}")
        self.model.load_state_dict({k: torch.as_tensor(v).clone() for k, v in zip(state, parameters)})
        train_local(self.model, self.x, self.y, epochs=int(self.config["local-epochs"]),
                    batch_size=int(self.config["batch-size"]), lr=float(self.config["learning-rate"]),
                    seed=int(self.config["seed"]) + self.site_id)
        if os.getenv("FDL_SITE_DATA_DIR"):
            # Host-side receipt only (never sent to the server): digest of the PLAINTEXT update.
            digest = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                             for v in self.model.state_dict().values())).hexdigest()
            record_client_event(os.environ["FDL_CLIENT_ATTEST_DIR"], event_type="train",
                                site_id=self.site_id, run_id=self.run_id,
                                dataset_sha256=os.environ["FDL_SITE_SHA256"],
                                samples=len(self.y), weights_sha256=digest)
        # No per-client loss or validation metric is exported here.
        return self.get_parameters({}), len(self.y), {}

    def evaluate(self, parameters, config):
        # Server explicitly disables unprotected client evaluation. Keep the
        # NumPyClient contract complete, but do not export sensitive metrics.
        raise RuntimeError("Per-client evaluation disabled in SecAgg+ research mode")


def create_client(context: Context):
    return SiteClient(context).to_client()


def conditional_secaggplus_mod(msg, ctxt, call_next):
    """Apply Flower's upstream secaggplus_mod only when the run enables secure aggregation.

    With use-secagg=false the SAME app serves as the matched plaintext control; the mod's
    own protocol messages are never reimplemented here.
    """
    if bool(ctxt.run_config.get("use-secagg", True)):
        return secaggplus_mod(msg, ctxt, call_next)
    return call_next(msg, ctxt)


app = ClientApp(client_fn=create_client, mods=[conditional_secaggplus_mod])
