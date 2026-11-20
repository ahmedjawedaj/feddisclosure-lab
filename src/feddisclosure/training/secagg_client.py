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
from feddisclosure.training.task import DigitMLP, evaluate_local, train_local


class SiteClient(NumPyClient):
    def __init__(self, context: Context):
        self.config = context.run_config
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
        # No per-client loss or validation metric is exported here.
        return self.get_parameters({}), len(self.y), {}

    def evaluate(self, parameters, config):
        # Server explicitly disables unprotected client evaluation. Keep the
        # NumPyClient contract complete, but do not export sensitive metrics.
        raise RuntimeError("Per-client evaluation disabled in SecAgg+ research mode")


def create_client(context: Context):
    return SiteClient(context).to_client()


app = ClientApp(client_fn=create_client, mods=[secaggplus_mod])
