"""Fail-closed OpenDP count adapter; no custom DP noise generator.

A separate privacy budget ledger is enforced by the site service. This adapter
asserts OpenDP's own privacy map before permitting any released value.
"""
from __future__ import annotations
import math


class DPLibraryUnavailable(RuntimeError):
    """OpenDP cannot execute in this environment."""


class DPConfigurationError(ValueError):
    """Invalid target epsilon or OpenDP proof map."""


class OpenDPCount:
    """Pure (epsilon, 0)-DP for ONE local count with add/remove-one adjacency.

    The count's global sensitivity must be 1 and the input must be a single
    bounded per-record count statistic. This cannot validate the preceding
    query/selection process; the site restricts it to an allowlisted count.
    """

    def __init__(self, epsilon: float):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise DPConfigurationError("epsilon must be finite and positive")
        try:
            import opendp.prelude as dp
        except ImportError as exc:
            raise DPLibraryUnavailable("Install OpenDP to execute this defense") from exc
        dp.enable_features("contrib")
        # Small safety margin avoids a floating-point privacy-map overshoot.
        self.measurement = dp.m.make_laplace(
            dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float),
            scale=(1.0 + 1e-9) / epsilon,
        )
        proof_epsilon = float(self.measurement.map(d_in=1.0))
        if not math.isfinite(proof_epsilon) or proof_epsilon > epsilon:
            raise DPConfigurationError("OpenDP privacy map exceeds configured epsilon")
        self.epsilon = epsilon
        self.verified_map_epsilon = proof_epsilon

    def release(self, count: int) -> float:
        if not isinstance(count, int) or not 0 <= count < 2**53:
            raise ValueError("Only finite nonnegative integer counts are supported")
        result = float(self.measurement(float(count)))
        if not math.isfinite(result):
            raise RuntimeError("Refusing nonfinite OpenDP release")
        # Do not clip at zero here: raw-dependent postprocessing is avoided.
        return result
