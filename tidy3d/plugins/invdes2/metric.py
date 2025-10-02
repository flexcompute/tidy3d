from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Union

import autograd.numpy as np

import tidy3d as td


@dataclass
class Metric:
    """Abstract base class for simulation-derived objective terms.

    Attributes
    ----------
    weight:
        Scalar multiplied with the metric value during aggregation.
    """

    monitor_name: str
    weight: float = 1.0

    @abstractmethod
    def evaluate(self, mnt_data: td.MonitorData) -> float:
        """Return a scalar score computed from `monitor_data`."""


@dataclass
class FluxMetric(Metric):
    """Metric based on the flux of a monitor."""

    def evaluate(self, mnt_data: td.FluxData) -> float:
        """Return the flux of the monitor."""
        return np.sum(mnt_data.flux.values)


MetricType = Union[FluxMetric]
