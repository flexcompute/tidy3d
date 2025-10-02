from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import tidy3d.web as web


@dataclass
class Metric:
    """Abstract base class for simulation-derived objective terms.

    Attributes
    ----------
    weight:
        Scalar multiplied with the metric value during aggregation.
    """

    weight: float = 1.0

    @abstractmethod
    def evaluate(self, sim_data: web.SimulationData) -> float:
        """Return a scalar score computed from `sim_data`."""
