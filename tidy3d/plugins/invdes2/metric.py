from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import tidy3d.web as web


@dataclass
class Metric:
    weight: float = 1.0

    @abstractmethod
    def evaluate(self, sim_data: web.SimulationData) -> float:
        pass
