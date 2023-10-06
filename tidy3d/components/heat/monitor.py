"""Objects that define how data is recorded from simulation."""
from abc import ABC
from typing import Union

from ..types import ArrayFloat1D
from ..base_sim.monitor import AbstractMonitor


BYTES_REAL = 4


class HeatMonitor(AbstractMonitor, ABC):
    """Abstract base class for heat monitors."""


class TemperatureMonitor(HeatMonitor):
    """Temperature monitor."""

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 real number per grid cell, per time step, per field
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps * num_cells * len(self.fields)


# types of monitors that are accepted by heat simulation
HeatMonitorType = Union[TemperatureMonitor]
