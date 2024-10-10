"""Objects that define how data is recorded from simulation."""

from abc import ABC
from typing import Union

import pydantic.v1 as pd

from ..base_sim.monitor import AbstractMonitor
from ..types import ArrayFloat1D

BYTES_REAL = 4


class HeatChargeMonitor(AbstractMonitor, ABC):
    """Abstract base class for heat-charge monitors."""

    unstructured: bool = pd.Field(
        False,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )

    conformal: bool = pd.Field(
        False,
        title="Conformal Monitor Meshing",
        description="If ``True`` the heat simulation mesh will conform to the monitor's geometry. "
        "While this can be set for both Cartesian and unstructured monitors, it bears higher "
        "significance for the latter ones. Effectively, setting ``conformal = True`` for "
        "unstructured monitors (``unstructured = True``) ensures that returned temperature values "
        "will not be obtained by interpolation during postprocessing but rather directly "
        "transferred from the computational grid.",
    )

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 real number per grid cell, per time step, per field
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps * num_cells * len(self.fields)


class TemperatureMonitor(HeatChargeMonitor):
    """Temperature monitor."""


class VoltageMonitor(HeatChargeMonitor):
    """Electric potential monitor."""


class ChargeSimulationMonitor(HeatChargeMonitor):
    """Charge monitor for Charge simulations."""

    # NOTE: supporting unstructured only for the time being
    unstructured = True


class TemporalTemperatureMonitor(HeatChargeMonitor):
    """Monitor for time-varying temperature field."""

    # NOTE: supporting unstructured only for the time being
    unstructured = True


class TemporalVoltageMonitor(HeatChargeMonitor):
    """Monitor for time-varying voltage field."""

    # NOTE: supporting unstructured only for the time being
    unstructured = True


# types of monitors that are accepted by heat simulation
HeatChargeMonitorType = Union[
    TemperatureMonitor,
    VoltageMonitor,
    TemporalTemperatureMonitor,
    TemporalVoltageMonitor,
    ChargeSimulationMonitor,
]
