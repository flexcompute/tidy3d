"""Objects that define how data is recorded from simulation."""

from abc import ABC
from typing import Union

import pydantic.v1 as pd

from ...log import log
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


class StaticVoltageMonitor(HeatChargeMonitor):
    """Electric potential monitor."""

    @pd.root_validator(skip_on_failure=True)
    def check_unstructured(cls, values):
        """Currently, we're supporting only unstructured monitors in Charge"""
        unstructured = values["unstructured"]
        name = values["name"]
        if not unstructured:
            log.warning(
                "Currently, charge simulations support only unstructured monitors. If monitor "
                f"'{name}' is associated with a charge simulation, please set it tu unstructured. "
                f"This can be done with 'your_monitor = tidy3d.StaticVoltageMonitor(unstructured=True)'"
            )
        return values


class StaticChargeCarrierMonitor(HeatChargeMonitor):
    """Free-carrier monitor for Charge simulations."""

    # NOTE: for the time being supporting unstructured
    unstructured = True


class StaticCapacitanceMonitor(HeatChargeMonitor):
    """Capacitance monitor associated with a charge simulation."""

    unstructured = True


# types of monitors that are accepted by heat simulation
HeatChargeMonitorTypes = Union[
    TemperatureMonitor,
    StaticVoltageMonitor,
    StaticChargeCarrierMonitor,
    StaticCapacitanceMonitor,
]
