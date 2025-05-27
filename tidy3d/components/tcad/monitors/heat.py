"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from pydantic.v1 import Field, PositiveInt

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class TemperatureMonitor(HeatChargeMonitor):
    """Temperature monitor."""

    interval: PositiveInt = Field(
        1,
        title="Interval",
        description="Sampling rate of the monitor: number of time steps between each measurement. "
        "Set ``interval`` to 1 for the highest possible resolution in time. "
        "Higher integer values down-sample the data by measuring every ``interval`` time steps. "
        "This can be useful for reducing data storage as needed by the application."
        "NOTE: this is only relevant for unsteady (transient) Heat simulations. ",
    )
