"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from pydantic import Field, PositiveInt

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class TemperatureMonitor(HeatChargeMonitor):
    """Temperature monitor."""

    interval: PositiveInt = Field(
        1,
        title="Interval",
        description="Sampling rate of the monitor: number of time steps between each measurement. "
        "Set ``interval`` to 1 for the highest possible resolution in time. "
        "Higher integer values down-sample the data by measuring every ``interval`` time steps. "
        "This can be useful for reducing data storage as needed by the application. "
        "The last time step is included in the output when ``interval`` is commensurate "
        "with ``total_time_steps`` (i.e. ``total_time_steps`` is divisible by ``interval``). "
        "To capture only the final state, set ``interval`` equal to ``total_time_steps`` "
        "in the simulation's ``UnsteadySpec``. "
        "NOTE: this is only relevant for unsteady (transient) Heat simulations. ",
    )
