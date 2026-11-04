"""Defines the temperature monitor of a fast-heat simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, PositiveInt

from tidy3d.components.base_sim.monitor import BoxMonitor

if TYPE_CHECKING:
    from tidy3d.components.types import ArrayFloat1D

BYTES_FLOAT64 = 8


class FastHeatTemperatureMonitor(BoxMonitor):
    """Temperature monitor of a :class:`.FastHeatSimulation`, recording the temperature on the
    simulation grid over its region.

    Example
    -------
    >>> monitor = FastHeatTemperatureMonitor(size=(1, 1, 0), name="temperature")
    """

    interval: PositiveInt = Field(
        default=1,
        title="Interval",
        description="Sampling rate of the monitor: number of time steps between each "
        "measurement, taken at steps ``interval``, ``2 * interval``, ... up to "
        "``total_time_steps``. Only relevant for transient simulations.",
    )

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage: one float64 temperature per cell and sample."""
        return BYTES_FLOAT64 * num_cells * max(1, len(tmesh) // self.interval)
