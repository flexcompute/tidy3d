"""Objects that define how data is recorded from simulation."""

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class TemperatureMonitor(HeatChargeMonitor):
    """Temperature monitor."""
