"""Defines heat-charge material specifications for 'HeatChargeSimulation'"""

from __future__ import annotations

from tidy3d.components.tcad.source.abstract import GlobalHeatChargeSource


class HeatFromElectricSource(GlobalHeatChargeSource):
    """Volumetric heat source generated from an electric simulation.

    Notes
    -----

        If a :class`HeatFromElectricSource` is specified as a source, appropriate boundary
        conditions for an electric simulation must be provided, since such a simulation
        will be executed before the heat simulation can run.

    Example
    -------
    >>> heat_source = HeatFromElectricSource()
    """
