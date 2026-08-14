"""Defines heat-charge material specifications for 'HeatChargeSimulation'"""

from __future__ import annotations

from tidy3d.components.tcad.source.abstract import GlobalHeatChargeSource


class HeatFromElectricSource(GlobalHeatChargeSource):
    """Volumetric heat source generated from an electric simulation.

    Notes
    -----

        If a :class:`HeatFromElectricSource` is specified as a source, appropriate boundary
        conditions for an electric simulation must be provided, since such a simulation
        will be executed before the heat simulation can run.

        This source draws its heating from a ``Conduction`` simulation. It cannot drive a
        heat simulation from a ``Charge`` simulation: a charge simulation solves at a list
        of bias points while a heat simulation solves once, so which bias the temperature
        should correspond to would be ambiguous. Record the charge simulation's
        self-heating with a :class:`SelfHeatingMonitor` and hand a chosen bias point to a
        separate heat simulation as a :class:`HeatSource` instead. For a self-consistent
        charge-heat solution, use a non-isothermal analysis spec
        (:class:`SteadyChargeDCAnalysis`), which couples both physics internally.

    Example
    -------
    >>> heat_source = HeatFromElectricSource()
    """
