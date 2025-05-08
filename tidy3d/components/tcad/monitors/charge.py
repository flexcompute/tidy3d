"""Objects that define how data is recorded from simulation."""

from typing import Literal

import pydantic.v1 as pd

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class SteadyPotentialMonitor(HeatChargeMonitor):
    """
    Electric potential (:math:`\\psi`) monitor.

    Example
    -------
    >>> import tidy3d as td
    >>> voltage_monitor_z0 = td.SteadyPotentialMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="voltage_z0", unstructured=True,
    ... )
    """


class SteadyFreeCarrierMonitor(HeatChargeMonitor):
    """
    Free-carrier monitor for Charge simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> voltage_monitor_z0 = td.SteadyFreeCarrierMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="voltage_z0", unstructured=True,
    ... )
    """

    # NOTE: for the time being supporting unstructured
    unstructured: Literal[True] = pd.Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyEnergyBandMonitor(HeatChargeMonitor):
    """
    Energy bands monitor for Charge simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> energy_monitor_z0 = td.SteadyEnergyBandMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="bands_z0", unstructured=True,
    ... )
    """

    # NOTE: for the time being supporting unstructured
    unstructured: Literal[True] = pd.Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyCapacitanceMonitor(HeatChargeMonitor):
    """
    Capacitance monitor associated with a charge simulation.

    Example
    -------
    >>> import tidy3d as td
    >>> capacitance_global_mnt = td.SteadyCapacitanceMonitor(
    ... center=(0, 0.14, 0), size=(td.inf, td.inf, 0), name="capacitance_global_mnt",
    ... )
    """

    # NOTE: for the time being supporting unstructured
    unstructured: Literal[True] = pd.Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )
