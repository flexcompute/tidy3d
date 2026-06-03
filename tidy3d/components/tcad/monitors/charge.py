"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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
    >>> carrier_monitor_z0 = td.SteadyFreeCarrierMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="carrier_z0", unstructured=True,
    ... )
    """

    # NOTE: for the time being supporting unstructured
    unstructured: Literal[True] = Field(
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
    unstructured: Literal[True] = Field(
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
    unstructured: Literal[True] = Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyElectricFieldMonitor(HeatChargeMonitor):
    """
    Electric field monitor for Charge/Conduction simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> electric_field_monitor_z0 = td.SteadyElectricFieldMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="electric_field_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyCurrentDensityMonitor(HeatChargeMonitor):
    """
    Current density monitor for Charge/Conduction simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> current_density_monitor_z0 = td.SteadyCurrentDensityMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="current_density_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyChargeResidualMonitor(HeatChargeMonitor):
    """
    Per-node residual monitor for Charge simulations (debug tool).

    Notes
    -----
    Records the per-node signed residual of each governing equation:
    :math:`R_\\psi` (Poisson), :math:`R_n` (electron continuity), :math:`R_p`
    (hole continuity), and :math:`R_T` (heat) when the thermal solver is active.
    The electron and hole continuity equations express carrier conservation.
    The values are dimensionless and on the same scale as the simulation's
    convergence tolerance, so the nodes with the largest :math:`|R|` (those
    approaching or exceeding that tolerance) are where the solution least
    satisfies the equations (the least-converged regions). Available only
    through the accelerated solver.

    Example
    -------
    >>> import tidy3d as td
    >>> residual_monitor = td.SteadyChargeResidualMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="residual_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )
