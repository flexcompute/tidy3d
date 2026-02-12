"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor
from tidy3d.log import log


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

    @model_validator(mode="before")
    @classmethod
    def _warn_unstructured_default_change(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Warn users that the default value of 'unstructured' will change to True after the 2.11 release."""
        # Only warn if 'unstructured' is not explicitly set (using default False)
        if "unstructured" not in values:
            log.warning(
                "The default value of 'unstructured' for 'SteadyPotentialMonitor' will change "
                "from 'False' to 'True' after the 2.11 release. To avoid this warning and ensure "
                "consistent behavior, please explicitly set 'unstructured=True' or 'unstructured=False' "
                "when creating the monitor."
            )
        return values


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
