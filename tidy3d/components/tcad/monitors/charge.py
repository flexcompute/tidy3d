"""Objects that define how data is recorded from simulation."""

import pydantic.v1 as pd

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

    @pd.root_validator(skip_on_failure=True)
    def check_unstructured(cls, values):
        """Currently, we're supporting only unstructured monitors in Charge"""
        unstructured = values["unstructured"]
        name = values["name"]
        if not unstructured:
            log.warning(
                "Currently, charge simulations support only unstructured monitors. If monitor "
                f"'{name}' is associated with a charge simulation, please set it tu unstructured. "
                f"This can be done with 'your_monitor = tidy3d.SteadyVoltageMonitor(unstructured=True)'"
            )
        return values


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
    unstructured = True


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

    unstructured = True
