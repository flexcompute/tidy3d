import pydantic.v1 as pd

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor
from tidy3d.log import log


class StaticVoltageMonitor(HeatChargeMonitor):
    """Electric potential monitor."""

    @pd.root_validator(skip_on_failure=True)
    def check_unstructured(cls, values):
        """Currently, we're supporting only unstructured monitors in Charge"""
        unstructured = values["unstructured"]
        name = values["name"]
        if not unstructured:
            log.warning(
                "Currently, charge simulations support only unstructured monitors. If monitor "
                f"'{name}' is associated with a charge simulation, please set it tu unstructured. "
                f"This can be done with 'your_monitor = tidy3d.StaticVoltageMonitor(unstructured=True)'"
            )
        return values


class StaticChargeCarrierMonitor(HeatChargeMonitor):
    """Free-carrier monitor for Charge simulations."""

    # NOTE: for the time being supporting unstructured
    unstructured = True


class StaticCapacitanceMonitor(HeatChargeMonitor):
    """Capacitance monitor associated with a charge simulation."""

    unstructured = True
