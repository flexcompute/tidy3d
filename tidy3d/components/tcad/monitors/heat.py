"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

import pydantic.v1 as pd
from pydantic.v1 import Field, PositiveInt

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor
from tidy3d.log import log


class TemperatureMonitor(HeatChargeMonitor):
    """Temperature monitor."""

    interval: PositiveInt = Field(
        1,
        title="Interval",
        description="Sampling rate of the monitor: number of time steps between each measurement. "
        "Set ``interval`` to 1 for the highest possible resolution in time. "
        "Higher integer values down-sample the data by measuring every ``interval`` time steps. "
        "This can be useful for reducing data storage as needed by the application."
        "NOTE: this is only relevant for unsteady (transient) Heat simulations. ",
    )

    @pd.root_validator(pre=True)
    def _warn_unstructured_default_change(cls, values):
        """Warn users that the default value of 'unstructured' will change to True after the 2.11 release."""
        # Only warn if 'unstructured' is not explicitly set (using default False)
        if "unstructured" not in values:
            log.warning(
                "The default value of 'unstructured' for 'TemperatureMonitor' will change "
                "from 'False' to 'True' after the 2.11 release. To avoid this warning and ensure "
                "consistent behavior, please explicitly set 'unstructured=True' or 'unstructured=False' "
                "when creating the monitor."
            )
        return values
