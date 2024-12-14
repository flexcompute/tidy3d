"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.data.data_array import ElectroStaticDataArray
from tidy3d.components.tcad.data.monitor_data.abstract import (
    FieldDatasetTypes,
    HeatChargeDataset,
    HeatChargeMonitorData,
)
from tidy3d.components.tcad.monitors.charge import (
    StaticCapacitanceMonitor,
    StaticChargeCarrierMonitor,
    StaticVoltageMonitor,
)
from tidy3d.constants import VOLT
from tidy3d.log import log


# Isn't this just static voltage?
class StaticPotentialData(HeatChargeMonitorData):
    """Class that stores electric potential from a charge simulation."""

    monitor: StaticVoltageMonitor = pd.Field(
        ...,
        title="Voltage monitor",
        description="Electric potential monitor associated with a Charge simulation.",
    )

    voltage_series: HeatChargeDataset = pd.Field(
        None, title="Voltage series", description="Contains the voltages."
    )

    @pd.validator("voltage_series", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> StaticPotentialData:
        """Return copy of self with symmetry applied."""

        new_voltages = self._symmetry_expanded_copy(property=self.voltage_series.field_series)

        return self.updated_copy(
            voltage_series=self.voltage_series.updated_copy(field_series=new_voltages)
        )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|²"
        else:
            return "V"


class StaticChargeCarrierData(HeatChargeMonitorData):
    """Class that stores free carrier concentration in Charge simulations."""

    monitor: StaticChargeCarrierMonitor = pd.Field(
        ...,
        title="Free carrier monitor",
        description="Free carrier data associated with a Charge simulation.",
    )

    electrons: HeatChargeDataset = pd.Field(
        None, title="Electrons series", description="Contains the electrons."
    )

    holes: HeatChargeDataset = pd.Field(
        None, title="Holes series", description="Contains the electrons."
    )

    @pd.root_validator(skip_on_failure=True)
    def warn_no_data(cls, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")
        electrons = values.get("electrons")
        holes = values.get("holes")

        if electrons is None or holes is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return values

    @property
    def symmetry_expanded_copy(self) -> StaticChargeCarrierData:
        """Return copy of self with symmetry applied."""

        new_electrons = self._symmetry_expanded_copy(property=self.electrons.field_series)
        new_holes = self._symmetry_expanded_copy(property=self.holes.field_series)

        return self.updated_copy(
            electrons_series=self.electrons.updated_copy(field_series=new_electrons),
            holes_series=self.holes.updated_copy(field_series=new_holes),
        )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "Electrons², Holes²"
        else:
            return "Electrons, Holes"


class StaticCapacitanceData(HeatChargeMonitorData):
    """Class that stores capacitance data from a Charge simulation."""

    monitor: StaticCapacitanceMonitor = pd.Field(
        ...,
        title="Capacitance monitor",
        description="Capacitance data associated with a Charge simulation.",
    )

    hole_capacitance: ElectroStaticDataArray = pd.Field(
        None,
        title="Hole capacitance",
        description="Small signal capacitance (dQh/dV) associated to the monitor.",
    )

    electron_capacitance: ElectroStaticDataArray = pd.Field(
        None,
        title="Electron capacitance",
        description="Small signal capacitance (dQe/dV) associated to the monitor.",
    )

    @pd.validator("hole_capacitance", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        return ""


class StaticVoltageData(HeatChargeMonitorData):
    """Data associated with a :class:`VoltageMonitor`: spatial electric potential field.

    Example
    -------
    >>> from tidy3d import StaticVoltageMonitor, SpatialDataArray
    >>> import numpy as np
    >>> voltage_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> voltage_mnt = StaticVoltageMonitor(size=(1, 2, 3), name="voltage")
    >>> voltage_mnt_data = StaticVoltageData(
    ...     monitor=voltage_mnt, voltage=voltage_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> voltage_mnt_data_expanded = voltage_mnt_data.symmetry_expanded_copy
    """

    monitor: StaticVoltageMonitor = pd.Field(
        ..., title="Monitor", description="Electric potential monitor associated with the data."
    )

    voltage: Optional[FieldDatasetTypes] = pd.Field(
        ...,
        title="Voltage (electric potential)",
        description="Spatial electric potential field.",
        units=VOLT,
    )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|², sigma²"
        else:
            return "V, sigma"

    @pd.validator("voltage", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> StaticVoltageData:
        """Return copy of self with symmetry applied."""

        new_phi = self._symmetry_expanded_copy(property=self.voltage)
        return self.updated_copy(voltage=new_phi, symmetry=(0, 0, 0))
