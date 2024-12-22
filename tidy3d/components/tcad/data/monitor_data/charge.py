"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Dict, Optional, Union

import pydantic.v1 as pd

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.data.data_array import (
    DataArray,
    IndexedDataArray,
    IndexVoltageDataArray,
    SpatialDataArray,
    SteadyCapacitanceVoltageDataArray,
)
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyFreeChargeCarrierMonitor,
    SteadyVoltageMonitor,
)
from tidy3d.components.types import TYPE_TAG_STR, annotate_type
from tidy3d.constants import VOLT
from tidy3d.log import log

FieldDataset = Union[
    SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
]
UnstructuredFieldType = Union[TriangularGridDataset, TetrahedralGridDataset]


class SteadyVoltageData(HeatChargeMonitorData):
    """Data associated with a :class:`SteadyVoltageMonitor`: spatial electric potential field.

    Example
    -------
    >>> from tidy3d import SteadyVoltageMonitor, SpatialDataArray
    >>> import numpy as np
    >>> voltage_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> voltage_mnt = SteadyVoltageMonitor(size=(1, 2, 3), name="voltage")
    >>> voltage_mnt_data = SteadyVoltageData(
    ...     monitor=voltage_mnt, voltage=voltage_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> voltage_mnt_data_expanded = voltage_mnt_data.symmetry_expanded_copy
    """

    monitor: SteadyVoltageMonitor = pd.Field(
        ..., title="Monitor", description="Electric potential monitor associated with the data."
    )

    voltage: Optional[FieldDataset] = pd.Field(
        ...,
        title="Voltage (electric potential)",
        description="Spatial electric potential field.",
        units=VOLT,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(voltage=self.voltage)

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

    @pd.validator("voltage", always=True)
    @skip_if_fields_missing(["monitor"])
    def check_correct_data_type(cls, val, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")

        if isinstance(val, TetrahedralGridDataset) or isinstance(val, TriangularGridDataset):
            if not isinstance(val.values, IndexedDataArray):
                raise ValueError(
                    f"Monitor {mnt} of type 'SteadyVoltageMonitor' cannot be associated with data arrays "
                    "of type 'IndexVoltageDataArray'."
                )

        return val

    @property
    def symmetry_expanded_copy(self) -> SteadyVoltageData:
        """Return copy of self with symmetry applied."""

        new_phi = self._symmetry_expanded_copy(property=self.voltage)
        return self.updated_copy(voltage=new_phi, symmetry=(0, 0, 0))


class SteadyPotentialData(HeatChargeMonitorData):
    """Class that stores electric potential from a charge simulation."""

    monitor: SteadyVoltageMonitor = pd.Field(
        ...,
        title="Voltage monitor",
        description="Electric potential monitor associated with a Charge simulation.",
    )

    potential: UnstructuredFieldType = pd.Field(
        None,
        title="Voltage series",
        description="Contains the voltages.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(potential=self.potential)

    @pd.validator("potential", always=True)
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

    @pd.validator("potential", always=True)
    @skip_if_fields_missing(["monitor"])
    def check_correct_data_type(cls, val, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")

        if isinstance(val, TetrahedralGridDataset) or isinstance(val, TriangularGridDataset):
            if not isinstance(val.values, IndexVoltageDataArray):
                raise ValueError(
                    f"Monitor {mnt} of type 'SteadyVoltageMonitor' is not associated with data arrays "
                    "of type 'IndexVoltageDataArray' and cannot be associated with an applied voltage."
                )

        return val

    @property
    def symmetry_expanded_copy(self) -> SteadyPotentialData:
        """Return copy of self with symmetry applied."""

        new_potential = self._symmetry_expanded_copy(property=self.potential)
        return self.updated_copy(potential=new_potential, symmetry=(0, 0, 0))

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|²"
        else:
            return "V"


class SteadyFreeCarrierData(HeatChargeMonitorData):
    """Class that stores free carrier concentration in Charge simulations."""

    monitor: SteadyFreeChargeCarrierMonitor = pd.Field(
        ...,
        title="Free carrier monitor",
        description="Free carrier data associated with a Charge simulation.",
    )

    electrons: UnstructuredFieldType = pd.Field(
        None,
        title="Electrons series",
        description="Contains the electrons.",
        discriminator=TYPE_TAG_STR,
    )

    holes: UnstructuredFieldType = pd.Field(
        None,
        title="Holes series",
        description="Contains the electrons.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(electrons=self.electrons, holes=self.holes)

    @pd.root_validator(skip_on_failure=True)
    def check_correct_data_type(cls, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")
        field_data = {field: values.get(field) for field in ["electrons", "holes"]}

        for field, data in field_data.items():
            if isinstance(data, TetrahedralGridDataset) or isinstance(data, TriangularGridDataset):
                if not isinstance(data.values, IndexVoltageDataArray):
                    raise ValueError(
                        f"In the data associated with monitor {mnt}, the field {field} does not contain "
                        "data associated to any voltage value."
                    )

        return values

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
    def symmetry_expanded_copy(self) -> SteadyFreeCarrierData:
        """Return copy of self with symmetry applied."""

        new_electrons = self._symmetry_expanded_copy(property=self.electrons)
        new_holes = self._symmetry_expanded_copy(property=self.holes)

        return self.updated_copy(
            electrons=new_electrons,
            holes=new_holes,
            symmetry=(0, 0, 0),
        )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "Electrons², Holes²"
        else:
            return "Electrons, Holes"


class SteadyCapacitanceData(HeatChargeMonitorData):
    """Class that stores capacitance data from a Charge simulation."""

    monitor: SteadyCapacitanceMonitor = pd.Field(
        ...,
        title="Capacitance monitor",
        description="Capacitance data associated with a Charge simulation.",
    )

    hole_capacitance: SteadyCapacitanceVoltageDataArray = pd.Field(
        None,
        title="Hole capacitance",
        description="Small signal capacitance (dQh/dV) associated to the monitor.",
    )

    electron_capacitance: SteadyCapacitanceVoltageDataArray = pd.Field(
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
