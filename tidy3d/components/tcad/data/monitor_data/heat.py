"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import Field, model_validator

from tidy3d.components.data.data_array import (
    DataArray,
    IndexedDataArray,
    ScalarFieldTimeDataArray,
    SpatialDataArray,
)
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.components.types import discriminated_union
from tidy3d.constants import KELVIN
from tidy3d.log import log

FieldDataset = Union[
    discriminated_union(Union[TriangularGridDataset, TetrahedralGridDataset]),
    SpatialDataArray,
    ScalarFieldTimeDataArray,
    SpatialDataArray,
]
UnstructuredFieldType = Union[TriangularGridDataset, TetrahedralGridDataset]


class TemperatureData(HeatChargeMonitorData):
    """Data associated with a :class:`TemperatureMonitor`: spatial temperature field.

    Example
    -------
    >>> from tidy3d import TemperatureMonitor, SpatialDataArray
    >>> import numpy as np
    >>> temp_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> temp_mnt = TemperatureMonitor(size=(1, 2, 3), name="temperature")
    >>> temp_mnt_data = TemperatureData(
    ...     monitor=temp_mnt, temperature=temp_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> temp_mnt_data_expanded = temp_mnt_data.symmetry_expanded_copy
    """

    monitor: TemperatureMonitor = Field(
        title="Monitor",
        description="Temperature monitor associated with the data.",
    )

    temperature: Optional[FieldDataset] = Field(
        None,
        title="Temperature",
        description="Spatial temperature field.",
        units=KELVIN,
    )

    @property
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return {"temperature": self.temperature}

    @model_validator(mode="after")
    def warn_no_data(self):
        """Warn if no data provided."""
        if self.temperature is None:
            log.warning(
                f"No data is available for monitor '{self.monitor.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )
        return self

    @model_validator(mode="after")
    def check_correct_data_type(self):
        """Issue error if incorrect data type is used"""
        if isinstance(self.temperature, TetrahedralGridDataset) or isinstance(
            self.temperature, TriangularGridDataset
        ):
            if not isinstance(self.temperature.values, IndexedDataArray):
                raise ValueError(
                    f"Monitor {self.monitor} of type 'TemperatureMonitor' cannot be "
                    "associated with data arrays of type 'IndexVoltageDataArray'."
                )
        return self

    def field_name(self, val: str = "") -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|T|², K²"
        else:
            return "T, K"

    @property
    def symmetry_expanded_copy(self) -> TemperatureData:
        """Return copy of self with symmetry applied."""

        new_temp = self._symmetry_expanded_copy(property=self.temperature)
        return self.updated_copy(temperature=new_temp, symmetry=(0, 0, 0))
