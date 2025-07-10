"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Optional, Union

import pydantic.v1 as pd

from tidy3d.components.data.data_array import (
    DataArray,
    ScalarFieldTimeDataArray,
    SpatialDataArray,
)
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.heat import (
    TemperatureMonitor,
)
from tidy3d.components.types import annotate_type
from tidy3d.constants import KELVIN

FieldDataset = Union[
    SpatialDataArray,
    ScalarFieldTimeDataArray,
    annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset]),
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

    monitor: TemperatureMonitor = pd.Field(
        ..., title="Monitor", description="Temperature monitor associated with the data."
    )

    temperature: Optional[FieldDataset] = pd.Field(
        ...,
        title="Temperature",
        description="Spatial temperature field.",
        units=KELVIN,
    )

    @property
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return {"temperature": self.temperature}
