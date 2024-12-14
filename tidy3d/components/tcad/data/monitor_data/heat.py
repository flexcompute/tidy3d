"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.constants import KELVIN
from tidy3d.log import log

# """Monitor level data, store the DataArrays associated with a single heat-charge monitor."""
#
# from __future__ import annotations
#
# import copy
# from abc import ABC, abstractmethod
# from typing import Optional, Tuple, Union
#
# import numpy as np
# import pydantic.v1 as pd
#
# from ...constants import KELVIN, VOLT
# from ...log import log
# from ..base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
# from ..base_sim.data.monitor_data import AbstractMonitorData
# from ..data.data_array import DCCapacitanceDataArray, SpatialDataArray
# from ..data.dataset import IndexedDataArray, TetrahedralGridDataset, TriangularGridDataset
# from ..types import Coordinate, ScalarSymmetry, annotate_type
# from .monitor import (
#     StaticCapacitanceMonitor,
#     StaticChargeCarrierMonitor,
#     HeatChargeMonitorTypes,
#     TemperatureMonitor,
#     StaticVoltageMonitor,
# )
#
# FieldDataset = Union[
#     SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
# ]


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

    @pd.validator("temperature", always=True)
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
        if val == "abs^2":
            return "|T|², K²"
        else:
            return "T, K"

    @property
    def symmetry_expanded_copy(self) -> TemperatureData:
        """Return copy of self with symmetry applied."""

        new_temp = self._symmetry_expanded_copy(property=self.temperature)
        return self.updated_copy(temperature=new_temp, symmetry=(0, 0, 0))
