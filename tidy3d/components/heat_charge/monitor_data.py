"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...constants import KELVIN, VOLT
from ...log import log
from ..base import skip_if_fields_missing
from ..base_sim.data.monitor_data import AbstractMonitorData
from ..data.data_array import SpatialDataArray
from ..data.dataset import TetrahedralGridDataset, TriangularGridDataset
from ..types import Coordinate, ScalarSymmetry, annotate_type
from .monitor import HeatChargeMonitorType, TemperatureMonitor, VoltageMonitor

FieldDataset = Union[
    SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
]


class HeatChargeMonitorData(AbstractMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatChargeMonitor`."""

    monitor: HeatChargeMonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
    )

    symmetry: Tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry of the original simulation in x, y, and z.",
    )

    symmetry_center: Coordinate = pd.Field(
        (0, 0, 0),
        title="Symmetry Center",
        description="Symmetry center of the original simulation in x, y, and z.",
    )

    @property
    def symmetry_expanded_copy(self) -> HeatChargeMonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()

    @abstractmethod
    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""

    # def _symmetry_expanded_copy(self, property: FieldDataset) -> FieldDataset:
    def _symmetry_expanded_copy(self, property):
        """Return the property with symmetry applied."""

        # no symmetry
        if all(sym == 0 for sym in self.symmetry):
            return property

        new_property = copy.copy(property)

        mnt_bounds = np.array(self.monitor.bounds)

        if isinstance(new_property, SpatialDataArray):
            data_bounds = [
                [np.min(new_property.x), np.min(new_property.y), np.min(new_property.z)],
                [np.max(new_property.x), np.max(new_property.y), np.max(new_property.z)],
            ]
        else:
            data_bounds = new_property.bounds

        dims_need_clipping_left = []
        dims_need_clipping_right = []
        for dim in range(3):
            # do not expand monitor with zero size along symmetry direction
            # this is done because 2d unstructured data does not support this
            if self.symmetry[dim] == 1:
                center = self.symmetry_center[dim]

                if mnt_bounds[1][dim] < data_bounds[0][dim]:
                    # (note that mnt_bounds[0][dim] < 2 * center - data_bounds[0][dim] will be satisfied based on backend behavior)
                    # simple reflection
                    new_property = new_property.reflect(
                        axis=dim, center=center, reflection_only=True
                    )
                elif mnt_bounds[0][dim] < 2 * center - data_bounds[0][dim]:
                    # expand only if monitor bounds missing data
                    # if we do expand, simply reflect symmetrically the whole data
                    new_property = new_property.reflect(axis=dim, center=center)

                    # if it turns out that we expanded too much, we will trim unnecessary data later
                    if mnt_bounds[0][dim] > 2 * center - data_bounds[1][dim]:
                        dims_need_clipping_left.append(dim)

                    # likewise, if some of original data was only for symmetry expansion, thim excess on the right
                    if mnt_bounds[1][dim] < data_bounds[1][dim]:
                        dims_need_clipping_right.append(dim)

        # trim over-expanded data
        if len(dims_need_clipping_left) > 0 or len(dims_need_clipping_right) > 0:
            # enlarge clipping domain on positive side arbitrary by 1
            # should not matter by how much
            clip_bounds = [mnt_bounds[0] - 1, mnt_bounds[1] + 1]
            for dim in dims_need_clipping_left:
                clip_bounds[0][dim] = mnt_bounds[0][dim]

            for dim in dims_need_clipping_right:
                clip_bounds[1][dim] = mnt_bounds[1][dim]

            if isinstance(new_property, SpatialDataArray):
                new_property = new_property.sel_inside(clip_bounds)
            else:
                new_property = new_property.box_clip(bounds=clip_bounds)

        return new_property


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


class VoltageData(HeatChargeMonitorData):
    """Data associated with a :class:`VoltageMonitor`: spatial electric potential field.

    Example
    -------
    >>> from tidy3d import VoltageMonitor, SpatialDataArray
    >>> import numpy as np
    >>> voltage_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> voltage_mnt = VoltageMonitor(size=(1, 2, 3), name="voltage")
    >>> voltage_mnt_data = VoltageData(
    ...     monitor=voltage_mnt, voltage=voltage_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> voltage_mnt_data_expanded = voltage_mnt_data.symmetry_expanded_copy
    """

    monitor: VoltageMonitor = pd.Field(
        ..., title="Monitor", description="Electric potential monitor associated with the data."
    )

    voltage: Optional[FieldDataset] = pd.Field(
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
    def symmetry_expanded_copy(self) -> VoltageData:
        """Return copy of self with symmetry applied."""

        new_phi = self._symmetry_expanded_copy(property=self.voltage)
        return self.updated_copy(voltage=new_phi, symmetry=(0, 0, 0))


HeatChargeMonitorDataType = Union[TemperatureData, VoltageData]
