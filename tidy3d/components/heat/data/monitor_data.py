"""Monitor level data, store the DataArrays associated with a single heat monitor."""

from __future__ import annotations

from abc import ABC
from typing import Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ....constants import KELVIN
from ....log import log
from ...base import cached_property, skip_if_fields_missing
from ...base_sim.data.monitor_data import AbstractMonitorData
from ...data.data_array import SpatialDataArray
from ...data.dataset import TetrahedralGridDataset, TriangularGridDataset
from ...types import Coordinate, ScalarSymmetry, annotate_type
from ..monitor import HeatMonitorType, TemperatureMonitor


class HeatMonitorData(AbstractMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatMonitor`."""

    monitor: HeatMonitorType = pd.Field(
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

    @cached_property
    def symmetry_expanded_copy(self) -> HeatMonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()


class TemperatureData(HeatMonitorData):
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

    temperature: Optional[
        Union[SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])]
    ] = pd.Field(
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

    @cached_property
    def symmetry_expanded_copy(self) -> TemperatureData:
        """Return copy of self with symmetry applied."""

        # case when no info was recorded (bad placement of monitor and not caught by frontend)
        if self.temperature is None:
            return self.updated_copy(symmetry=(0, 0, 0))

        # no symmetry
        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        new_temp = self.temperature

        mnt_bounds = np.array(self.monitor.bounds)

        if isinstance(new_temp, SpatialDataArray):
            data_bounds = [
                [np.min(new_temp.x), np.min(new_temp.y), np.min(new_temp.z)],
                [np.max(new_temp.x), np.max(new_temp.y), np.max(new_temp.z)],
            ]
        else:
            data_bounds = new_temp.bounds

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
                    new_temp = new_temp.reflect(axis=dim, center=center, reflection_only=True)
                elif mnt_bounds[0][dim] < 2 * center - data_bounds[0][dim]:
                    # expand only if monitor bounds missing data
                    # if we do expand, simply reflect symmetrically the whole data
                    new_temp = new_temp.reflect(axis=dim, center=center)

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

            if isinstance(new_temp, SpatialDataArray):
                new_temp = new_temp.sel_inside(clip_bounds)
            else:
                new_temp = new_temp.box_clip(bounds=clip_bounds)

        return self.updated_copy(temperature=new_temp, symmetry=(0, 0, 0))


HeatMonitorDataType = Union[TemperatureData]
