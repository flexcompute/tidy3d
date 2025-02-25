"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base_sim.data.monitor_data import AbstractMonitorData
from tidy3d.components.data.data_array import (
    SpatialDataArray,
)
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.types import (
    HeatChargeMonitorType,
)
from tidy3d.components.types import Coordinate, ScalarSymmetry, annotate_type

FieldDataset = Union[
    SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
]
UnstructuredFieldType = Union[TriangularGridDataset, TetrahedralGridDataset]


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

    @abstractmethod
    def symmetry_expanded_copy(self) -> HeatChargeMonitorData:
        """Return copy of self with symmetry applied."""

    @abstractmethod
    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""

    # def _symmetry_expanded_copy(self, property):
    def _symmetry_expanded_copy(self, property: FieldDataset) -> FieldDataset:
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
