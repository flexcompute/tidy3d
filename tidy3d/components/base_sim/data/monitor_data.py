"""Abstract base for monitor data structures."""

from __future__ import annotations

import copy
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from tidy3d.components.base_sim.monitor import AbstractMonitor
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.data.dataset import Dataset
from tidy3d.components.types import Coordinate, Symmetry

if TYPE_CHECKING:
    from typing import Literal

    from xarray import DataArray as XrDataArray

    from tidy3d.components.data.utils import UnstructuredGridDatasetType


class AbstractMonitorData(Dataset, ABC):
    """Abstract base class of objects that store data pertaining to a single
    :class:`AbstractMonitor`.
    """

    monitor: AbstractMonitor = Field(
        title="Monitor",
        description="Monitor associated with the data.",
    )

    @property
    def symmetry_expanded_copy(self) -> AbstractMonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()


class AbstractUnstructuredMonitorData(AbstractMonitorData, ABC):
    """Abstract base class of objects that store data from unstructured monitors."""

    symmetry: tuple[Symmetry, Symmetry, Symmetry] = Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry of the original simulation in x, y, and z.",
    )

    symmetry_center: Coordinate = Field(
        (0, 0, 0),
        title="Symmetry Center",
        description="Symmetry center of the original simulation in x, y, and z.",
    )

    def _symmetry_expanded_copy_base(
        self,
        data: UnstructuredGridDatasetType | SpatialDataArray,
        custom_symmetry: tuple[Literal[-1, 1] | XrDataArray, ...] | None = None,
    ) -> UnstructuredGridDatasetType | SpatialDataArray:
        """Return the data with symmetry applied."""

        # no symmetry
        if all(sym == 0 for sym in self.symmetry):
            return data

        new_data = copy.copy(data)

        mnt_bounds = np.array(self.monitor.bounds)

        if isinstance(new_data, SpatialDataArray):
            data_bounds = [
                [np.min(new_data.x), np.min(new_data.y), np.min(new_data.z)],
                [np.max(new_data.x), np.max(new_data.y), np.max(new_data.z)],
            ]
        else:
            data_bounds = new_data.bounds

        dims_need_clipping_left = []
        dims_need_clipping_right = []
        for dim in range(3):
            # do not expand monitor with zero size along symmetry direction
            # this is done because 2d unstructured data does not support this
            if self.symmetry[dim] != 0:
                symmetry_factor = (
                    self.symmetry[dim] if custom_symmetry is None else custom_symmetry[dim]
                )
                center = self.symmetry_center[dim]

                if mnt_bounds[1][dim] < data_bounds[0][dim]:
                    # (note that mnt_bounds[0][dim] < 2 * center - data_bounds[0][dim] will be satisfied based on backend behavior)
                    # simple reflection
                    new_data = new_data.reflect(
                        axis=dim, center=center, reflection_only=True, symmetry=symmetry_factor
                    )
                elif mnt_bounds[0][dim] < 2 * center - data_bounds[0][dim]:
                    # expand only if monitor bounds missing data
                    # if we do expand, simply reflect symmetrically the whole data
                    new_data = new_data.reflect(axis=dim, center=center, symmetry=symmetry_factor)

                    # if it turns out that we expanded too much, we will trim unnecessary data later
                    if mnt_bounds[0][dim] > 2 * center - data_bounds[1][dim]:
                        dims_need_clipping_left.append(dim)

                    # likewise, if some of original data was only for symmetry expansion, trim excess on the right
                    if mnt_bounds[1][dim] < data_bounds[1][dim]:
                        dims_need_clipping_right.append(dim)

        # trim over-expanded data
        if dims_need_clipping_left or dims_need_clipping_right:
            # enlarge clipping domain on positive side arbitrary by 1
            # should not matter by how much
            clip_bounds = [mnt_bounds[0] - 1, mnt_bounds[1] + 1]
            for dim in dims_need_clipping_left:
                clip_bounds[0][dim] = mnt_bounds[0][dim]

            for dim in dims_need_clipping_right:
                clip_bounds[1][dim] = mnt_bounds[1][dim]

            if isinstance(new_data, SpatialDataArray):
                new_data = new_data.sel_inside(clip_bounds)
            else:
                new_data = new_data.box_clip(bounds=clip_bounds)

        return new_data
