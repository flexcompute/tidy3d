"""Monitor level data, store the DataArrays associated with a single heat monitor."""
from __future__ import annotations
from typing import Union, Tuple

from abc import ABC

import numpy as np
import pydantic.v1 as pd

from ..monitor import TemperatureMonitor, HeatMonitorType
from ...base_sim.data.monitor_data import AbstractMonitorData
from ...data.data_array import SpatialDataArray
from ...types import ScalarSymmetry, Coordinate
from ....constants import KELVIN


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

    @property
    def symmetry_expanded_copy(self) -> HeatMonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()


class TemperatureData(HeatMonitorData):
    """Data associated with a :class:`TemperatureMonitor`: spatial temperature field.

    Example
    -------
    >>> from tidy3d import TemperatureMonitor, SpatialDataArray
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

    temperature: SpatialDataArray = pd.Field(
        ..., title="Temperature", description="Spatial temperature field.", units=KELVIN
    )

    @property
    def symmetry_expanded_copy(self) -> TemperatureData:
        """Return copy of self with symmetry applied."""

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        coords = list(self.temperature.coords.values())
        data = np.array(self.temperature.data)

        for dim in range(3):
            if self.symmetry[dim] == 1:

                sym_center = self.symmetry_center[dim]

                if sym_center == coords[dim].data[0]:
                    num_duplicates = 1
                else:
                    num_duplicates = 0

                shape = np.array(np.shape(data))
                old_len = shape[dim]
                shape[dim] = 2 * old_len - num_duplicates

                ind_left = [slice(shape[0]), slice(shape[1]), slice(shape[2])]
                ind_right = [slice(shape[0]), slice(shape[1]), slice(shape[2])]

                ind_left[dim] = slice(old_len - 1, None, -1)
                ind_right[dim] = slice(old_len - num_duplicates, None)

                new_data = np.zeros(shape)

                new_data[ind_left[0], ind_left[1], ind_left[2]] = data
                new_data[ind_right[0], ind_right[1], ind_right[2]] = data

                new_coords = np.zeros(shape[dim])
                new_coords[old_len - num_duplicates :] = coords[dim]
                new_coords[old_len - 1 :: -1] = 2 * sym_center - coords[dim]

                coords[dim] = new_coords
                data = new_data

        coords_dict = dict(zip("xyz", coords))
        new_temperature = SpatialDataArray(data, coords=coords_dict)

        return self.updated_copy(temperature=new_temperature, symmetry=(0, 0, 0))


HeatMonitorDataType = Union[TemperatureData]
