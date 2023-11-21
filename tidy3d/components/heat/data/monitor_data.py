"""Monitor level data, store the DataArrays associated with a single heat monitor."""
from __future__ import annotations
from typing import Union, Tuple

from abc import ABC

import numpy as np
import pydantic.v1 as pd

from ..monitor import TemperatureMonitor, HeatMonitorType
from ...base_sim.data.monitor_data import AbstractMonitorData
from ...data.data_array import SpatialDataArray
from ...data.dataset import TriangularGridDataset, TetrahedralGridDataset
from ...types import ScalarSymmetry, Coordinate, TYPE_TAG_STR, annotate_type
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

    temperature: Union[
        SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
    ]= pd.Field(
        ..., title="Temperature", description="Spatial temperature field.", units=KELVIN,
    )

    @property
    def symmetry_expanded_copy(self) -> TemperatureData:
        """Return copy of self with symmetry applied."""

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        new_temp = self.temperature

        for dim in range(3):
            if self.symmetry[dim] == 1:

                new_temp = new_temp.reflect(axis=dim, center=self.symmetry_center[dim])

        return self.updated_copy(temperature=new_temp, symmetry=(0, 0, 0))


HeatMonitorDataType = Union[TemperatureData]
