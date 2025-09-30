"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base_sim.data.monitor_data import AbstractUnstructuredMonitorData
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


class HeatChargeMonitorData(AbstractUnstructuredMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatChargeMonitor`."""

    monitor: HeatChargeMonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
    )

    symmetry: tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry of the original simulation in x, y, and z.",
    )

    @abstractmethod
    def symmetry_expanded_copy(self) -> HeatChargeMonitorData:
        """Return copy of self with symmetry applied."""

    @abstractmethod
    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
