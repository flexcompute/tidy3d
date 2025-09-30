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
from tidy3d.constants import MICROMETER
from tidy3d.log import log

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
    def field_components(self) -> dict:
        """Maps the field components to their associated data."""

    def field_name(self, val: str = "") -> str:
        """Gets the name of the fields to be plot."""
        fields = self.field_components.keys()
        name = ""
        for field in fields:
            if val == "abs^2":
                name = f"{field}²"
            else:
                name = f"{field}"
        return name

    @property
    def symmetry_expanded_copy(self) -> HeatChargeMonitorData:
        """Return copy of self with symmetry applied."""

        new_field_components = {}
        for field, val in self.field_components.items():
            new_field_components[field] = self._symmetry_expanded_copy_base(property=val)

        return self.updated_copy(symmetry=(0, 0, 0), **new_field_components)


    def _post_init_validators(self):
        """Call validators taking ``self`` that get run after init."""
        # validate that data exists for all fields
        for field_name, field in self.field_components.items():
            if field is None:
                log.warning(
                    f"No data is available for monitor '{self.monitor.name}' field '{field_name}'. "
                    "This is typically caused by monitor not intersecting any solid medium."
                )
