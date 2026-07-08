"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tidy3d.components.base_sim.data.monitor_data import AbstractUnstructuredMonitorData
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.types import HeatChargeMonitorType
from tidy3d.components.types import TYPE_TAG_STR, ScalarSymmetry
from tidy3d.components.types.base import discriminated_union
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self

FieldDataset = SpatialDataArray | discriminated_union(
    TriangularGridDataset | TetrahedralGridDataset
)
UnstructuredFieldType = TriangularGridDataset | TetrahedralGridDataset


class HeatChargeMonitorData(AbstractUnstructuredMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatChargeMonitor`."""

    monitor: HeatChargeMonitorType = Field(
        discriminator=TYPE_TAG_STR,
        title="Monitor",
        description="Monitor associated with the data.",
    )

    symmetry: tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry of the original simulation in x, y, and z.",
    )

    @abstractmethod
    def field_components(self) -> dict:
        """Maps the field components to their associated data."""

    @model_validator(mode="after")
    def _warn_missing_fields(self) -> Self:
        """Warn when a monitor field has no data available."""
        for field_name, field_data in self.field_components.items():
            if field_data is None:
                log.warning(
                    f"No data is available for monitor '{self.monitor.name}' field '{field_name}'. "
                    "This is typically caused by monitor not intersecting any solid medium."
                )
        return self

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
            new_field_components[field] = self._symmetry_expanded_copy_base(data=val)

        return self.updated_copy(
            symmetry=(0, 0, 0), **new_field_components, deep=False, validate=False
        )
