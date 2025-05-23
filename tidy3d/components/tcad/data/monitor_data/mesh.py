"""Monitor data for unstructured volume mesh monitors."""

from __future__ import annotations

from typing import Dict, Union

import pydantic.v1 as pd

from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.mesh import VolumeMeshMonitor

UnstructuredFieldType = Union[TriangularGridDataset, TetrahedralGridDataset]


class VolumeMeshData(HeatChargeMonitorData):
    """Data associated with a :class:`VolumeMeshMonitor`: stores the unstructured mesh."""

    monitor: VolumeMeshMonitor = pd.Field(
        ..., title="Monitor", description="Mesh monitor associated with the data."
    )

    mesh: UnstructuredFieldType = pd.Field(
        ...,
        title="Mesh",
        description="Dataset storing the mesh.",
    )

    @property
    def field_components(self) -> Dict[str, UnstructuredFieldType]:
        """Maps the field components to their associated data."""
        return dict(mesh=self.mesh)

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        return "Mesh"

    @property
    def symmetry_expanded_copy(self) -> VolumeMeshData:
        """Return copy of self with symmetry applied."""

        new_temp = self._symmetry_expanded_copy(property=self.temperature)
        return self.updated_copy(temperature=new_temp, symmetry=(0, 0, 0))
