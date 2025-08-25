"""Monitor data for unstructured volume mesh monitors."""

from __future__ import annotations

from typing import Union

import pydantic.v1 as pd

from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.mesh import VolumeMeshMonitor

UnstructuredFieldType = Union[TriangularGridDataset, TetrahedralGridDataset]


class VolumeMeshData(HeatChargeMonitorData):
    """Data associated with a :class:`VolumeMeshMonitor`: stores the unstructured mesh.

    Example
    -------

    >>> import tidy3d as td
    >>> import numpy as np
    >>> mesh_mnt = td.VolumeMeshMonitor(size=(1, 2, 3), name="mesh")
    >>> tet_grid_points = td.PointDataArray(
    ...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ...     dims=("index", "axis"),
    ... )
    >>> tet_grid_cells = td.CellDataArray(
    ...     [[0, 1, 2, 4], [1, 2, 3, 4]],
    ...     dims=("cell_index", "vertex_index"),
    ... )
    >>> tet_grid_values = td.IndexedDataArray(
    ...     np.zeros((tet_grid_points.shape[0],)),
    ...     dims=("index",),
    ...     name="Mesh",
    ... )
    >>> tet_grid = td.TetrahedralGridDataset(
    ...     points=tet_grid_points,
    ...     cells=tet_grid_cells,
    ...     values=tet_grid_values,
    ... )
    >>> mesh_mnt_data = td.VolumeMeshData(monitor=mesh_mnt, mesh=tet_grid) # doctest: +SKIP
    """

    monitor: VolumeMeshMonitor = pd.Field(
        ..., title="Monitor", description="Mesh monitor associated with the data."
    )

    mesh: UnstructuredFieldType = pd.Field(
        ...,
        title="Mesh",
        description="Dataset storing the mesh.",
    )

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType]:
        """Maps the field components to their associated data."""
        return {"mesh": self.mesh}

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        return "Mesh"
