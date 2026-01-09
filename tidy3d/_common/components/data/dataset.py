"""Collections of DataArrays."""

from __future__ import annotations

from abc import ABC

from pydantic import Field

from tidy3d._common.components.base import Tidy3dBaseModel
from tidy3d._common.components.data.data_array import (
    DataArray,
    TriangleMeshDataArray,
)

DEFAULT_MAX_SAMPLES_PER_STEP = 10_000
DEFAULT_MAX_CELLS_PER_STEP = 10_000
DEFAULT_TOLERANCE_CELL_FINDING = 1e-6


class Dataset(Tidy3dBaseModel, ABC):
    """Abstract base class for objects that store collections of `:class:`.DataArray`s."""

    @property
    def data_arrs(self) -> dict:
        """Returns a dictionary of all `:class:`.DataArray`s in the dataset."""
        data_arrs = {}
        for key in self.__fields__.keys():
            data = getattr(self, key)
            if isinstance(data, DataArray):
                data_arrs[key] = data
        return data_arrs


class TriangleMeshDataset(Dataset):
    """Dataset for storing triangular surface data."""

    surface_mesh: TriangleMeshDataArray = Field(
        title="Surface mesh data",
        description="Dataset containing the surface triangles and corresponding face indices "
        "for a surface mesh.",
    )
