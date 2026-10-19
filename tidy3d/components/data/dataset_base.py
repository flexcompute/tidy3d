"""The dataset base class and the triangle-mesh dataset.

These sit below :mod:`tidy3d.components.data.dataset`, which also gathers the
electromagnetic, point-cloud and unstructured datasets. ``TriangleMesh`` stores
its surface as a ``TriangleMeshDataset``, and naming that type should not
require the rest of them.
"""

from __future__ import annotations

from abc import ABC

from pydantic import Field

from tidy3d.components.base import Tidy3dBaseModel

from .data_array import DataArray, TriangleMeshDataArray

__all__ = ["Dataset", "TriangleMeshDataset"]


class Dataset(Tidy3dBaseModel, ABC):
    """Abstract base class for objects that store collections of `:class:`.DataArray`s."""

    @property
    def data_arrs(self) -> dict:
        """Returns a dictionary of all `:class:`.DataArray`s in the dataset."""
        data_arrs = {}
        for key in self.__class__.model_fields.keys():
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
