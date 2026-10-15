"""Point, cell, triangle-mesh, and index-keyed data arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .base import DataArray


class PointDataArray(DataArray):
    """A two-dimensional array that stores coordinates/field components for a collection of points.
    Dimension ``index`` denotes the index of a point in the collection, and dimension ``axis``
    denotes the field component (or point coordinate) in that direction.

    Example
    -------
    >>> point_array = PointDataArray(
    ...     (1+1j) * np.random.random((5, 3)), coords=dict(index=np.arange(5), axis=np.arange(3)),
    ... )
    >>> # get coordinates of a point number 3
    >>> point3 = point_array.sel(index=3)
    >>> # get x coordinates of all points
    >>> x_coords = point_array.sel(axis=0)
    >>>
    >>> field_da = PointDataArray(
    ...     np.random.random((120, 3)), coords=dict(index=np.arange(120), axis=np.arange(3)),
    ... )
    >>> # get field of point number 90
    >>> field_point90 = field_da.sel(index=90)
    >>> # get z component of all points
    >>> z_field = field_da.sel(axis=2)
    """

    __slots__ = ()
    _dims = ("index", "axis")

    def __init__(self, data: Any, *args: Any, **kwargs: Any) -> None:
        skip_canonicalization = kwargs.pop("_skip_bare_canonicalization", False)
        if (
            not skip_canonicalization
            and not args
            and "coords" not in kwargs
            and "dims" not in kwargs
        ):
            shape = np.shape(data)
            if (
                len(shape) == 2
                and shape[1] == 3
                and not isinstance(data, xr.Variable | xr.DataArray)
            ):
                kwargs["dims"] = ("index", "axis")
                kwargs["coords"] = {"index": np.arange(shape[0]), "axis": np.arange(3)}
        super().__init__(data, *args, **kwargs)

    @classmethod
    def _constructor_kwargs_for_pydantic(cls) -> dict[str, Any]:
        """Keep pydantic parsing strict for raw lists and arrays."""
        return {"_skip_bare_canonicalization": True}


class CellDataArray(DataArray):
    """A two-dimensional array that stores indices of points composing each cell in a collection of
    cells of the same type (for example: triangles, tetrahedra, etc). Dimension ``cell_index``
    denotes the index of a cell in the collection, and dimension ``vertex_index`` denotes placement
    (index) of a point in a cell (for example: 0, 1, or 2 for triangles; 0, 1, 2, or 3 for
    tetrahedra).

    Example
    -------
    >>> cell_array = CellDataArray(
    ...     (1+1j) * np.random.random((4, 3)),
    ...     coords=dict(cell_index=np.arange(4), vertex_index=np.arange(3)),
    ... )
    >>> # get indices of points composing cell number 3
    >>> cell3 = cell_array.sel(cell_index=3)
    >>> # get indices of points that represent the first vertex in each cell
    >>> first_vertices = cell_array.sel(vertex_index=0)
    """

    __slots__ = ()
    _dims = ("cell_index", "vertex_index")


class TriangleMeshDataArray(DataArray):
    """Data of the triangles of a surface mesh as in the STL file format."""

    __slots__ = ()
    _dims = ("face_index", "vertex_index", "axis")
    _data_attrs = {"long_name": "surface mesh triangles"}


class IndexedDataArray(DataArray):
    """Stores a one-dimensional array enumerated by coordinate ``index``. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated data or a ``CellDataArray``
    to store cell-associated data.

    Example
    -------
    >>> indexed_array = IndexedDataArray(
    ...     (1+1j) * np.random.random((3,)), coords=dict(index=np.arange(3))
    ... )
    """

    __slots__ = ()
    _dims = ("index",)


class IndexedVoltageDataArray(DataArray):
    """Stores a two-dimensional array with coordinates ``index`` and ``voltage``, where
    ``index`` is usually associated with ``PointDataArray`` and ``voltage`` indicates at what
    bias/DC-voltage the data was obtained with.

    Example
    -------
    >>> indexed_array = IndexedVoltageDataArray(
    ...     (1+1j) * np.random.random((3,2)), coords=dict(index=np.arange(3), voltage=[-1, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "voltage")


class IndexedTimeDataArray(DataArray):
    """Stores a two-dimensional array with coordinates ``index`` and ``t``, where
    ``index`` is usually associated with ``PointDataArray`` and ``t`` indicates at what
    simulated time the data was obtained.

    Example
    -------
    >>> indexed_array = IndexedTimeDataArray(
    ...     (1+1j) * np.random.random((3,2)), coords=dict(index=np.arange(3), t=[0, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "t")


class IndexedFieldVoltageDataArray(DataArray):
    """Stores indexed values of vector fields for different voltages. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.
    Example
    -------
    >>> indexed_array = IndexedFieldVoltageDataArray(
    ...     (1+1j) * np.random.random((4,3,2)), coords=dict(index=np.arange(4), axis=np.arange(3), voltage=[-1, 1])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "axis", "voltage")


class IndexedSurfaceFreqDataArray(DataArray):
    """Stores indexed values of scalar fields on the sides of a surface. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated scalar data.

    Example
    -------
    >>> surface_side_array = IndexedSurfaceFreqDataArray(
    ...     (1+1j) * np.random.random((4,2,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "f")


class IndexedSurfaceTimeDataArray(DataArray):
    """Stores indexed values of scalar fields on the sides of a surface. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated scalar data.

    Example
    -------
    >>> surface_side_array = IndexedSurfaceTimeDataArray(
    ...     (1+1j) * np.random.random((4,2,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], t=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "t")


class IndexedSurfaceFieldDataArray(DataArray):
    """Stores indexed values of vector fields on the sides of a surface in frequency domain.
    It is typically used in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedSurfaceFieldDataArray(
    ...     (1+1j) * np.random.random((4,2,3,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], axis=np.arange(3), f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "axis", "f")


class IndexedSurfaceFieldTimeDataArray(DataArray):
    """Stores indexed values of vector fields on the sides of a surface in time domain.
    It is typically used in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedSurfaceFieldTimeDataArray(
    ...     (1+1j) * np.random.random((4,2,3,1)), coords=dict(index=np.arange(4), side=["outside", "inside"], axis=np.arange(3), t=[0])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "side", "axis", "t")


class IndexedFieldDataArray(DataArray):
    """Stores indexed values of vector fields in frequency domain. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedFieldDataArray(
    ...     (1+1j) * np.random.random((4,3,1)), coords=dict(index=np.arange(4), axis=np.arange(3), f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "axis", "f")


class IndexedFieldTimeDataArray(DataArray):
    """Stores indexed values of vector fields in time domain. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedFieldTimeDataArray(
    ...     (1+1j) * np.random.random((4,3,1)), coords=dict(index=np.arange(4), axis=np.arange(3), t=[0])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "axis", "t")


class IndexedFreqDataArray(DataArray):
    """Stores indexed values of scalar fields in frequency domain. It is typically used
    in conjuction with a ``PointDataArray`` to store point-associated vector data.

    Example
    -------
    >>> indexed_array = IndexedFreqDataArray(
    ...     (1+1j) * np.random.random((4,1)), coords=dict(index=np.arange(4), f=[1e9])
    ... )
    """

    __slots__ = ()
    _dims = ("index", "f")


IndexedDataArrayTypes = (
    IndexedDataArray
    | IndexedVoltageDataArray
    | IndexedSurfaceFieldDataArray
    | IndexedSurfaceFieldTimeDataArray
    | IndexedFieldDataArray
    | IndexedFieldTimeDataArray
    | IndexedFreqDataArray
    | IndexedTimeDataArray
    | IndexedFieldVoltageDataArray
    | IndexedSurfaceFreqDataArray
    | IndexedSurfaceTimeDataArray
    | PointDataArray
)
