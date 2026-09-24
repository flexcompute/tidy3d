"""Defines abstract base for unstructured datasets."""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import RangeIndex
from pydantic import Field, field_validator, model_validator
from xarray import DataArray as XrDataArray
from xarray import concat as xr_concat

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.data_array import (
    DATA_ARRAY_MAP,
    MAX_SPATIAL_SAMPLES,
    CellDataArray,
    IndexedDataArray,
    IndexedDataArrayTypes,
    PointDataArray,
    SpatialDataArray,
    _clamp_steps_to_budget,
    _grid_from_bounds,
    _grid_steps,
)
from tidy3d.constants import fp_eps, inf
from tidy3d.exceptions import DataError, Tidy3dNotImplementedError, ValidationError
from tidy3d.log import log
from tidy3d.packaging import requires_vtk, vtk

if TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray
    from pydantic import PositiveInt
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkDataSet,
        vtkPointData,
        vtkPolyData,
        vtkUnstructuredGrid,
    )

    from tidy3d.compat import Self
    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.types import ArrayLike, Axis, Bound

DEFAULT_MAX_SAMPLES_PER_STEP = 10_000
DEFAULT_MAX_CELLS_PER_STEP = 10_000
DEFAULT_TOLERANCE_CELL_FINDING = 1e-6

# Carries each cell's own index through a plane cut, so a cut point can be traced back to
# the mesh edge it was interpolated on.
SOURCE_CELL_ARRAY_NAME = "_tidy3d_source_cell"

# Cut points whose mesh edge is searched for at a time. The search is vectorised over a
# '(rows, num_edges, 2, 3)' tensor of edge endpoint coordinates, so on a large slice the
# chunk, not the slice, is what sets its peak memory.
CUT_POINT_EDGE_SEARCH_CHUNK = 16_384

# Allow boundary roundoff without accepting unstable weights from near-degenerate cells.
BARYCENTRIC_WEIGHT_TOLERANCE = 1e-6

# Scales with extent (not absolute coordinate) so detection is origin-independent;
# absolute floor handles degenerate/tiny slices where the relative term underflows.
PLANAR_ZERO_DIM_TOLERANCE_ABS = 1e-6
PLANAR_ZERO_DIM_TOLERANCE_REL = 2e-8
_WARN_UNUSED_POINTS = ContextVar("tidy3d_warn_unused_unstructured_points", default=True)


@contextmanager
def _suppress_unstructured_grid_unused_point_warnings() -> Generator[None]:
    """Suppress unused-point warnings while rebuilding trusted internal solver datasets."""
    token = _WARN_UNUSED_POINTS.set(False)
    try:
        yield
    finally:
        _WARN_UNUSED_POINTS.reset(token)


def planar_zero_dim_tolerance(size_scale: float) -> float:
    """Tolerance for treating a slice's nominally zero-thickness axis as zero."""
    return max(PLANAR_ZERO_DIM_TOLERANCE_ABS, PLANAR_ZERO_DIM_TOLERANCE_REL * size_scale)


def _merge_representative_points(
    group_of_point: NDArray, source_edges: NDArray, subset: NDArray, num_points: int
) -> NDArray:
    """Cut point each cut point merges into: the lowest-indexed one of its merge group.

    Two coincident cut points are the same node when the mesh edges they were interpolated
    from share an original point, and that relation is transitive -- the several edges
    meeting at a vertex the plane passes through all collapse together. Solved for the
    whole slice at once as a connected-components problem: each coincident point is joined
    to the two ``(coordinate group, original point)`` keys of its edge, so points sharing
    an original point meet at that key, while a key names one coordinate group by
    construction and no component can span two coordinates.

    ``source_edges`` holds one row per point selected by ``subset``, in point order.
    """

    representative = np.arange(num_points)
    num_rows = len(source_edges)
    if num_rows == 0:
        return representative

    # scipy is a dependency but not a module-level import anywhere in the client
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # one integer per key: both factors are bounded by point counts, so this stays well
    # inside int64, and packing them lets the labelling sort numbers instead of rows
    stride = np.int64(source_edges.max()) + 1
    keys = np.repeat(group_of_point[subset].astype(np.int64), 2) * stride
    keys += source_edges.ravel()
    _, key_of_entry = np.unique(keys, return_inverse=True)
    key_of_entry = key_of_entry.ravel()

    # rows 0..num_rows-1 are the points, the rest are the keys they meet at
    num_nodes = num_rows + int(key_of_entry.max()) + 1
    incidence = coo_matrix(
        (
            np.ones(len(key_of_entry), dtype=np.int8),
            (np.repeat(np.arange(num_rows), 2), num_rows + key_of_entry),
        ),
        shape=(num_nodes, num_nodes),
    )
    _, label_of_row = connected_components(incidence, directed=False, return_labels=True)
    label_of_row = label_of_row[:num_rows]

    # lowest row of each component, by scattering the rows in reverse so it writes last
    lowest_row = np.empty(int(label_of_row.max()) + 1, dtype=np.int64)
    lowest_row[label_of_row[::-1]] = np.arange(num_rows - 1, -1, -1)

    point_of_row = np.flatnonzero(subset)
    representative[point_of_row] = point_of_row[lowest_row[label_of_row]]
    return representative


@requires_vtk
def cut_point_neighbor_positions(cut: vtkPolyData, axis: Axis) -> NDArray:
    """Mean position along ``axis`` of the points each point shares a line segment with.

    Two points of a jump pair sit at one coordinate, so the coordinate alone cannot order
    them. Their segments can: each connects only to its own side of the interface, so the
    point whose neighbors lie lower is the lower side's value. Points with no segment
    report their own position.
    """

    points = vtk["vtk_to_numpy"](cut.GetPoints().GetData())
    totals = np.zeros(len(points))
    counts = np.zeros(len(points))

    lines = cut.GetLines()
    if lines is not None and lines.GetNumberOfCells() > 0:
        connectivity = vtk["vtk_to_numpy"](lines.GetConnectivityArray())
        offsets = vtk["vtk_to_numpy"](lines.GetOffsetsArray())
        sizes = np.diff(offsets)
        cell_of_entry = np.repeat(np.arange(len(sizes)), sizes)
        own = points[connectivity, axis]

        # a point's neighbors are the rest of its cell, so the cell's total less the point
        # itself. A plane laid on a mesh node cuts zero-length segments that list one point
        # twice, and a point is not its own neighbor, so every copy of it is discounted.
        entry_key = cell_of_entry * len(points) + connectivity
        _, group_of_entry, copies = np.unique(entry_key, return_inverse=True, return_counts=True)
        copies = copies[group_of_entry]

        cell_totals = np.bincount(cell_of_entry, weights=own, minlength=len(sizes))
        totals = np.bincount(
            connectivity, weights=cell_totals[cell_of_entry] - copies * own, minlength=len(points)
        )
        counts = np.bincount(
            connectivity, weights=sizes[cell_of_entry] - copies, minlength=len(points)
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), points[:, axis])


@requires_vtk
def _cut_cell_point_pairs(cut: vtkPolyData) -> tuple[NDArray, NDArray]:
    """Every (cut point, original cell) incidence in the cut, as two parallel arrays.

    The cutter is asked not to merge its points, so a point is normally listed by exactly
    one cut cell. Several means the cutter merged anyway, which the caller checks rather
    than assumes benign: cut points merged across a material interface name mesh edges
    with no node in common.
    """

    source_array = cut.GetCellData().GetArray(SOURCE_CELL_ARRAY_NAME)
    if source_array is None:
        raise DataError(
            f"Cut output carries no '{SOURCE_CELL_ARRAY_NAME}' cell array, so its points cannot "
            "be traced back to the cells they were cut from."
        )
    source_of_cut_cell = vtk["vtk_to_numpy"](source_array)

    points, sources = [], []
    first_cut_cell = 0
    for cell_array in _polydata_cell_arrays(cut):
        num_cut_cells = cell_array.GetNumberOfCells()
        if num_cut_cells == 0:
            continue
        connectivity = vtk["vtk_to_numpy"](cell_array.GetConnectivityArray())
        offsets = vtk["vtk_to_numpy"](cell_array.GetOffsetsArray())
        points.append(connectivity)
        # VTK numbers cells across the arrays in turn, so each array continues the slice
        sources.append(
            np.repeat(
                source_of_cut_cell[first_cut_cell : first_cut_cell + num_cut_cells],
                np.diff(offsets),
            )
        )
        first_cut_cell += num_cut_cells

    if not points:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(points), np.concatenate(sources)


@requires_vtk
def _polydata_cell_arrays(polydata: vtkPolyData) -> list[vtkCellArray]:
    """The polydata's cell arrays, in the order VTK numbers their cells."""

    arrays = [polydata.GetVerts(), polydata.GetLines(), polydata.GetPolys(), polydata.GetStrips()]
    return [array for array in arrays if array is not None]


@requires_vtk
def _rebuild_polydata(
    cut: vtkPolyData, points: NDArray, point_remap: NDArray, kept: NDArray
) -> vtkPolyData:
    """Rebuild ``cut`` on a reduced point set, re-indexing cells through ``point_remap``.

    Every buffer handed to VTK is deep-copied: the numpy arrays here are locals, and a
    shallow wrapper would outlive them.
    """

    mod = vtk["mod"]
    out = mod.vtkPolyData()

    vtk_points = mod.vtkPoints()
    vtk_points.SetData(vtk["numpy_to_vtk"](points, deep=True))
    out.SetPoints(vtk_points)

    setters = {
        "Verts": out.SetVerts,
        "Lines": out.SetLines,
        "Polys": out.SetPolys,
        "Strips": out.SetStrips,
    }
    for name, setter in setters.items():
        cell_array = getattr(cut, f"Get{name}")()
        if cell_array is None or cell_array.GetNumberOfCells() == 0:
            continue
        connectivity = vtk["vtk_to_numpy"](cell_array.GetConnectivityArray())
        offsets = vtk["vtk_to_numpy"](cell_array.GetOffsetsArray())
        rebuilt = mod.vtkCellArray()
        rebuilt.SetData(
            vtk["numpy_to_vtkIdTypeArray"](offsets.astype(vtk["id_type"]), deep=True),
            vtk["numpy_to_vtkIdTypeArray"](
                point_remap[connectivity].astype(vtk["id_type"]), deep=True
            ),
        )
        setter(rebuilt)

    source_data, target_data = cut.GetPointData(), out.GetPointData()
    for array_ind in range(source_data.GetNumberOfArrays()):
        source_array = source_data.GetArray(array_ind)
        values = vtk["vtk_to_numpy"](source_array)[kept]
        target_array = vtk["numpy_to_vtk"](np.ascontiguousarray(values), deep=True)
        target_array.SetName(source_array.GetName())
        target_data.AddArray(target_array)

    return out


def _as_sequence_selector(value: Any) -> Any:
    """Wrap a scalar in a length-1 list so xarray keeps the dimension it indexes.

    Array-likes pass through. Uses ``np.ndim`` rather than ``isinstance(value, list)``,
    which nested numpy arrays and made xarray reject them as multi-dimensional indexers.
    """
    return value if np.ndim(value) > 0 else [value]


class UnstructuredDataset(Tidy3dBaseModel, np.lib.mixins.NDArrayOperatorsMixin, ABC):
    """Abstract base for datasets that store unstructured grid or surface data."""

    points: PointDataArray = Field(
        title="Grid Points",
        description="Coordinates of points composing the unstructured grid.",
    )

    values: IndexedDataArrayTypes = Field(
        title="Point Values",
        description="Values stored at the grid points.",
    )

    cells: CellDataArray = Field(
        title="Grid Cells",
        description="Cells composing the unstructured grid specified as connections between grid "
        "points.",
    )

    """ Fundametal parameters to set up based on grid dimensionality """

    @classmethod
    @abstractmethod
    def _point_dims(cls) -> PositiveInt:
        """Dimensionality of stored grid point coordinates."""

    @classmethod
    @abstractmethod
    def _cell_num_vertices(cls) -> PositiveInt:
        """Number of vertices in a cell."""

    @classmethod
    def _cell_edges(cls) -> NDArray:
        """Vertex-index pairs of a cell's edges, as (num_edges, 2)."""
        num_vertices = cls._cell_num_vertices()
        return np.array(
            [(i, j) for i in range(num_vertices - 1) for j in range(i + 1, num_vertices)],
            dtype=int,
        )

    """ Validators """

    @field_validator("points")
    @classmethod
    def points_right_dims(cls, val: PointDataArray) -> PointDataArray:
        """Check that point coordinates have the right dimensionality."""
        # currently support only the standard axis ordering, that is 01(2)
        axis_coords_expected = np.arange(cls._point_dims())
        axis_coords_given = val.axis.data
        if np.any(axis_coords_given != axis_coords_expected):
            raise ValidationError(
                f"Points array is expected to have {axis_coords_expected} coord values along 'axis'"
                f" (given: {axis_coords_given})."
            )
        return val

    @field_validator("points")
    @classmethod
    def points_right_indexing(cls, val: PointDataArray) -> PointDataArray:
        """Check that points are indexed corrrectly."""
        indices_expected = np.arange(len(val.data))
        indices_given = val.index.data
        if np.any(indices_expected != indices_given):
            raise ValidationError(
                "Coordinate 'index' of array 'points' is expected to have values (0, 1, 2, ...). "
                "This can be easily achieved, for example, by using "
                "PointDataArray(data, dims=['index', 'axis'])."
            )
        return val

    @field_validator("values")
    @classmethod
    def first_values_dim_is_index(cls, val: IndexedDataArrayTypes) -> IndexedDataArrayTypes:
        """Check that the number of data values matches the number of grid points."""
        if val.dims[0] != "index":
            raise ValidationError("First dimension of array 'values' must be 'index'.")
        return val

    @field_validator("values")
    @classmethod
    def values_right_indexing(cls, val: IndexedDataArrayTypes) -> IndexedDataArrayTypes:
        """Check that data values are indexed correctly."""
        # currently support only simple ordered indexing of points, that is, 0, 1, 2, ...
        indices_expected = np.arange(len(val.index.data))
        indices_given = val.index.data
        if np.any(indices_expected != indices_given):
            raise ValidationError(
                "Coordinate 'index' of array 'values' is expected to have values (0, 1, 2, ...). "
                "This can be easily achieved, for example, by using "
                "IndexedDataArray(data, dims=['index'])."
            )
        return val

    @model_validator(mode="after")
    def number_of_values_matches_points(self) -> Self:
        """Check that the number of data values matches the number of grid points."""
        num_values = len(self.values.index)
        num_points = len(self.points)

        if num_points != num_values:
            raise ValidationError(
                f"The number of data values ({num_values}) does not match the number of grid "
                f"points ({num_points})."
            )
        return self

    @field_validator("cells")
    @classmethod
    def match_cells_to_vtk_type(cls, val: CellDataArray) -> CellDataArray:
        """Check that cell connections does not have duplicate points."""
        if vtk is None:
            return val

        # using val.astype(np.int32/64) directly causes issues when dataarray are later checked ==
        return CellDataArray(val.data.astype(vtk["id_type"], copy=False), coords=val.coords)

    @field_validator("cells")
    @classmethod
    def cells_right_type(cls, val: CellDataArray) -> CellDataArray:
        """Check that cell are of the right type."""
        # only supporting the standard ordering of cell vertices 012(3)
        vertex_coords_expected = np.arange(cls._cell_num_vertices())
        vertex_coords_given = val.vertex_index.data
        if np.any(vertex_coords_given != vertex_coords_expected):
            raise ValidationError(
                f"Cell connections array is expected to have {vertex_coords_expected} coord values"
                f" along 'vertex_index' (given: {vertex_coords_given})."
            )
        return val

    @model_validator(mode="after")
    def check_cell_vertex_range(self) -> Self:
        """Check that cell connections use only defined points."""
        val = getattr(self, "cells", None)
        if val is None:
            return self
        all_point_indices_used = val.data.ravel()
        # skip validation if zero size data
        if len(all_point_indices_used) > 0:
            min_index_used = np.min(all_point_indices_used)
            max_index_used = np.max(all_point_indices_used)

            num_points = len(self.points)

            if max_index_used > num_points - 1 or min_index_used < 0:
                raise ValidationError(
                    "Cell connections array uses undefined point indices in the range "
                    f"[{min_index_used}, {max_index_used}]. The valid range of point indices is "
                    f"[0, {num_points - 1}]."
                )
        return self

    @field_validator("cells")
    @classmethod
    def warn_degenerate_cells(cls, val: CellDataArray) -> CellDataArray:
        """Check that cell connections does not have duplicate points."""
        degenerate_cells = cls._find_degenerate_cells(val)
        num_degenerate_cells = len(degenerate_cells)
        if num_degenerate_cells > 0:
            log.warning(
                f"Unstructured grid contains {num_degenerate_cells} degenerate cell(s). "
                "Such cells can be removed by using function "
                "'.clean(remove_degenerate_cells: bool = True, remove_unused_points: bool = True)'. "
                "For example, 'dataset = dataset.clean()'."
            )
        return val

    @model_validator(mode="before")
    @classmethod
    def _warn_if_none(cls, data: Any) -> Any:
        """Warn if any of data arrays are not loaded."""

        if not isinstance(data, dict):
            return data  # already validated

        no_data_fields = []
        for field_name in ["points", "cells", "values"]:
            field = data.get(field_name)
            if isinstance(field, str) and field in DATA_ARRAY_MAP.keys():
                no_data_fields.append(field_name)

        if len(no_data_fields) > 0:
            formatted_names = [f"'{fname}'" for fname in no_data_fields]
            log.warning(
                f"Loading {', '.join(formatted_names)} without data. Constructing an empty dataset."
            )
            data["points"] = PointDataArray(
                np.zeros((0, cls._point_dims())), dims=["index", "axis"]
            )
            data["cells"] = CellDataArray(
                np.zeros((0, cls._cell_num_vertices())), dims=["cell_index", "vertex_index"]
            )
            data["values"] = IndexedDataArray(np.zeros(0), dims=["index"])

        return data

    @model_validator(mode="before")
    @classmethod
    def _add_default_coords(cls, data: dict) -> dict:
        def _add_default_coords(da: DataArray) -> DataArray:
            """Add 0..N-1 coordinates to any dimension that does not already have one.
            Note: We use a pandas `RangeIndex` here for constant memory.
            """
            missing = {d: RangeIndex(da.sizes[d]) for d in da.dims if d not in da.coords}
            return da.assign_coords(missing) if missing else da

        if "points" in data:
            data["points"] = _add_default_coords(data["points"])
        if "cells" in data:
            data["cells"] = _add_default_coords(data["cells"])
        if "values" in data:
            data["values"] = _add_default_coords(data["values"])
        return data

    @model_validator(mode="after")
    def _warn_unused_points(self) -> Self:
        """Warn if some points are unused.

        Uses efficient NumPy boolean array instead of Python sets for O(n) performance.
        """
        if not _WARN_UNUSED_POINTS.get():
            return self

        num_points = len(self.points.data)
        cell_indices = self.cells.values.ravel()

        # Use boolean array: O(n) time and O(n) space, much faster than Python sets
        used = np.zeros(num_points, dtype=bool)
        # Clip to valid range to handle any out-of-bounds indices gracefully
        valid_indices = cell_indices[(cell_indices >= 0) & (cell_indices < num_points)]
        used[valid_indices] = True

        if not np.all(used):
            log.warning(
                "Unstructured grid dataset contains unused points. "
                "Consider calling 'clean()' to remove them."
            )

        return self

    """ Convenience properties """

    @property
    def name(self) -> str:
        """Dataset name."""
        # we redirect name to values.name
        return self.values.name

    def rename(self, name: str) -> UnstructuredDataset:
        """Return a renamed array."""
        return self.updated_copy(values=self.values.rename(name), deep=False)

    @property
    def is_complex(self) -> bool:
        """Data type."""
        return np.iscomplexobj(self.values)

    @property
    def _double_type(self) -> DTypeLike:
        """Corresponding double data type."""
        return np.complex128 if self.is_complex else np.float64

    @property
    def is_uniform(self) -> bool:
        """Whether each element is of equal value in ``values``."""
        return self.values.is_uniform

    @cached_property
    def _non_spatial_coords_dict(self) -> dict[str, Any]:
        """Non-spatial dimensions are corresponding coordinate values of stored data."""
        coord_dict = {dim: self.values.coords[dim].data for dim in self.values.dims}
        _ = coord_dict.pop("index")
        return coord_dict

    @cached_property
    def _non_spatial_dims(self) -> list[str]:
        """Non-spatial dimensions are corresponding coordinate values of stored data."""
        return [dim for dim in self.values.dims if dim != "index"]

    @cached_property
    def _non_spatial_shape(self) -> list[int]:
        """Shape in which fields are stored at each point."""
        return [len(coord) for coord in self._non_spatial_coords_dict.values()]

    @cached_property
    def _num_fields(self) -> int:
        """Total number of stored fields."""
        return 1 if len(self._non_spatial_shape) == 0 else np.prod(self._non_spatial_shape)

    @cached_property
    def _values_type(self) -> type:
        """Type of array storing values."""
        return type(self.values)

    @cached_property
    def bounds(self) -> Bound:
        """Grid bounds."""
        return tuple(np.min(self.points.data, axis=0)), tuple(np.max(self.points.data, axis=0))

    @cached_property
    @abstractmethod
    def _points_3d_array(self) -> None:
        """3D coordinates of grid points."""

    """ Grid cleaning """

    @classmethod
    def _find_degenerate_cells(cls, cells: CellDataArray) -> set[int]:
        """Find explicitly degenerate cells if any.
        That is, cells that use the same point indices for their different vertices.
        """
        indices = cells.data
        # skip validation if zero size data
        degenerate_cell_inds = set()
        if len(indices) > 0:
            for i in range(cls._cell_num_vertices() - 1):
                for j in range(i + 1, cls._cell_num_vertices()):
                    new_inds = np.where(indices[:, i] == indices[:, j])[0]
                    degenerate_cell_inds |= {int(k) for k in new_inds}

        return degenerate_cell_inds

    @classmethod
    def _remove_degenerate_cells(cls, cells: CellDataArray) -> CellDataArray:
        """Remove explicitly degenerate cells if any.
        That is, cells that use the same point indices for their different vertices.
        """
        degenerate_cells = cls._find_degenerate_cells(cells=cells)
        if len(degenerate_cells) > 0:
            data = np.delete(cells.values, list(degenerate_cells), axis=0)
            cell_index = np.delete(cells.cell_index.values, list(degenerate_cells))
            return CellDataArray(
                data=data, coords={"cell_index": cell_index, "vertex_index": cells.vertex_index}
            )
        return cells

    @classmethod
    def _remove_unused_points(
        cls, points: PointDataArray, values: IndexedDataArrayTypes, cells: CellDataArray
    ) -> tuple[PointDataArray, IndexedDataArrayTypes, CellDataArray]:
        """Remove unused points if any.
        That is, points that are not used in any grid cell.
        """

        used_indices = np.unique(cells.values.ravel())
        num_points = len(points)

        if len(used_indices) != num_points or np.any(np.diff(used_indices) != 1):
            min_index = np.min(used_indices)
            map_len = np.max(used_indices) - min_index + 1
            index_map = np.zeros(map_len)
            index_map[used_indices - min_index] = np.arange(len(used_indices))

            cells = CellDataArray(data=index_map[cells.data - min_index], coords=cells.coords)
            points = PointDataArray(points.data[used_indices, :], dims=["index", "axis"])
            values = values.sel(index=used_indices)
            if "index" in values.coords:
                # renumber if index given as a coordinate
                values["index"] = np.arange(len(used_indices))

        return points, values, cells

    def clean(
        self, remove_degenerate_cells: bool = True, remove_unused_points: bool = True
    ) -> Self:
        """Remove degenerate cells and/or unused points."""
        if remove_degenerate_cells:
            cells = self._remove_degenerate_cells(cells=self.cells)
        else:
            cells = self.cells

        if remove_unused_points:
            points, values, cells = self._remove_unused_points(self.points, self.values, cells)
        else:
            points = self.points
            values = self.values

        return self.updated_copy(points=points, values=values, cells=cells, deep=False)

    """ Arithmetic operations """

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Self | numbers.Number, **kwargs: Any
    ) -> Self | tuple[Self, ...] | None:
        """Override of numpy functions."""

        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with a scalar or an unstructured grid dataset of the same spatial dimensionality
            if not (
                isinstance(x, numbers.Number)
                or (isinstance(x, type(self)) and x._point_dims() == self._point_dims())
            ):
                raise Tidy3dNotImplementedError(
                    f"Cannot perform arithmetic operations between instances of different classes ({type(self)} and {type(x)})."
                )

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.values if isinstance(x, type(self)) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.values if isinstance(x, type(self)) else x for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(self.updated_copy(values=x, deep=False) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return self.updated_copy(values=result, deep=False)

    @property
    def real(self) -> Self:
        """Real part of dataset."""
        return self.updated_copy(values=self.values.real, deep=False)

    @property
    def imag(self) -> UnstructuredDataset:
        """Imaginary part of dataset."""
        return self.updated_copy(values=self.values.imag, deep=False)

    @property
    def abs(self) -> UnstructuredDataset:
        """Absolute value of dataset."""
        return self.updated_copy(values=self.values.abs, deep=False)

    def conj(self) -> UnstructuredDataset:
        """Complex conjugate value of dataset."""
        return self.updated_copy(values=self.values.conj())

    def norm(self, dim: str) -> UnstructuredDataset:
        """Compute vector norm along a given dimension."""
        return self.updated_copy(values=np.sqrt(self.values.dot(self.values.conj(), dim=dim).real))

    """ VTK interfacing """

    @classmethod
    @abstractmethod
    @requires_vtk
    def _vtk_cell_type(cls) -> None:
        """VTK cell type to use in the VTK representation."""

    @cached_property
    def _vtk_offsets(self) -> ArrayLike:
        """Offsets array to use in the VTK representation."""
        offsets = np.arange(len(self.cells) + 1) * self._cell_num_vertices()
        if vtk is None:
            return offsets

        return offsets.astype(vtk["id_type"], copy=False)

    @property
    @requires_vtk
    def _vtk_cells(self) -> vtkCellArray:
        """VTK cell array to use in the VTK representation."""
        cells = vtk["mod"].vtkCellArray()
        cells.SetData(
            vtk["numpy_to_vtkIdTypeArray"](self._vtk_offsets),
            vtk["numpy_to_vtkIdTypeArray"](self.cells.data.ravel()),
        )
        return cells

    @property
    @requires_vtk
    def _vtk_points(self) -> vtkPoints:
        """VTK point array to use in the VTK representation."""
        pts = vtk["mod"].vtkPoints()
        pts.SetData(vtk["numpy_to_vtk"](self._points_3d_array))
        return pts

    @property
    @requires_vtk
    def _vtk_obj_empty(self) -> vtkUnstructuredGrid:
        """A VTK representation (vtkUnstructuredGrid) of the grid."""

        grid = vtk["mod"].vtkUnstructuredGrid()

        grid.SetPoints(self._vtk_points)
        grid.SetCells(self._vtk_cell_type(), self._vtk_cells)

        return grid

    @property
    @requires_vtk
    def _vtk_obj(self) -> vtkUnstructuredGrid:
        """A VTK representation (vtkUnstructuredGrid) of the grid."""

        grid = self._vtk_obj_empty

        if self.is_complex:
            # vtk doesn't support complex numbers
            # so we will store our complex array as a two-component vtk array
            data_values = self.values.values.view("(2,)float")
        else:
            data_values = self.values.values

        if len(self._non_spatial_shape) > 0:
            data_values = data_values.reshape(
                (len(self.points.values), (1 + self.is_complex) * self._num_fields)
            )

        point_data_vtk = vtk["numpy_to_vtk"](data_values)
        point_data_vtk.SetName(self.name)
        grid.GetPointData().AddArray(point_data_vtk)

        return grid

    @staticmethod
    @requires_vtk
    def _read_vtkUnstructuredGrid(fname: PathLike) -> vtkUnstructuredGrid:
        """Load a :class:`vtkUnstructuredGrid` from a file."""
        fname = str(fname)
        reader = vtk["mod"].vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()
        grid = reader.GetOutput()

        return grid

    @staticmethod
    @requires_vtk
    def _read_vtkLegacyFile(fname: PathLike) -> vtkUnstructuredGrid:
        """Load a grid from a legacy `.vtk` file."""
        fname = str(fname)
        reader = vtk["mod"].vtkGenericDataObjectReader()
        reader.SetFileName(fname)
        reader.Update()
        grid = reader.GetOutput()

        return grid

    @classmethod
    def _construct_from_vtk_arrays(cls, warn_unused_points: bool = True, **data: Any) -> Self:
        """Construct a dataset while optionally suppressing trusted internal cleanup hints."""
        if warn_unused_points:
            return cls(**data)

        with _suppress_unstructured_grid_unused_point_warnings():
            return cls(**data)

    @classmethod
    def _cell_types_numpy(cls, vtk_obj: vtkUnstructuredGrid) -> np.ndarray:
        """Return per-cell VTK types, falling back when the consolidated array is unavailable."""
        cell_types_array = vtk_obj.GetCellTypesArray()
        if cell_types_array is not None:
            return np.array(vtk["vtk_to_numpy"](cell_types_array), copy=True)

        num_cells = vtk_obj.GetNumberOfCells()
        return np.fromiter(
            (vtk_obj.GetCellType(ind) for ind in range(num_cells)), dtype=int, count=num_cells
        )

    @classmethod
    @requires_vtk
    def _from_vtk_obj(
        cls,
        vtk_obj: vtkUnstructuredGrid,
        field: str | None = None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
        values_type: type = IndexedDataArray,
        expect_complex: bool | None = None,
        ignore_invalid_cells: bool = False,
        warn_unused_points: bool = True,
    ) -> UnstructuredDataset:
        """Initialize from a vtkUnstructuredGrid instance."""

        # read point, cells, and values info from a vtk instance
        cells_numpy = vtk["vtk_to_numpy"](vtk_obj.GetCells().GetConnectivityArray())
        points_numpy = vtk["vtk_to_numpy"](vtk_obj.GetPoints().GetData())
        values = cls._get_values_from_vtk(
            vtk_obj, len(points_numpy), field, values_type, expect_complex
        )

        # verify cell_types
        cells_types = cls._cell_types_numpy(vtk_obj)
        invalid_cells = cells_types != cls._vtk_cell_type()
        if any(invalid_cells):
            if ignore_invalid_cells:
                cell_offsets = vtk["vtk_to_numpy"](vtk_obj.GetCells().GetOffsetsArray())
                valid_cell_offsets = cell_offsets[:-1][invalid_cells == 0]
                cells_numpy = cells_numpy[
                    np.ravel(
                        valid_cell_offsets[:, None]
                        + np.arange(cls._cell_num_vertices(), dtype=int)[None, :]
                    )
                ]
            else:
                raise DataError(
                    f"Unsupported cell types found in 'vtkUnstructuredGrid' for '{cls.__name__}'."
                )

        # pack point and cell information into Tidy3D arrays
        num_cells = len(cells_numpy) // cls._cell_num_vertices()
        cells_numpy = np.reshape(cells_numpy, (num_cells, cls._cell_num_vertices()))

        cells = CellDataArray(
            cells_numpy,
            coords={
                "cell_index": np.arange(num_cells),
                "vertex_index": np.arange(cls._cell_num_vertices()),
            },
        )

        points = PointDataArray(
            points_numpy,
            coords={"index": np.arange(len(points_numpy)), "axis": np.arange(cls._point_dims())},
        )

        if remove_degenerate_cells:
            cells = cls._remove_degenerate_cells(cells=cells)

        if remove_unused_points:
            points, values, cells = cls._remove_unused_points(
                points=points, values=values, cells=cells
            )

        return cls._construct_from_vtk_arrays(
            warn_unused_points=warn_unused_points,
            points=points,
            cells=cells,
            values=values,
        )

    @requires_vtk
    def _from_vtk_obj_internal(
        self,
        vtk_obj: vtkUnstructuredGrid,
        remove_degenerate_cells: bool = True,
        remove_unused_points: bool = True,
    ) -> UnstructuredDataset:
        """Initialize from a vtk object when performing internal operations. When we do that we
        pass structure of possibly multidimensional nature of values through parametes field and
        values_type. We also turn on by default cleaning of geometry."""
        return self._from_vtk_obj(
            vtk_obj=vtk_obj,
            field=self._non_spatial_coords_dict,
            remove_degenerate_cells=remove_degenerate_cells,
            remove_unused_points=remove_unused_points,
            values_type=self._values_type,
            expect_complex=self.is_complex,
        )

    @classmethod
    @requires_vtk
    def from_vtu(
        cls,
        file: PathLike,
        field: str | None = None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
        ignore_invalid_cells: bool = False,
    ) -> UnstructuredDataset:
        """Load unstructured data from a vtu file.

        Parameters
        ----------
        file : PathLike
            Full path to the .vtu file to load the unstructured data from.
        field : str = None
            Name of the field to load.
        remove_degenerate_cells : bool = False
            Remove explicitly degenerate cells.
        remove_unused_points : bool = False
            Remove unused points.
        ignore_invalid_cells : bool = False
            Whether to ignore invalid cells during loading.

        Returns
        -------
        UnstructuredDataset
            Unstructured dataset.
        """
        grid = cls._read_vtkUnstructuredGrid(file)
        return cls._from_vtk_obj(
            grid,
            field=field,
            remove_degenerate_cells=remove_degenerate_cells,
            remove_unused_points=remove_unused_points,
            ignore_invalid_cells=ignore_invalid_cells,
        )

    @classmethod
    @requires_vtk
    def from_vtk(
        cls,
        file: PathLike,
        field: str | None = None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
        ignore_invalid_cells: bool = False,
    ) -> UnstructuredDataset:
        """Load unstructured data from a vtk file.

        Parameters
        ----------
        file : PathLike
            Full path to the .vtk file to load the unstructured data from.
        field : str = None
            Name of the field to load.
        remove_degenerate_cells : bool = False
            Remove explicitly degenerate cells.
        remove_unused_points : bool = False
            Remove unused points.
        remove_invalid_cells : bool = False
            Remove invalid cells.

        Returns
        -------
        UnstructuredDataset
            Unstructured data.
        """
        grid = cls._read_vtkLegacyFile(file)
        return cls._from_vtk_obj(
            grid,
            field=field,
            remove_degenerate_cells=remove_degenerate_cells,
            remove_unused_points=remove_unused_points,
            ignore_invalid_cells=ignore_invalid_cells,
        )

    @requires_vtk
    def to_vtu(self, fname: PathLike) -> None:
        """Exports unstructured grid data into a .vtu file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .vtu file to save the unstructured data to.
        """
        fname = str(fname)
        writer = vtk["mod"].vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self._vtk_obj)
        writer.Write()

    @classmethod
    @requires_vtk
    def _cell_to_point_data(
        cls,
        vtk_obj: vtkCellArray,
    ) -> vtkPointData:
        """Get point data values from a VTK object."""

        cellDataToPointData = vtk["mod"].vtkCellDataToPointData()
        cellDataToPointData.SetInputData(vtk_obj)
        cellDataToPointData.Update()

        return cellDataToPointData.GetOutput()

    @classmethod
    @requires_vtk
    def _get_values_from_vtk(
        cls,
        vtk_obj: vtkDataSet,
        num_points: PositiveInt,
        field: str | None = None,
        values_type: type = IndexedDataArray,
        expect_complex: bool | None = None,
    ) -> IndexedDataArray:
        """Get point data values from a VTK object."""

        point_data = vtk_obj.GetPointData()
        num_point_arrays = point_data.GetNumberOfArrays()

        if num_point_arrays == 0:
            log.warning(
                "No point data is found in a VTK object. '.values' will be initialized to zeros."
            )
            values_numpy = np.zeros(num_points)
            values_coords = {"index": np.arange(num_points)}
            values_name = None

        else:
            field_ind = field if isinstance(field, str) else 0

            array_vtk = point_data.GetAbstractArray(field_ind)
            # currently we assume data is real or complex scalar
            num_components = array_vtk.GetNumberOfComponents()
            if num_components > 2 and not isinstance(field, dict):
                raise DataError(
                    "Found point data array in a VTK object is expected to have maximum 2 "
                    "components (1 is for real data, 2 is for complex data). "
                    f"Found {num_components} components."
                )

            # check that number of values matches number of grid points
            num_tuples = array_vtk.GetNumberOfTuples()
            if num_tuples != num_points:
                raise DataError(
                    f"The length of found point data array ({num_tuples}) does not match the number"
                    f" of grid points ({num_points})."
                )

            # copy=True is required because vtk_to_numpy may return a view into VTK's
            # internal memory buffer, which can be invalidated when the VTK object is
            # modified or garbage collected, causing data corruption.
            values_numpy = np.array(vtk["vtk_to_numpy"](array_vtk), copy=True)
            values_name = array_vtk.GetName()

            # vtk doesn't support complex numbers
            # we store our complex array as a two-component vtk array
            # so here we convert that into a single component complex array
            if (num_components == 2 and expect_complex is None) or expect_complex is True:
                values_numpy = values_numpy.view("complex")

            new_shape = [num_points]
            if isinstance(field, dict):
                new_shape = new_shape + [len(coord) for coord in field.values()]

            values_numpy = np.reshape(values_numpy, new_shape)

            # currently we assume there is only one point data array provided in the VTK object
            if num_point_arrays > 1 and field is None:
                log.warning(
                    f"{num_point_arrays} point data arrays are found in a VTK object. "
                    f"Only the first array (name: {values_name}) will be used to initialize "
                    "'.values' while the rest will be ignored."
                )

            values_coords = {"index": np.arange(num_points)}
            if isinstance(field, dict):
                values_coords.update(field)

        values = values_type(values_numpy, coords=values_coords, name=values_name)

        return values

    def get_cell_values(self, **kwargs: Any) -> NDArray:
        """This function returns the cell values for the fields stored in the UnstructuredDataset.
        If multiple fields are stored per point, like in an IndexedVoltageDataArray, cell values
        will be provided for each of the fields unless a selection argument is provided, e.g., voltage=0.2
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the xarray sel() function.
        Returns
        -------
        numpy.ndarray
            Extracted data.
        """

        values = self.values.sel(**kwargs)

        return values[self.cells].mean(dim="vertex_index").values

    @abstractmethod
    def get_cell_volumes(self) -> DataArray:
        """Get the volumes/areas associated to each cell."""

    """ Grid operations """

    @requires_vtk
    def _plane_slice_raw(self, axis: Axis, pos: float) -> vtkPolyData:
        """Slice dataset with a plane and return the resulting VTK object."""

        if pos > self.bounds[1][axis] or pos < self.bounds[0][axis]:
            raise DataError(
                f"Slicing plane (axis: {axis}, pos: {pos}) does not intersect the unstructured grid "
                f"(extent along axis {axis}: {self.bounds[0][axis]}, {self.bounds[1][axis]})."
            )

        origin = [0, 0, 0]
        origin[axis] = pos

        normal = [0, 0, 0]
        # orientation of normal is important for edge (literally) cases
        normal[axis] = -1
        if pos > (self.bounds[0][axis] + self.bounds[1][axis]) / 2:
            normal[axis] = 1

        # create cutting plane
        plane = vtk["mod"].vtkPlane()
        plane.SetOrigin(origin[0], origin[1], origin[2])
        plane.SetNormal(normal[0], normal[1], normal[2])

        # cut points coincide wherever cells share an edge, whether or not the cutter merged
        # them itself. Tagging each cell with its own index -- cell data passes through the
        # cutter untouched -- is what lets the merge below tell those apart from the two sides
        # of a material interface, which are also coincident but not the same node. See
        # '_merge_coincident_cut_points'.
        grid = self._vtk_obj
        source_cells = vtk["numpy_to_vtk"](
            np.arange(grid.GetNumberOfCells(), dtype=np.int64), deep=True
        )
        source_cells.SetName(SOURCE_CELL_ARRAY_NAME)
        grid.GetCellData().AddArray(source_cells)

        # create cutter
        cutter = vtk["mod"].vtkPlaneCutter()
        cutter.SetPlane(plane)
        cutter.SetInputData(grid)
        cutter.InterpolateAttributesOn()
        # the dataset this cut becomes holds triangles only, and some VTK versions cut a
        # cell into a polygon unless asked otherwise
        cutter.GeneratePolygonsOff()
        # merging by coordinate folds the two sides of a material interface into one point
        # before the provenance below can tell them apart, and the survivor is then alone at
        # its coordinate, so nothing downstream can even detect the loss. Off is the default
        # here, but the cut is not left resting on it. A version that merges anyway is caught
        # by '_check_shared_cut_point_sources' rather than trusted.
        cutter.MergePointsOff()
        cutter.Update()

        merged = self._merge_coincident_cut_points(cutter.GetOutput(), axis=axis, pos=pos)
        self._warn_if_slice_drops_a_side(merged, axis=axis, pos=pos)

        return merged

    @requires_vtk
    def _merge_coincident_cut_points(self, cut: vtkPolyData, axis: Axis, pos: float) -> vtkPolyData:
        """Collapse cut points that are the same mesh node, keeping genuine jumps apart.

        Replaces ``vtkCleanPolyData``, which merges by coordinate alone. Where the grid
        carries per-zone duplicate nodes -- how a heterojunction holds its band offset, or
        a contact resistance its temperature step -- the cuts of the cells on either side
        land on one coordinate with different values, and merging by geometry keeps an
        arbitrary one of them.

        Two coincident cut points are the same node exactly when the mesh edges they were
        interpolated from share an original point. Cells on one side of an interface share
        the edge they both cut; cells across it share nothing, because their nodes are
        duplicated. That test needs no tolerance, and it also merges the several edges
        meeting at a vertex when the plane passes through one.
        """

        num_points = cut.GetNumberOfPoints()
        if num_points == 0:
            return cut

        points = np.array(vtk["vtk_to_numpy"](cut.GetPoints().GetData()), copy=True)

        # group by exact coordinate, as 'vtkCleanPolyData' does with its default tolerance.
        # Labelled through a lexicographic sort rather than 'np.unique(points, axis=0)',
        # which reads each row as raw bytes through a void view -- that both copies the
        # array and separates values that compare equal, -0.0 from 0.0 among them.
        order = np.lexsort(points.T)
        ordered = points[order]
        opens_group = np.empty(num_points, dtype=bool)
        opens_group[0] = True
        np.any(ordered[1:] != ordered[:-1], axis=1, out=opens_group[1:])
        group_of_point = np.empty(num_points, dtype=np.int64)
        group_of_point[order] = np.cumsum(opens_group) - 1
        del order, ordered, opens_group

        # a point alone at its coordinate has nothing to merge with, so the source-edge
        # search -- which holds a coordinate tensor per candidate cell edge, this slice's
        # dominant temporary -- is asked about the rest only
        coincident = np.bincount(group_of_point)[group_of_point] > 1

        # the cutter is asked not to merge its points, so a point listed by several cut cells
        # means it merged anyway -- and such a point is alone at its coordinate, because the
        # merge is what removed its twin. 'coincident' would therefore step over exactly the
        # points whose provenance is in doubt, so they are carried in as well and the check
        # below sees them. With merging off none exist and this is 'coincident' unchanged.
        shared = np.bincount(_cut_cell_point_pairs(cut)[0], minlength=num_points) > 1
        subset = coincident | shared

        source_edges = self._cut_point_source_edges(cut, points, subset, axis, pos)
        representative = _merge_representative_points(
            group_of_point, source_edges, subset, num_points
        )

        kept = np.unique(representative)
        remap = np.zeros(num_points, dtype=np.int64)
        remap[kept] = np.arange(len(kept))
        return _rebuild_polydata(cut, points[kept], remap[representative], kept)

    @requires_vtk
    def _cut_point_source_edges(
        self, cut: vtkPolyData, cut_points: NDArray, subset: NDArray, axis: Axis, pos: float
    ) -> NDArray:
        """Original point indices of the mesh edge each selected cut point sits on.

        ``subset`` is a boolean mask over ``cut_points``; the result is one row per selected
        point, in point order. A point that landed on a vertex is on several edges at once
        and gets an arbitrary one of them; the vertex is an endpoint of each, which is all
        the merge compares.
        """

        point_entries, source_entries = _cut_cell_point_pairs(cut)

        # A point listed by several cut cells is resolved once, not once per use: the edge
        # under every (point, cell) incidence of the subset is found in a single pass, which
        # is then both what the agreement check reads and what the merge reduces.
        selected = subset[point_entries]
        point, source = point_entries[selected], source_entries[selected]
        del point_entries, source_entries, selected
        edges = self._mesh_edges_of_cut_points(cut_points[point], source, axis, pos)

        self._check_shared_cut_point_sources(point, edges)

        # one row per selected point, in point order. The incidences of a shared point name
        # one edge between them -- the check above is exactly that -- so the last write wins
        # without the choice mattering.
        found = np.zeros((np.count_nonzero(subset), 2), dtype=edges.dtype)
        row_of_point = np.cumsum(subset) - 1
        found[row_of_point[point]] = edges
        return found

    def _mesh_edges_of_cut_points(
        self, cut_points: NDArray, source_cell: NDArray, axis: Axis, pos: float
    ) -> NDArray:
        """Mesh edge under each cut point, given the cell it was cut from, as ``(N, 2)``.

        The cut point sits on the one edge of its cell that the plane crosses beneath it.
        Where an edge crosses is fixed by its two endpoints' ``axis`` coordinates alone, so
        the search reads that single column, places the crossing along the edge, and then
        compares only the two in-plane coordinates -- every candidate already agrees with the
        cut point on the third, which is ``pos``. Edges the plane does not reach are excluded
        outright rather than measured, which is also what keeps an edge parallel to ``axis``
        from matching on its in-plane coordinates alone.

        Walked in chunks: these arrays are the slice's dominant temporaries, and a chunk
        bounds them at a fixed cost while leaving the result untouched.
        """

        cell_vertices = self.cells.data
        points_3d = np.asarray(self._points_3d_array)
        cell_edges = self._cell_edges()
        # the cutter works in 3D, so read the 3D form of the grid's points -- a triangular
        # grid stores only its two in-plane columns
        in_plane = [other for other in range(3) if other != axis]
        along = points_3d[:, axis]

        found = np.empty((len(cut_points), 2), dtype=cell_vertices.dtype)
        for start in range(0, len(cut_points), CUT_POINT_EDGE_SEARCH_CHUNK):
            rows = slice(start, start + CUT_POINT_EDGE_SEARCH_CHUNK)
            # (rows, num_edges, 2) original point indices of the candidate edges
            edges = cell_vertices[source_cell[rows]][:, cell_edges]
            behind, ahead = edges[:, :, 0], edges[:, :, 1]

            low, high = along[behind], along[ahead]
            span = high - low
            reaches_plane = (np.minimum(low, high) <= pos) & (pos <= np.maximum(low, high))

            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(span != 0, (pos - low) / span, 0.0)
            np.clip(frac, 0.0, 1.0, out=frac)

            # squared distance from the cut point to where the plane meets each candidate,
            # over the coordinates that can still differ. The two are accumulated one at a
            # time, so nothing wider than '(rows, num_edges)' is ever held and the grid's
            # points are read through a column view rather than gathered into a copy.
            residual = np.zeros_like(frac)
            for coordinate in in_plane:
                column = points_3d[:, coordinate]
                crossing = column[behind]
                offsets = column[ahead] - crossing
                offsets *= frac
                offsets += crossing
                offsets -= cut_points[rows, coordinate][:, None]
                # a square root would not move the minimum
                offsets *= offsets
                residual += offsets

            # the cut point lies on exactly one edge of its cell; pick that edge
            on_edge = np.argmin(np.where(reaches_plane, residual, np.inf), axis=1)
            found[rows] = edges[np.arange(len(on_edge)), on_edge]

        return found

    def _check_shared_cut_point_sources(self, point: NDArray, edges: NDArray) -> None:
        """Check that the cut cells sharing a cut point place it on a common mesh node.

        The merge keeps one incidence's edge and discards the rest, which is sound exactly
        when they all name a node in common. A point interior to an edge is named by that one
        edge; a point on a vertex is named by each of the edges meeting there, whose common
        endpoint is the vertex -- and the endpoints are all the merge goes on to compare. So
        the edges need not be equal, which they are not once the cutter merges points itself.

        ``point`` and ``edges`` describe the (point, cell) incidences of the merged subset
        alone -- the points whose edge is actually read -- so a cut whose points are all
        unshared, or all alone at their coordinate, pays nothing beyond the pass the merge
        already needed.
        """

        if len(point) == 0:
            return

        order = np.argsort(point, kind="stable")
        point, edge = point[order], edges[order]
        first = np.flatnonzero(np.concatenate(([True], point[1:] != point[:-1])))
        counts = np.diff(np.concatenate((first, [len(point)])))

        def every_incidence_touches(side: int) -> NDArray:
            """Whether every incidence of each point names the first one's ``side`` node."""
            node = np.repeat(edge[first, side], counts)
            return np.logical_and.reduceat((edge[:, 0] == node) | (edge[:, 1] == node), first)

        # a node common to every incidence has to be one of the first incidence's two
        if not np.all(every_incidence_touches(0) | every_incidence_touches(1)):
            raise DataError(
                "Cut cells sharing a cut point place it on mesh edges with no node in common, "
                "so the node behind that point is ambiguous and coincident cut points cannot "
                "be merged reliably. The cut asks 'vtkPlaneCutter' not to merge its points, so "
                "a shared point means the installed VTK merged them despite that, folding the "
                "two sides of a material interface into one; slicing across a material "
                "interface is not supported on this VTK."
            )

    @requires_vtk
    def _warn_if_slice_drops_a_side(self, cut: vtkPolyData, axis: Axis, pos: float) -> None:
        """Warn when the slice is coplanar with per-zone duplicate nodes.

        A plane lying exactly on a material interface crosses no cell there: it grazes the
        coplanar faces of both sides, and the cutter keeps whichever the plane normal
        favours -- which depends on where the interface sits inside the grid's bounds. The
        result then reports one material with no way to tell which.

        Detected by comparing what survived: a duplicate node group the plane merely
        crosses keeps one cut point per side, while a coplanar one comes back short.
        Nothing downstream can recover the missing side, and the grid alone cannot say
        which side the caller wanted, so this reports the situation rather than guessing.
        """

        points_3d = np.asarray(self._points_3d_array)
        in_plane = points_3d[:, axis] == pos
        if np.count_nonzero(in_plane) < 2:
            return

        duplicated, counts = np.unique(points_3d[in_plane], axis=0, return_counts=True)
        duplicated = duplicated[counts > 1]
        if len(duplicated) == 0:
            return

        cut_points = (
            np.array(vtk["vtk_to_numpy"](cut.GetPoints().GetData()), copy=True)
            if cut.GetNumberOfPoints() > 0
            else np.zeros((0, 3))
        )
        survivors, survivor_counts = (
            np.unique(cut_points, axis=0, return_counts=True)
            if len(cut_points) > 0
            else (np.zeros((0, 3)), np.zeros(0, dtype=int))
        )

        # one pass instead of a survivor rescan per node: label both sets against their
        # common unique rows, then read each duplicated node's survivor count by index
        rows, labels = np.unique(
            np.concatenate([duplicated, survivors]), axis=0, return_inverse=True
        )
        labels = labels.ravel()
        found = np.zeros(len(rows), dtype=np.int64)
        found[labels[len(duplicated) :]] = survivor_counts
        if np.any(found[labels[: len(duplicated)]] < counts[counts > 1]):
            log.warning(
                f"The slice at {'xyz'[axis]} = {pos} lies along a material interface, where "
                "the grid holds one node per side. Only one side's cells are cut there, so "
                "the result reports a single material and which one is not predictable. "
                "Offset the slice into the zone you want."
            )

    @abstractmethod
    @requires_vtk
    def plane_slice(self, axis: Axis, pos: float) -> XrDataArray | UnstructuredDataset:
        """Slice dataset with a plane and return the Tidy3D representation of the result
        (``UnstructuredDataset``).

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.

        Returns
        -------
        Union[xarray.DataArray, UnstructuredDataset]
            The resulting slice.
        """

    @requires_vtk
    def box_clip(self, bounds: Bound) -> UnstructuredDataset:
        """Clip the unstructured dataset using a box defined by ``bounds``.

        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        UnstructuredDataset
            Clipped dataset.
        """

        # make and run a VTK clipper
        clipper = vtk["mod"].vtkBoxClipDataSet()
        clipper.SetOrientation(0)
        clipper.SetBoxClip(
            bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1], bounds[0][2], bounds[1][2]
        )
        clipper.SetInputData(self._vtk_obj)
        clipper.GenerateClipScalarsOn()
        clipper.GenerateClippedOutputOff()
        clipper.Update()
        clip = clipper.GetOutput()

        # clean grid from unused points
        grid_cleaner = vtk["mod"].vtkRemoveUnusedPoints()
        grid_cleaner.SetInputData(clip)
        grid_cleaner.GenerateOriginalPointIdsOff()
        grid_cleaner.Update()
        clean_clip = grid_cleaner.GetOutput()

        # no intersection check
        if clean_clip.GetNumberOfPoints() == 0:
            raise DataError("Clipping box does not intersect the unstructured grid.")

        return self._from_vtk_obj_internal(clean_clip)

    @cached_property
    @requires_vtk
    def _boundary_points_indices(self) -> NDArray:
        """Find points that lie on open edges/faces."""

        surface_filter = vtk["mod"].vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(self._vtk_obj_empty)
        surface_filter.PassThroughPointIdsOn()  # Important for getting original indices
        surface_filter.Update()

        boundary_edges_vtk = surface_filter.GetOutput()

        if self._cell_num_vertices() == 3:
            feature_edges = vtk["mod"].vtkFeatureEdges()
            feature_edges.SetInputData(boundary_edges_vtk)

            # Enable the extraction of boundary edges
            feature_edges.BoundaryEdgesOn()

            # Disable other types of edges to get only the boundary
            feature_edges.FeatureEdgesOff()
            feature_edges.ManifoldEdgesOff()
            feature_edges.NonManifoldEdgesOff()

            feature_edges.Update()
            boundary_edges_vtk = feature_edges.GetOutput()

        if boundary_edges_vtk.GetNumberOfCells() == 0:
            # Mesh is watertight, no boundary
            return np.array([], dtype=int)

        # Get the original point indices
        original_ids_array = vtk["vtk_to_numpy"](
            boundary_edges_vtk.GetPointData().GetArray("vtkOriginalPointIds")
        )
        boundary_point_indices = original_ids_array.copy()
        return boundary_point_indices.astype(int)

    @requires_vtk
    def reflect(
        self,
        axis: Axis,
        center: float,
        reflection_only: bool = False,
        symmetry: Literal[-1, 1] | XrDataArray = 1,
    ) -> UnstructuredDataset:
        """Reflect unstructured dataset across the plane define by parameters ``axis`` and ``center``.
        By default the original dataset is preserved, setting ``reflection_only`` to ``True`` will
        produce only reflected dataset.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected dataset.
        symmetry : Union[Literal[-1, 1], XrDataArray] = 1
            Symmetry of the reflected field.

        Returns
        -------
        UnstructuredDataset
            Dataset after reflextion is performed.
        """
        # validate that if symmetry is an xarray, its dims are a subset of the values dims
        # and coords values coincide along those dims
        if isinstance(symmetry, XrDataArray):
            value_dims = set(self.values.dims) - {"index"}
            sym_dims = set(symmetry.dims)
            if not sym_dims.issubset(value_dims):
                raise DataError(
                    f"Symmetry xarray dimensions {sym_dims} must be a subset of values dimensions {value_dims}"
                )
            # Check that coordinates match along shared dimensions
            for dim in sym_dims:
                if not np.array_equal(symmetry.coords[dim], self.values.coords[dim]):
                    raise DataError(
                        f"Coordinate values for dimension '{dim}' must match between symmetry and values"
                    )

        if reflection_only:
            reflected_points = self.points.copy()
            reflected_points.loc[{"axis": axis}] = 2 * center - self.points.sel(axis=axis)
            return self.updated_copy(points=reflected_points, values=self.values * symmetry)

        # record number of existing points
        num_points = len(self.points)

        # detect points that are not on the reflection plane and on open edges
        # Those will need to be duplicated
        points_off_plane_map = np.ones(len(self.points), dtype=bool)
        points_off_plane_map[self._boundary_points_indices] = ~np.isclose(
            self.points.sel(axis=axis).data[self._boundary_points_indices],
            center,
            atol=fp_eps,
            rtol=fp_eps,
        )
        num_new_points = np.sum(points_off_plane_map)

        if num_new_points == 0:
            return self.updated_copy()

        # create new points id
        new_points_id = np.arange(num_new_points) + num_points

        new_points_id_map = -1 * np.ones_like(points_off_plane_map, dtype=int)
        new_points_id_map[points_off_plane_map] = new_points_id

        new_points = self.points.sel(index=points_off_plane_map).copy()
        new_points.loc[{"axis": axis}] = 2 * center - new_points.sel(axis=axis)

        # create new cells: only reflect cells that have at least one off-plane
        # vertex; cells with all vertices on the plane are shared and must not
        # be duplicated.
        cells_off_plane = np.any(points_off_plane_map[self.cells.data], axis=-1)
        new_cells = self.cells.sel(cell_index=cells_off_plane).copy()
        new_points_in_new_cells_map = points_off_plane_map[new_cells]
        new_points_in_new_cells_orig_id = new_cells.data[new_points_in_new_cells_map]
        new_cells.data[new_points_in_new_cells_map] = new_points_id_map[
            new_points_in_new_cells_orig_id
        ]

        # create new values
        new_values = self.values.sel(index=points_off_plane_map).copy() * symmetry

        # combine with original data
        combined_points = xr_concat([self.points, new_points], dim="index")
        combined_values = xr_concat([self.values, new_values], dim="index")
        combined_cells = xr_concat([self.cells, new_cells], dim="cell_index")

        combined_points.coords["index"] = np.arange(len(combined_points))
        combined_values.coords["index"] = np.arange(len(combined_values))
        combined_cells.coords["cell_index"] = np.arange(len(combined_cells))

        return self.updated_copy(
            points=combined_points, cells=combined_cells, values=combined_values
        )

    """ Data selection """

    @abstractmethod
    def sel(
        self,
        x: float | ArrayLike = None,
        y: float | ArrayLike = None,
        z: float | ArrayLike = None,
        method: Literal["None", "nearest", "pad", "ffill", "backfill", "bfill"] | None = None,
        **sel_kwargs: Any,
    ) -> UnstructuredDataset | XrDataArray:
        """Extract/interpolate data along one or more spatial or non-spatial directions. Must provide at least one argument
        among 'x', 'y', 'z' or non-spatial dimensions through additional arguments. Along spatial dimensions a suitable slicing of
        grid is applied (plane slice, line slice, or interpolation). Selection along non-spatial dimensions is forwarded to
        .sel() xarray function. Parameter 'method' applies only to non-spatial dimensions.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinate of the slice.
        y : Union[float, ArrayLike] = None
            y-coordinate of the slice.
        z : Union[float, ArrayLike] = None
            z-coordinate of the slice.
        method: Literal[None, "nearest", "pad", "ffill", "backfill", "bfill"] = None
            Method to use in xarray sel() function.
        **sel_kwargs : dict
            Keyword arguments to pass to the xarray sel() function.

        Returns
        -------
        Union[TriangularGridDataset, xarray.DataArray]
            Extracted data.
        """

    def _non_spatial_sel(
        self,
        method: Any = None,
        **sel_kwargs: Any,
    ) -> XrDataArray:
        """Select/interpolate data along one or more non-Cartesian directions.

        Parameters
        ----------
        **sel_kwargs : dict
            Keyword arguments to pass to the xarray sel() function.

        Returns
        -------
        xarray.DataArray
            Extracted data.
        """

        if "index" in sel_kwargs.keys():
            raise DataError("Cannot select along dimension 'index'.")

        sel_kwargs_only_lists = {
            key: value if key not in self._non_spatial_dims else _as_sequence_selector(value)
            for key, value in sel_kwargs.items()
        }
        return self.updated_copy(
            values=self.values.sel(**sel_kwargs_only_lists, method=method),
            deep=False,
        )

    def isel(
        self,
        **sel_kwargs: Any,
    ) -> XrDataArray:
        """Select data along one or more non-Cartesian directions by coordinate index.

        Parameters
        ----------
        drop : bool = False
            Drop the selected dimension instead of keeping it at length 1. Dropping every
            non-spatial dimension is what lets a later ``interp()`` return a
            :class:`.SpatialDataArray` again.
        **sel_kwargs : dict
            Keyword arguments to pass to the xarray isel() function.

        Returns
        -------
        xarray.DataArray
            Extracted data.
        """

        if "index" in sel_kwargs.keys():
            raise DataError("Cannot select along dimension 'index'.")

        # extract ``drop`` so it isn't treated as a dimension selector
        drop = sel_kwargs.pop("drop", False)

        # a length-1 sequence selector would stop xarray honouring ``drop=True``, so
        # only normalize when the caller is keeping the dimension
        if drop:
            sel_kwargs_processed = sel_kwargs
        else:
            sel_kwargs_processed = {
                key: value if key not in self._non_spatial_dims else _as_sequence_selector(value)
                for key, value in sel_kwargs.items()
            }
        return self.updated_copy(
            values=self.values.isel(**sel_kwargs_processed, drop=drop),
            deep=False,
        )

    """ Interpolation """

    def interp(
        self,
        x: float | ArrayLike = None,
        y: float | ArrayLike = None,
        z: float | ArrayLike = None,
        fill_value: float
        | Literal["extrapolate"]
        | None = "extrapolate",  # TODO: an array if multiple fields?
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
        **coords_kwargs: Any,
    ) -> XrDataArray:
        """Interpolate data along spatial dimensions x, y, and z and/or non-spatial dimensions.
        For spatial sampling points must provide all x, y, and z.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinates of sampling points.
        y : Union[float, ArrayLike] = None
            y-coordinates of sampling points.
        z : Union[float, ArrayLike] = None
            z-coordinates of sampling points.
        fill_value : Union[float, Literal["extrapolate"], None] = "extrapolate"
            Value to use when filling points without interpolated values. If ``"extrapolate"`` then
            nearest values are used. Passing ``None`` is equivalent to ``"extrapolate"``.
        use_vtk : bool = False
            Use vtk's interpolation functionality or Tidy3D's own implementation. Note: this
            option will be removed in a future version.
        method: Literal["linear", "nearest"] = "linear"
            Interpolation method to use.
        max_samples_per_step : int = 1e4
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int = 1e4
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float = 1e-6
            Relative tolerance when determining whether a point belongs to a cell.
        **coords_kwargs : dict
            Keyword arguments specifying non-spatial coordinates for interpolation (e.g., t=[0, 1, 2]).

        Returns
        -------
        xarray.DataArray
            Interpolated data.
        """

        # Treat None as "extrapolate" for backward compatibility
        if fill_value is None:
            fill_value = "extrapolate"

        spatial_dims_given = any(comp is not None for comp in [x, y, z])

        if spatial_dims_given:
            if any(comp is None for comp in [x, y, z]):
                raise DataError("Must provide either all or none of 'x', 'y', and 'z'")

        if not spatial_dims_given and len(coords_kwargs) == 0:
            raise DataError(
                "Must provide either 'x', 'y', and 'z' or points along other non-spatial dimensions."
            )

        result = self
        if len(coords_kwargs) > 0:
            result = result._non_spatial_interp(
                method=method, fill_value=fill_value, **coords_kwargs
            )

        if spatial_dims_given:
            result = result._spatial_interp(
                x=x,
                y=y,
                z=z,
                fill_value=fill_value,
                use_vtk=use_vtk,
                method=method,
                max_samples_per_step=max_samples_per_step,
                max_cells_per_step=max_cells_per_step,
                rel_tol=rel_tol,
            )

        return result

    def _non_spatial_interp(
        self,
        method: Literal["linear", "nearest"] = "linear",
        fill_value: float | Literal["extrapolate"] = np.nan,
        **coords_kwargs: Any,
    ) -> Self:
        """Interpolate data at non-spatial dimensions using xarray's interp() function.

        Parameters
        ----------
        method: Literal["linear", "nearest"] = "linear"
            Interpolation method to use.
        fill_value : Union[float, Literal["extrapolate"]] = 0
            Value to use when filling points without interpolated values. If ``"extrapolate"`` then
            nearest values are used. Note: in a future version the default value will be changed
            to ``"extrapolate"``.
        **coords_kwargs : dict
            Keyword arguments to pass to the xarray interp() function.

        Returns
        -------
        UnstructuredDataset
            Dataset with interpolated values.
        """
        if fill_value is None:
            fill_value = np.nan if method == "linear" else "extrapolate"

        # every key here is non-spatial by construction, so no dimension guard is needed
        coords_kwargs_only_lists = {
            key: _as_sequence_selector(value) for key, value in coords_kwargs.items()
        }

        interp_kwargs = {"method": method, "kwargs": {"fill_value": fill_value}}

        return self.updated_copy(
            values=self.values.interp(**coords_kwargs_only_lists, **interp_kwargs),
            deep=False,
        )

    @abstractmethod
    def _spatial_interp(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        z: float | ArrayLike,
        fill_value: float
        | Literal["extrapolate"]
        | None = None,  # TODO: an array if multiple fields?
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
    ) -> XrDataArray:
        """Interpolate data along spatial dimensions at provided x, y, and z.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : Union[float, Literal["extrapolate"]] = 0
            Value to use when filling points without interpolated values. If ``"extrapolate"`` then
            nearest values are used. Note: in a future version the default value will be changed
            to ``"extrapolate"``.
        use_vtk : bool = False
            Use vtk's interpolation functionality or Tidy3D's own implementation. Note: this
            option will be removed in a future version.
        method: Literal["linear", "nearest"] = "linear"
            Interpolation method to use.
        max_samples_per_step : int = 1e4
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int = 1e4
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float = 1e-6
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        xarray.DataArray
            Interpolated data.
        """


class UnstructuredGridDataset(UnstructuredDataset, ABC):
    """Abstract base for datasets that store unstructured grid data."""

    """ Interpolation """

    def _clamp_sampling_steps(self, steps: NDArray, dl: float | ArrayLike) -> NDArray:
        """Raise ``steps`` until the target grid holds no more points than the budget allows.

        An unstructured grid has no per-axis resolution to compare a request against, so the
        budget is the point count: asking for a spacing the mesh cannot support would invent
        detail and, since the sampled array is allocated dense, can ask for hundreds of
        gigabytes on a device-scale mesh. A mesh larger than the global sample cap is held to
        that cap instead, so an over-fine request is coarsened here rather than rejected by
        :func:`_grid_from_bounds` naming arguments ``downsample`` does not take.
        """
        rmin, rmax = self.bounds
        extents = np.array(rmax, dtype=float) - np.array(rmin, dtype=float)
        varying = extents > 0
        if not np.any(varying):
            return steps

        budget = min(len(self.points), MAX_SPATIAL_SAMPLES)
        raised = _clamp_steps_to_budget(steps[varying], extents[varying], budget=budget)
        if raised is None:
            return steps

        clamped = np.array(steps, dtype=float)
        clamped[varying], counts = raised
        log.warning(
            f"The requested sampling of '{dl}' is finer than this {len(self.points)}-point "
            f"grid supports; it has been clamped to {np.array2string(clamped, precision=4)}, "
            f"for a {'x'.join(str(int(count)) for count in counts)} grid. Specify a larger "
            "down-sampling step to suppress this warning.",
            log_once=True,
        )
        return clamped

    def downsample(
        self, dl: float | ArrayLike, method: Literal["linear", "nearest"] = "linear"
    ) -> SpatialDataArray:
        """Resample onto a uniform Cartesian grid spanning the same bounds, at spacing ``dl``.

        Unstructured charge and heat fields are typically far finer than the optical simulation
        that consumes them; sampling onto a coarse Cartesian grid first keeps
        :meth:`Simulation.perturbed_mediums_copy <tidy3d.Simulation.perturbed_mediums_copy>`
        from embedding a multi-hundred-megabyte custom medium in every simulation. Sampling is at
        the grid nodes, not averaged over cells, so it does not conserve the integral of the
        field: a feature thinner than ``dl`` is erased or, if a node lands on it, widened to
        ``dl``. Check the result before relying on it.

        A node outside the mesh -- in the empty quadrant of an L-shaped device, or anywhere else
        the cells do not fill the bounding box -- takes its nearest value, as sampling the same
        dataset onto a simulation grid does. This differs from
        :meth:`HeatChargeMonitorData.to_spatial_data_array
        <tidy3d.HeatChargeMonitorData.to_spatial_data_array>`, which fills such points with zero
        because a source term has no contribution where nothing was solved.

        The grid spans the original bounds exactly, so the spacing it lands on is ``dl`` rounded
        down to the nearest whole number of steps across each axis, never up. Only a degenerate
        axis reduces to a single point, so a planar dataset stays planar while every other axis
        keeps both its endpoints. A ``dl`` finer than the grid can support is raised until the
        result holds no more points than the grid itself, with a warning.

        Parameters
        ----------
        dl : Union[float, ArrayLike]
            Target grid spacing (micron), either isotropic or one value per ``x``, ``y``, ``z``.
        method : Literal["linear", "nearest"] = "linear"
            Interpolation method used to sample onto the new grid.

        Returns
        -------
        :class:`.SpatialDataArray`
            Data sampled on the coarsened Cartesian grid.
        """
        if self._num_fields > 1:
            raise DataError(
                "Cannot down-sample a dataset containing multiple field values. Please select "
                "one before calling this function. This can be done with, e.g., "
                f"'.sel({self._non_spatial_dims[0]}=...)'."
            )

        steps = self._clamp_sampling_steps(_grid_steps(dl), dl=dl)
        x, y, z = _grid_from_bounds(bounds=self.bounds, resolution=steps)
        result = self.interp(x=x, y=y, z=z, method=method)

        # '.sel()' leaves the selected coordinate behind as a singleton dim; drop it so the
        # result is a plain 'SpatialDataArray' rather than an untyped 'xarray.DataArray'
        extra_dims = [dim for dim in result.dims if dim not in ("x", "y", "z")]
        if extra_dims:
            result = result.isel(dict.fromkeys(extra_dims, 0), drop=True)
            result = SpatialDataArray(result.transpose("x", "y", "z"), name=self.values.name)

        return result

    def _spatial_interp(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        z: float | ArrayLike,
        fill_value: float
        | Literal["extrapolate"] = "extrapolate",  # TODO: an array if multiple fields?
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
    ) -> XrDataArray:
        """Interpolate data along spatial dimensions at provided x, y, and z.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : Union[float, Literal["extrapolate"]] = "extrapolate"
            Value to use when filling points without interpolated values. If ``"extrapolate"`` then
            nearest values are used.
        use_vtk : bool = False
            Use vtk's interpolation functionality or Tidy3D's own implementation. Note: this
            option will be removed in a future version.
        method: Literal["linear", "nearest"] = "linear"
            Interpolation method to use.
        max_samples_per_step : int = 1e4
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int = 1e4
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float = 1e-6
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        xarray.DataArray
            Interpolated data.
        """

        # calculate the resulting array shape
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        if method == "nearest":
            interpolated_values = self._interp_nearest(x=x, y=y, z=z)
        else:
            if fill_value == "extrapolate":
                fill_value_actual = np.nan
            else:
                fill_value_actual = fill_value

            if use_vtk:
                if self.is_complex:
                    raise DataError("Option 'use_vtk=True' is not supported for complex datasets.")
                if len(self._non_spatial_shape) > 0:
                    raise DataError(
                        "Option 'use_vtk=True' is not supported for multidimensional datasets."
                    )
                log.warning("Note that option 'use_vtk=True' will be removed in future versions.")
                interpolated_values = self._interp_vtk(x=x, y=y, z=z, fill_value=fill_value_actual)
            else:
                interpolated_values = self._interp_py(
                    x=x,
                    y=y,
                    z=z,
                    fill_value=fill_value_actual,
                    max_samples_per_step=max_samples_per_step,
                    max_cells_per_step=max_cells_per_step,
                    rel_tol=rel_tol,
                )

            if fill_value == "extrapolate" and method != "nearest":
                interpolated_values = self._fill_nans_from_nearests(
                    interpolated_values, x=x, y=y, z=z
                )

        coords_dict = {"x": x, "y": y, "z": z}
        coords_dict.update(self._non_spatial_coords_dict)

        if len(self._non_spatial_coords_dict) == 0:
            return SpatialDataArray(interpolated_values, coords=coords_dict, name=self.values.name)
        else:
            return XrDataArray(interpolated_values, coords=coords_dict, name=self.values.name)

    def _interp_nearest(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
    ) -> ArrayLike:
        """Interpolate data at provided x, y, and z using Scipy's nearest neighbor interpolator.

        Parameters
        ----------
        x : ArrayLike
            x-coordinates of sampling points.
        y : ArrayLike
            y-coordinates of sampling points.
        z : ArrayLike
            z-coordinates of sampling points.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """
        from scipy.interpolate import NearestNDInterpolator

        # use scipy's nearest neighbor interpolator
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        interp = NearestNDInterpolator(self._points_3d_array, self.values.values)
        values = interp(X, Y, Z)

        return values

    def _fill_nans_from_nearests(
        self,
        values: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
    ) -> ArrayLike:
        """Replace nan's in ``values`` with nearest data points.

        Parameters
        ----------
        values : ArrayLike
            3D array containing nan's
        x : ArrayLike
            x-coordinates of sampling points.
        y : ArrayLike
            y-coordinates of sampling points.
        z : ArrayLike
            z-coordinates of sampling points.

        Returns
        -------
        ArrayLike
            Data without nan's.
        """

        # locate all nans
        # do a quick and dirty in case of multiple fields: just look at the very first field
        nans = np.isnan(values).reshape((len(x), len(y), len(z), self._num_fields))[:, :, :, 0]

        if np.sum(nans) > 0:
            from scipy.interpolate import NearestNDInterpolator

            # use scipy's nearest neighbor interpolator
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            interp = NearestNDInterpolator(self._points_3d_array, self.values.values)
            values_to_replace_nans = interp(X[nans], Y[nans], Z[nans])
            values[nans] = values_to_replace_nans

        return values

    @requires_vtk
    def _interp_vtk(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,  # TODO: an array if multidimensional
    ) -> ArrayLike:
        """Interpolate data at provided x, y, and z using vtk package.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float = 0
            Value to use when filling points without interpolated values.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

        shape = (len(x), len(y), len(z))

        # create a VTK rectilinear grid to sample onto
        structured_grid = vtk["mod"].vtkRectilinearGrid()
        structured_grid.SetDimensions(shape)
        structured_grid.SetXCoordinates(vtk["numpy_to_vtk"](x))
        structured_grid.SetYCoordinates(vtk["numpy_to_vtk"](y))
        structured_grid.SetZCoordinates(vtk["numpy_to_vtk"](z))

        # create and execute VTK interpolator
        interpolator = vtk["mod"].vtkResampleWithDataSet()
        interpolator.SetInputData(structured_grid)
        interpolator.SetSourceData(self._vtk_obj)
        interpolator.Update()
        interpolated = interpolator.GetOutput()

        # get results in a numpy representation
        array_id = 0 if self.values.name is None else self.values.name

        # TODO: generalize this
        values_numpy = np.array(
            vtk["vtk_to_numpy"](interpolated.GetPointData().GetAbstractArray(array_id)), copy=True
        )

        # fill points without interpolated values
        if fill_value != 0:
            mask = np.array(
                vtk["vtk_to_numpy"](
                    interpolated.GetPointData().GetAbstractArray("vtkValidPointMask")
                ),
                copy=True,
            )
            values_numpy[mask != 1] = fill_value

        # VTK arrays are the z-y-x order, reorder interpolation results to x-y-z order
        values_reordered = np.transpose(np.reshape(values_numpy, shape[::-1]), (2, 1, 0))

        return values_reordered

    @abstractmethod
    def _interp_py(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,
        max_samples_per_step: int,
        max_cells_per_step: int,
        rel_tol: float,
    ) -> ArrayLike:
        """Dimensionality-specific function (2D and 3D) to interpolate data at provided x, y, and z
        using vectorized python implementation.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float
            Value to use when filling points without interpolated values.
        max_samples_per_step : int
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

    def _interp_py_general(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,
        max_samples_per_step: int,
        max_cells_per_step: int,
        rel_tol: float,
        axis_ignore: Axis | None,
    ) -> ArrayLike:
        """A general function (2D and 3D) to interpolate data at provided x, y, and z using
        vectorized python implementation.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float
            Value to use when filling points without interpolated values.
        max_samples_per_step : int
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float
            Relative tolerance when determining whether a point belongs to a cell.
        axis_ignore : Union[Axis, None]
            When interpolating from a 2D dataset, must specify normal axis.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

        # get dimensionality of data
        num_dims = self._point_dims()

        if num_dims == 2 and axis_ignore is None:
            raise DataError("Must provide 'axis_ignore' when interpolating from a 2d dataset.")

        xyz_grid = [x, y, z]

        if axis_ignore is not None:
            xyz_grid.pop(axis_ignore)

        # get numpy arrays for points and cells
        cell_connections = (
            self.cells.values
        )  # (num_cells, num_cell_vertices), num_cell_vertices=num_cell_faces
        points = self.points.values  # (num_points, num_dims)

        num_cells = len(cell_connections)
        num_points = len(points)

        # compute tolerances based on total size of unstructured grid
        bounds = self.bounds
        size = np.subtract(bounds[1], bounds[0])
        tol = size * rel_tol
        diag_tol = np.linalg.norm(tol)

        # compute (index) positions of unstructured points w.r.t. target Cartesian grid points
        # (i.e. between which Cartesian grid points a given unstructured grid point is located)
        # we perturb grid values in both directions to make sure we don't miss any points
        # due to numerical precision
        xyz_pos_l = np.zeros((num_dims, num_points), dtype=int)
        xyz_pos_r = np.zeros((num_dims, num_points), dtype=int)
        for dim in range(num_dims):
            xyz_pos_l[dim] = np.searchsorted(xyz_grid[dim] + tol[dim], points[:, dim])
            xyz_pos_r[dim] = np.searchsorted(xyz_grid[dim] - tol[dim], points[:, dim])

        # let's allocate an array for resulting values
        # every time we process a chunk of samples, we will write into this array
        interpolated_values = fill_value + np.zeros(
            [len(xyz_comp) for xyz_comp in xyz_grid] + self._non_spatial_shape,
            dtype=self.values.dtype,
        )

        processed_cells_global = 0

        # to ovoid OOM for large datasets, we process only certain number of cells at a time
        while processed_cells_global < num_cells:
            target_processed_cells_global = min(
                num_cells, processed_cells_global + max_cells_per_step
            )

            connections_to_process = cell_connections[
                processed_cells_global:target_processed_cells_global
            ]

            # now we transfer this information to each cell. That is, each cell knows how its vertices
            # positioned relative to Cartesian grid points.
            # (num_dims, num_cells, num_vertices=num_cell_faces)
            xyz_pos_l_per_cell = xyz_pos_l[:, connections_to_process]
            xyz_pos_r_per_cell = xyz_pos_r[:, connections_to_process]

            # taking min/max among all cell vertices (per each dimension separately)
            # we get min and max indices of Cartesian grid points that may receive their values
            # from a given cell.
            # (num_dims, num_cells)
            cell_ind_min = np.min(xyz_pos_l_per_cell, axis=2)
            cell_ind_max = np.max(xyz_pos_r_per_cell, axis=2)

            # calculate number of Cartesian grid points where we will perform interpolation for a given
            # cell. Note that this number is much larger than actually needed, because essentially for
            # each cell we consider all Cartesian grid points that fall into the cell's bounding box.
            # We use word "sample" to represent such Cartesian grid points.
            # (num_cells,)
            num_samples_per_cell = np.prod(cell_ind_max - cell_ind_min, axis=0)

            # find cells that have non-zero number of samples
            # we use "ne" as a shortcut for "non empty"
            ne_cells = num_samples_per_cell > 0  # (num_cells,)
            num_ne_cells = np.sum(ne_cells)
            # indices of cells with non-zero number of samples in the original list of cells
            # (num_cells,)
            ne_cell_inds = np.arange(processed_cells_global, target_processed_cells_global)[
                ne_cells
            ]

            # restrict to non-empty cells only
            num_samples_per_ne_cell = num_samples_per_cell[ne_cells]
            cum_num_samples_per_ne_cell = np.cumsum(num_samples_per_ne_cell)

            ne_cell_ind_min = cell_ind_min[:, ne_cells]
            ne_cell_ind_max = cell_ind_max[:, ne_cells]

            # Next we need to perform actual interpolation at all sample points
            # this is computationally expensive operation and because we try to do everything
            # in the vectorized form, it can require a lot of memory, sometimes even causing OOM errors.
            # To avoid that, we impose restrictions on how many cells/samples can be processed at a time
            # effectivelly performing these operations in chunks.
            # Note that currently this is done sequentially, but could be relatively easy to parallelize

            # start counters of how many cells/samples have been processed
            processed_samples = 0
            processed_cells = 0

            while processed_cells < num_ne_cells:
                # how many cells we would like to process by the end of this step
                target_processed_cells = min(num_ne_cells, processed_cells + max_cells_per_step)

                # find how many cells we can processed based on number of allowed samples
                target_processed_samples = processed_samples + max_samples_per_step
                target_processed_cells_from_samples = (
                    np.searchsorted(cum_num_samples_per_ne_cell, target_processed_samples) + 1
                )

                # take min between the two
                target_processed_cells = min(
                    target_processed_cells, target_processed_cells_from_samples
                )

                # select cells and corresponding samples to process
                step_ne_cell_ind_min = ne_cell_ind_min[:, processed_cells:target_processed_cells]
                step_ne_cell_ind_max = ne_cell_ind_max[:, processed_cells:target_processed_cells]
                step_ne_cell_inds = ne_cell_inds[processed_cells:target_processed_cells]

                # process selected cells and points
                xyz_inds, interpolated = self._interp_py_chunk(
                    xyz_grid=xyz_grid,
                    cell_inds=step_ne_cell_inds,
                    cell_ind_min=step_ne_cell_ind_min,
                    cell_ind_max=step_ne_cell_ind_max,
                    sdf_tol=diag_tol,
                )

                if num_dims == 3:
                    interpolated_values[xyz_inds[0], xyz_inds[1], xyz_inds[2]] = interpolated
                else:
                    interpolated_values[xyz_inds[0], xyz_inds[1]] = interpolated

                processed_cells = target_processed_cells
                processed_samples = cum_num_samples_per_ne_cell[target_processed_cells - 1]

            processed_cells_global = target_processed_cells_global

        # in case of 2d grid broadcast results along normal direction assuming translational
        # invariance
        if num_dims == 2:
            orig_shape = [len(x), len(y), len(z), *self._non_spatial_shape]
            flat_shape = orig_shape.copy()
            flat_shape[axis_ignore] = 1
            interpolated_values = np.reshape(interpolated_values, flat_shape)
            interpolated_values = np.broadcast_to(interpolated_values, orig_shape).copy()

        return interpolated_values

    def _interp_py_chunk(
        self,
        xyz_grid: tuple[ArrayLike[float], ...],
        cell_inds: ArrayLike[int],
        cell_ind_min: ArrayLike[int],
        cell_ind_max: ArrayLike[int],
        sdf_tol: float,
    ) -> tuple[tuple[ArrayLike, ...], ArrayLike]:
        """For each cell listed in ``cell_inds`` perform interpolation at a rectilinear subarray of
        xyz_grid given by a (3D) index span (cell_ind_min, cell_ind_max).

        Parameters
        ----------
        xyz_grid : tuple[ArrayLike[float], ...]
            x, y, and z coordiantes defining rectilinear grid.
        cell_inds : ArrayLike[int]
            Indices of cells to perfrom interpolation from.
        cell_ind_min : ArrayLike[int]
            Starting x, y, and z indices of points for interpolation for each cell.
        cell_ind_max : ArrayLike[int]
            End x, y, and z indices of points for interpolation for each cell.
        sdf_tol : float
            Effective zero level set value, below which a point is considered to be inside a cell.

        Returns
        -------
        tuple[tuple[ArrayLike, ...], ArrayLike]
            x, y, and z indices of interpolated values and values themselves.
        """

        # get dimensionality of data
        num_dims = self._point_dims()
        num_cell_faces = self._cell_num_vertices()

        # get mesh info as numpy arrays
        points = self.points.values  # (num_points, num_dims)
        data_values = self.values  # (num_points,)
        cell_connections = self.cells.values[cell_inds]

        # compute number of samples to generate per cell
        num_samples_per_cell = np.prod(cell_ind_max - cell_ind_min, axis=0)

        # at this point we know how many samples we need to perform per each cell and we also
        # know span indices of these samples (in x, y, and z arrays)

        # we would like to perform all interpolations in a vectorized form, however, we have
        # a different number of interpolation samples for different cells. Thus, we need to
        # arange all samples in a linear way (flatten). Basically, we want to have data in this
        # form:
        # cell_ind | x_ind | y_ind | z_ind
        # --------------------------------
        #        0 |    23 |     5 |    11
        #        0 |    23 |     5 |    12
        #        0 |    23 |     6 |    11
        #        0 |    23 |     6 |    12
        #        1 |    41 |    11 |     0
        #        1 |    42 |    11 |     0
        #      ... |   ... |   ... |   ...

        # to do that we start with performing arange for each cell, but in vectorized way
        # this gives us something like this
        # [0, 1, 2, 3,   0, 1,   0, 1, 2, 3, 4, 5, 6,   ...]
        # |<-cell 0->|<-cell 1->|<-     cell 2    ->|<- ...

        num_cells = len(num_samples_per_cell)
        num_samples_cumul = num_samples_per_cell.cumsum()
        num_samples_total = num_samples_cumul[-1]

        # one big arange array
        inds_flat = np.arange(num_samples_total)
        # now subtract previous number of samples
        inds_flat[num_samples_per_cell[0] :] -= np.repeat(
            num_samples_cumul[:-1], num_samples_per_cell[1:]
        )

        # convert flat indices into 3d/2d indices as:
        # x_ind = [23, 23, 23, 23,   41, 41,      ...]
        # y_ind = [ 5,  5,  5,  5,    6,  6,      ...]
        # z_ind = [11, 12, 11, 12,    0,  0,      ...]
        #         |<-  cell 0  ->|<- cell 1 ->|<- ...
        num_samples_y = np.repeat(cell_ind_max[1] - cell_ind_min[1], num_samples_per_cell)

        # note: in 2d x, y correspond to (x, y, z).pop(normal_axis)
        if num_dims == 3:
            num_samples_z = np.repeat(cell_ind_max[2] - cell_ind_min[2], num_samples_per_cell)
            inds_flat, z_inds = np.divmod(inds_flat, num_samples_z)

        x_inds, y_inds = np.divmod(inds_flat, num_samples_y)

        start_inds = np.repeat(cell_ind_min, num_samples_per_cell, axis=1)
        x_inds = x_inds + start_inds[0]
        y_inds = y_inds + start_inds[1]
        if num_dims == 3:
            z_inds = z_inds + start_inds[2]

        # finally, we repeat cell indices corresponding number of times to obtain how
        # (x_ind, y_ind, z_ind) map to cell indices. So, now we have four arras:
        # x_ind    = [23, 23, 23, 23,   41, 41,      ...]
        # y_ind    = [ 5,  5,  5,  5,    6,  6,      ...]
        # z_ind    = [11, 12, 11, 12,    0,  0,      ...]
        # cell_map = [ 0,  0,  0,  0,    1,  1,      ...]
        #            |<-  cell 0  ->|<- cell 1 ->|<- ...
        step_cell_map = np.repeat(np.arange(num_cells), num_samples_per_cell)

        # let's put these arrays aside for a moment and perform the second preparatory step
        # specifically, for each face of each cell we will compute normal vector and distance
        # to the opposing cell vertex. This will allows us quickly calculate SDF of a cell at
        # each sample point as well as perform linear interpolation.

        # first, we collect coordinates of cell vertices into a single array
        # (num_cells, num_cell_vertices, num_dims)
        cell_vertices = np.float64(points[cell_connections, :])

        # array for resulting normals and distances
        normal = np.zeros((num_cell_faces, num_cells, num_dims))
        dist = np.zeros((num_cell_faces, num_cells))

        # loop face by face
        # note that by face_ind we denote both index of face in a cell and index of the opposing vertex
        for face_ind in range(num_cell_faces):
            # select vertices forming the given face
            face_pinds = list(np.arange(num_cell_faces))
            face_pinds.pop(face_ind)

            # calculate normal to the face
            # in 3D: cross product of two vectors lying in the face plane
            # in 2D: (-ty, tx) for a vector (tx, ty) along the face
            p0 = cell_vertices[:, face_pinds[0]]
            p01 = cell_vertices[:, face_pinds[1]] - p0
            p0Opp = cell_vertices[:, face_ind] - p0
            if num_dims == 3:
                p02 = cell_vertices[:, face_pinds[2]] - p0
                n = np.cross(p01, p02)
            else:
                n = np.roll(p01, 1, axis=1)
                n[:, 0] = -n[:, 0]
            n_norm = np.linalg.norm(n, axis=1)
            n = np.divide(
                n,
                n_norm[:, None],
                out=np.zeros_like(n),
                where=n_norm[:, None] > 0,
            )

            # compute distance to the opposing vertex by taking a dot product between normal
            # and a vector connecting the opposing vertex and the face
            d = np.einsum("ij,ij->i", n, p0Opp)

            # obtained normal direction is arbitrary here. We will orient it such that it points
            # away from the triangle (and distance to the opposing vertex is negative).
            to_flip = d > 0
            d[to_flip] *= -1
            n[to_flip, :] *= -1

            # set distances in degenerate triangles to something positive to ignore later
            dist_zero = d == 0
            if any(dist_zero):
                d[dist_zero] = 1

            # record obtained info
            normal[face_ind] = n
            dist[face_ind] = d

        # now we all set up to proceed with actual interpolation at each sample point
        # the main idea here is that:
        # - we use `cell_map` to grab normals and distances
        #   of cells in which the given sample point is (potentially) located.
        # - use `x_ind, y_ind, z_ind` to find actual coordinates of a given sample point
        # - combine the above two to calculate cell SDF and interpolated value at a given sample
        #   point
        # - having cell SDF at the sample point actually tells us whether its inside the cell
        #   (keep value) or outside of it (discard interpolated value)

        # to perform SDF calculation and interpolation we will loop face by face and recording
        # their contributions. That is,
        # cell_sdf = max(face0_sdf, face1_sdf, ...)
        # interpolated_value = value0 * face0_sdf / dist0_sdf + ...
        # (because face0_sdf / dist0_sdf is linear shape function for vertex0)
        sdf = -inf * np.ones(num_samples_total)
        interpolated = np.zeros(
            [num_samples_total, *self._non_spatial_shape], dtype=self._double_type
        )
        weight_min = np.full(num_samples_total, inf)
        weight_max = np.full(num_samples_total, -inf)
        weight_sum = np.zeros(num_samples_total)

        # coordinates of each sample point
        sample_xyz = np.zeros((num_samples_total, num_dims))
        sample_xyz[:, 0] = xyz_grid[0][x_inds]
        sample_xyz[:, 1] = xyz_grid[1][y_inds]
        if num_dims == 3:
            sample_xyz[:, 2] = xyz_grid[2][z_inds]

        # loop face by face
        for face_ind in range(num_cell_faces):
            # find a vector connecting sample point and face
            if face_ind == 0:
                vertex_ind = 1  # anythin other than 0
                vec = sample_xyz - cell_vertices[step_cell_map, vertex_ind, :]

            if face_ind == 1:  # since three faces share a point only do this once
                vertex_ind = 0  # it belongs to every face 1, 2, and 3
                vec = sample_xyz - cell_vertices[step_cell_map, 0, :]

            # compute distance from every sample point to the face of corresponding cell
            # using dot product
            tmp = normal[face_ind, step_cell_map, :] * vec
            d = np.sum(tmp, axis=1)

            # take max between distance to obtain the overall SDF of a cell
            sdf = np.maximum(sdf, d)

            # perform linear interpolation. Here we use the fact that when computing face SDF
            # at a given point and dividing it by the distance to the opposing vertex we get
            # a linear shape function for that vertex. So, we just need to multiply that by
            # the data value at that vertex to find its contribution into intepolated value.
            # (decomposed in an attempt to reduce memory consumption)
            tmp = self._double_type(
                data_values.sel(index=cell_connections[step_cell_map, face_ind]).data
            )
            weight = d / dist[face_ind, step_cell_map]
            tmp *= np.reshape(
                weight,
                [num_samples_total] + [1] * len(self._non_spatial_shape),
            )
            weight_min = np.minimum(weight_min, weight)
            weight_max = np.maximum(weight_max, weight)
            weight_sum += weight

            # ignore degenerate cells
            dist_zero = dist[face_ind, step_cell_map] > 0
            if any(dist_zero):
                sdf[dist_zero] = 10 * sdf_tol

            interpolated += tmp

        # The resulting array of interpolated values contain multiple candidate values for
        # every Cartesian point because bounding boxes of cells overlap.
        # Thus, we need to keep only those that come cell actually containing a given point.
        # This can be easily determined by the sign of the cell SDF sampled at a given point.
        valid_weights = (
            np.isfinite(weight_sum)
            & (np.abs(weight_sum - 1) <= BARYCENTRIC_WEIGHT_TOLERANCE)
            & (weight_min >= -BARYCENTRIC_WEIGHT_TOLERANCE)
            & (weight_max <= 1 + BARYCENTRIC_WEIGHT_TOLERANCE)
        )
        valid_samples = (sdf < sdf_tol) & valid_weights

        interpolated_valid = interpolated[valid_samples]
        xyz_valid_inds = []
        xyz_valid_inds.append(x_inds[valid_samples])
        xyz_valid_inds.append(y_inds[valid_samples])
        if num_dims == 3:
            xyz_valid_inds.append(z_inds[valid_samples])

        return xyz_valid_inds, interpolated_valid

    """ Data selection """

    @requires_vtk
    def sel_inside(self, bounds: Bound) -> UnstructuredGridDataset:
        """Return a new UnstructuredGridDataset that contains the minimal amount data necessary to
        cover a spatial region defined by ``bounds``.

        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        UnstructuredGridDataset
            Extracted spatial data array.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``."
            )

        data_bounds = self.bounds
        tol = 1e-6

        # For extracting cells covering target region we use vtk's filter that extract cells based
        # on provided implicit function. However, when we provide to it the implicit function of
        # the entire box, it has a couple of issues coming from the fact that the algorithm
        # eliminates every cells for which the implicit function has positive sign at all vertices.
        # As result, sometimes there are cells that despite overlaping with the target domain still
        # being eliminated. Two common cases:
        # - near corners of the target domain
        # - target domain is very thin
        # That's why we perform selection by sequentially eliminating cells on the outer side of
        # each of the 6 surfaces of the bounding box separately.
        tmp = self._vtk_obj
        for direction in range(2):
            for dim in range(3):
                sign = -1 + 2 * direction
                plane_pos = bounds[direction][dim]

                # Dealing with situation when target region does intersect with any cell:
                # in this case we shift target region so that it barely touches at least some
                # of cells
                if sign < 0 and plane_pos > data_bounds[1][dim] - tol:
                    plane_pos = data_bounds[1][dim] - tol
                if sign > 0 and plane_pos < data_bounds[0][dim] + tol:
                    plane_pos = data_bounds[0][dim] + tol

                # if all cells are on the inside side of the plane for a given surface
                # we don't need to check for intersection
                if plane_pos <= data_bounds[1][dim] and plane_pos >= data_bounds[0][dim]:
                    plane = vtk["mod"].vtkPlane()
                    center = [0, 0, 0]
                    normal = [0, 0, 0]
                    center[dim] = plane_pos
                    normal[dim] = sign
                    plane.SetOrigin(center)
                    plane.SetNormal(normal)
                    extractor = vtk["mod"].vtkExtractGeometry()
                    extractor.SetImplicitFunction(plane)
                    extractor.ExtractInsideOn()
                    extractor.ExtractBoundaryCellsOn()
                    extractor.SetInputData(tmp)
                    extractor.Update()
                    tmp = extractor.GetOutput()

        return self._from_vtk_obj_internal(tmp)

    def does_cover(self, bounds: Bound) -> bool:
        """Check whether data fully covers specified by ``bounds`` spatial region. If data contains
        only one point along a given direction, then it is assumed the data is constant along that
        direction and coverage is not checked.

        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        bool
            Full cover check outcome.
        """

        return all(
            (dmin <= smin and dmax >= smax)
            for dmin, dmax, smin, smax in zip(self.bounds[0], self.bounds[1], bounds[0], bounds[1])
        )
