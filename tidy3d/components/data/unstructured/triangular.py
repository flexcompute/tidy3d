"""Defines triangular grid datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field
from xarray import DataArray as XrDataArray

if TYPE_CHECKING:
    from matplotlib.tri import Triangulation

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    CellDataArray,
    IndexedDataArray,
    PointDataArray,
    SpatialDataArray,
)
from tidy3d.components.types import Axis
from tidy3d.components.viz import add_ax_if_none, equal_aspect, plot_params_grid
from tidy3d.constants import inf
from tidy3d.exceptions import DataError
from tidy3d.log import log
from tidy3d.packaging import requires_vtk, vtk

from .base import (
    DEFAULT_MAX_CELLS_PER_STEP,
    DEFAULT_MAX_SAMPLES_PER_STEP,
    DEFAULT_TOLERANCE_CELL_FINDING,
    UnstructuredGridDataset,
    cut_point_neighbor_positions,
    planar_zero_dim_tolerance,
)

# Separation between the two sides of an interface: a fraction of the sample spacing the
# grid resolves there, floored by a few ULPs of the coordinate itself. The float64 floor
# is what makes the two sides distinct coordinates at all, so it always applies; the
# float32 target is only met when the local spacing leaves room for it, which it does not
# once a float32 ULP has grown past the spacing the grid resolves.
INTERFACE_NUDGE_FRACTION = 0.25
INTERFACE_NUDGE_ULPS_FLOAT64 = 4
INTERFACE_NUDGE_ULPS_FLOAT32 = 8

if TYPE_CHECKING:
    from typing import Literal

    from pydantic import PositiveInt
    from vtkmodules.vtkCommonDataModel import vtkPointSet
    from xarray import DataArray

    from tidy3d.compat import Self
    from tidy3d.components.types import ArrayLike, Ax, Bound, Coordinate


def _interface_nudge_span(center: float, scale: float, count: int) -> float:
    """Half-width separating a run of ``count`` coincident samples about ``center``.

    ``scale`` is the sample spacing the grid resolves around the run. The separation aims
    at a float32 ULP, which keeps the two sides apart through a plot or an export, and
    never drops below a float64 ULP, which is what makes them distinct at all. Both are
    then held under ``scale``, so a nudged sample can never reach a neighbour -- where a
    float32 ULP is coarser than the spacing the grid resolves, only float64 is delivered.
    """

    # the run is spread across 2 * span, so the float64 floor grows with its length
    float64_floor = INTERFACE_NUDGE_ULPS_FLOAT64 * count * float(np.spacing(abs(center)))
    # measured at the local spacing where that exceeds the coordinate: a float32 ULP at
    # zero is a denormal, which would leave the samples on top of each other
    magnitude = max(abs(center), scale)
    if magnitude > float(np.finfo(np.float32).max):
        # past the float32 range no offset survives the cast, so only float64 is served
        float32_target = float64_floor
    else:
        float32_target = INTERFACE_NUDGE_ULPS_FLOAT32 * float(np.spacing(np.float32(magnitude)))
    span = max(min(INTERFACE_NUDGE_FRACTION * scale, float32_target), float64_floor)
    return min(span, 0.5 * scale)


def _nudge_repeated_coordinates(coords: np.ndarray) -> np.ndarray:
    """Separate the coincident coordinates a cut across a material interface leaves.

    The grid holds a node per side of an interface, so the cut returns a sample per
    side at one coordinate. Left coincident, which side a sample belongs to is carried
    only by array order, and any sort the caller applies breaks that tie arbitrarily --
    a band diagram that steps the wrong way. Separating the pair puts the side into the
    coordinate itself, where no reordering can lose it, at an offset held well under the
    spacing the grid resolves.

    ``coords`` arrives sorted ascending, as the cut hands it over, and the result is
    strictly increasing -- enforced here, because it is the only thing standing between a
    slice and a duplicated coordinate reaching interpolation. That guarantee is a float64
    one; whether the pair also survives a float32 round-trip depends on the local spacing.
    """

    nudged = np.asarray(coords, dtype=float).copy()
    start = 0
    while start < len(nudged):
        stop = start + 1
        while stop < len(nudged) and nudged[stop] == nudged[start]:
            stop += 1
        count = stop - start
        if count > 1:
            center = nudged[start]
            left_gap = center - nudged[start - 1] if start > 0 else np.inf
            right_gap = nudged[stop] - center if stop < len(nudged) else np.inf
            finite_gaps = [gap for gap in (left_gap, right_gap) if np.isfinite(gap) and gap > 0]
            # a run with a neighbour on neither side has no resolved spacing to hide under
            scale = min(finite_gaps) if finite_gaps else max(abs(center), 1.0)
            span = _interface_nudge_span(center, scale, count)
            # symmetric, so a run at either end of the array separates like any other
            nudged[start:stop] = np.linspace(center - span, center + span, count)
        start = stop
    if not np.all(np.diff(nudged) > 0):
        raise DataError(
            "slice coordinates are not strictly increasing after separating the interface "
            "samples: they must arrive sorted, and neighbouring samples must be far enough "
            "apart for float64 to hold the offset between them"
        )
    return nudged


class TriangularGridDataset(UnstructuredGridDataset):
    """Dataset for storing triangular grid data. Data values are associated with the nodes of
    the grid.

    Note
    ----
    To use full functionality of unstructured datasets one must install ``vtk`` package (``pip
    install tidy3d[vtk]`` or ``pip install vtk``). Otherwise the functionality of unstructured
    datasets is limited to creation, writing to/loading from a file, and arithmetic manipulations.

    Example
    -------
    >>> tri_grid_points = PointDataArray(
    ...     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    ...     coords=dict(index=np.arange(4), axis=np.arange(2)),
    ... )
    >>>
    >>> tri_grid_cells = CellDataArray(
    ...     [[0, 1, 2], [1, 2, 3]],
    ...     coords=dict(cell_index=np.arange(2), vertex_index=np.arange(3)),
    ... )
    >>>
    >>> tri_grid_values = IndexedDataArray(
    ...     [1.0, 2.0, 3.0, 4.0], coords=dict(index=np.arange(4)),
    ... )
    >>>
    >>> tri_grid = TriangularGridDataset(
    ...     normal_axis=1,
    ...     normal_pos=0,
    ...     points=tri_grid_points,
    ...     cells=tri_grid_cells,
    ...     values=tri_grid_values,
    ... )
    """

    normal_axis: Axis = Field(
        title="Grid Axis",
        description="Orientation of the grid.",
    )

    normal_pos: float = Field(
        title="Position",
        description="Coordinate of the grid along the normal direction.",
    )

    """ Fundamental parameters to set up based on grid dimensionality """

    @classmethod
    def _point_dims(cls) -> PositiveInt:
        """Dimensionality of stored grid point coordinates."""
        return 2

    @classmethod
    def _cell_num_vertices(cls) -> PositiveInt:
        """Number of vertices in a cell."""
        return 3

    """ Convenience properties """

    @cached_property
    def bounds(self) -> Bound:
        """Grid bounds."""
        bounds_2d = super().bounds
        bounds_3d = self._points_2d_to_3d(bounds_2d)
        return tuple(bounds_3d[0]), tuple(bounds_3d[1])

    def _points_2d_to_3d(self, pts: ArrayLike) -> ArrayLike:
        """Convert 2d points into 3d points."""
        return np.insert(pts, obj=self.normal_axis, values=self.normal_pos, axis=1)

    @cached_property
    def _points_3d_array(self) -> ArrayLike:
        """3D representation of grid points."""
        return self._points_2d_to_3d(self.points.data)

    """ VTK interfacing """

    @classmethod
    @requires_vtk
    def _vtk_cell_type(cls) -> int:
        """VTK cell type to use in the VTK representation."""
        return vtk["mod"].VTK_TRIANGLE

    @classmethod
    @requires_vtk
    def _from_vtk_obj(
        cls,
        vtk_obj: vtkPointSet,
        field: str | None = None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
        values_type: type = IndexedDataArray,
        expect_complex: bool | None = None,
        ignore_invalid_cells: bool = False,
        warn_unused_points: bool = True,
    ) -> Self:
        """Initialize from a vtkUnstructuredGrid instance."""

        # get points cells data from vtk object
        if isinstance(vtk_obj, vtk["mod"].vtkPolyData):
            cells_vtk = vtk_obj.GetPolys()
        elif isinstance(vtk_obj, vtk["mod"].vtkUnstructuredGrid):
            cells_vtk = vtk_obj.GetCells()

        cells_numpy = np.array(
            vtk["vtk_to_numpy"](cells_vtk.GetConnectivityArray()),
            copy=True,
        )

        # verify cell_types
        cell_offsets = np.array(vtk["vtk_to_numpy"](cells_vtk.GetOffsetsArray()), copy=True)
        invalid_cells = np.diff(cell_offsets) != cls._cell_num_vertices()
        if np.any(invalid_cells):
            if ignore_invalid_cells:
                valid_cell_offsets = cell_offsets[:-1][invalid_cells == 0]
                cells_numpy = cells_numpy[
                    np.ravel(valid_cell_offsets[:, None] + np.arange(3, dtype=int)[None, :])
                ]
            else:
                raise DataError(
                    "Only triangular 'vtkUnstructuredGrid' or 'vtkPolyData' can be converted into "
                    "'TriangularGridDataset'."
                )

        points_numpy = np.array(vtk["vtk_to_numpy"](vtk_obj.GetPoints().GetData()), copy=True)

        # data values are read directly into Tidy3D array
        values = cls._get_values_from_vtk(
            vtk_obj, len(points_numpy), field, values_type, expect_complex
        )

        # detect zero size dimension
        bounds = np.max(points_numpy, axis=0) - np.min(points_numpy, axis=0)
        size_scale = float(np.max(bounds)) if bounds.size > 0 else 0.0
        zero_dim_tol = planar_zero_dim_tolerance(size_scale)
        zero_dims = np.where(np.isclose(bounds, 0, atol=zero_dim_tol))[0]

        if len(zero_dims) != 1:
            raise DataError(
                f"Provided vtk grid does not represent a two dimensional grid. Found zero size dimensions are {zero_dims}."
            )

        normal_axis = int(zero_dims[0])
        normal_pos = points_numpy[0][normal_axis]
        tan_dims = [0, 1, 2]
        tan_dims.remove(normal_axis)

        # convert 3d coordinates into 2d
        points_2d_numpy = points_numpy[:, tan_dims]

        # create Tidy3D points and cells arrays
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
            points_2d_numpy,
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
            normal_axis=normal_axis,
            normal_pos=normal_pos,
            points=points,
            cells=cells,
            values=values,
        )

    """ Grid operations """

    @requires_vtk
    def plane_slice(self, axis: Axis, pos: float) -> XrDataArray:
        """Slice data with a plane and return the resulting line as a DataArray.

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.

        Returns
        -------
        xarray.DataArray
            The resulting slice.
        """

        if axis == self.normal_axis:
            raise DataError(
                f"Triangular grid (normal: {self.normal_axis}) cannot be sliced by a parallel "
                "plane."
            )

        # perform slicing in vtk and get unprocessed points and values
        slice_vtk = self._plane_slice_raw(axis=axis, pos=pos)

        # the plane can lie inside the bounds -- all '_plane_slice_raw' checks -- and still
        # miss every cell, e.g. crossing a hole in a non-convex grid. An empty slice carries
        # no points object at all, so this has to come before reading the points.
        if slice_vtk.GetNumberOfPoints() == 0:
            raise DataError(
                f"Slicing plane (axis: {axis}, pos: {pos}) does not intersect any cell of the "
                "unstructured grid."
            )

        points_numpy = np.array(vtk["vtk_to_numpy"](slice_vtk.GetPoints().GetData()), copy=True)
        values = self._get_values_from_vtk(
            slice_vtk,
            len(points_numpy),
            field=self._non_spatial_coords_dict,
            values_type=self._values_type,
            expect_complex=self.is_complex,
        )

        # axis of the resulting line
        slice_axis = 3 - self.normal_axis - axis

        # A line crossing a material interface carries both sides' values at that one
        # coordinate, so sorting on the coordinate alone leaves their order to the sort's
        # tie-breaking. Order them by the side their segments reach into, lower side first,
        # so the profile reads monotonically through the jump.
        along = points_numpy[:, slice_axis]
        order = np.lexsort((cut_point_neighbor_positions(slice_vtk, slice_axis), along))
        along = _nudge_repeated_coordinates(along[order])
        ordered_values = np.asarray(values.data)[order]

        # assemble coords for DataArray
        coords = [None, None, None]
        coords[axis] = [pos]
        coords[self.normal_axis] = [self.normal_pos]
        coords[slice_axis] = along
        coords_dict = dict(zip("xyz", coords))
        coords_dict.update(self._non_spatial_coords_dict)

        # reshape values from a 1d array into a 3d array
        new_shape = [1, 1, 1]
        new_shape[slice_axis] = len(values.index)
        new_shape = new_shape + list(np.shape(values.data))[1:]
        values_reshaped = np.reshape(ordered_values, new_shape)

        return XrDataArray(values_reshaped, coords=coords_dict, name=self.values.name)

    @requires_vtk
    def line_slice(self, axis: Axis, pos: Coordinate) -> XrDataArray:
        """Slice data with a line and return the resulting xarray.DataArray.

        Parameters
        ----------
        axis : Axis
            The axis of the slicing line.
        pos : Tuple[float, float, float]
            Position of the slicing line.

        Returns
        -------
        xarray.DataArray
            The resulting slice.

        Notes
        -----
            The line must lie in the plane of the grid. No 3D probing is needed, unlike
            the tetrahedral case: one in-plane slice already reduces the grid to a line.

            The result holds one sample per mesh node the line crosses. Where the grid
            carries a node per side of a material interface, both values are kept and
            their coordinates are separated by a small fraction of the neighboring
            sample spacing. This preserves the jump while keeping the coordinate unique
            for selection and interpolation.
        """

        if axis == self.normal_axis:
            raise DataError(
                f"Triangular grid (normal: {self.normal_axis}) cannot be sliced by a line "
                "along its normal direction."
            )

        # the in-plane slice ignores the normal coordinate, so an off-plane line would
        # silently return data for the wrong position
        size_scale = float(np.max(np.array(self.bounds[1]) - np.array(self.bounds[0])))
        if not np.isclose(
            pos[self.normal_axis],
            self.normal_pos,
            rtol=0.0,
            atol=planar_zero_dim_tolerance(size_scale),
        ):
            raise DataError(
                f"Slicing line at {'xyz'[self.normal_axis]} = {pos[self.normal_axis]} does not "
                f"lie in the plane of the triangular grid ({'xyz'[self.normal_axis]} = "
                f"{self.normal_pos})."
            )

        slice_axis = 3 - self.normal_axis - axis
        return self.plane_slice(axis=slice_axis, pos=pos[slice_axis])

    @requires_vtk
    def reflect(
        self, axis: Axis, center: float, reflection_only: bool = False, symmetry: float = 1.0
    ) -> UnstructuredGridDataset:
        """Reflect unstructured data across the plane define by parameters ``axis`` and ``center``.
        By default the original data is preserved, setting ``reflection_only`` to ``True`` will
        produce only reflected data.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected data.
        symmetry : float = 1.0
            Symmetry factor to apply to the data.

        Returns
        -------
        UnstructuredGridDataset
            Data after reflextion is performed.
        """

        # disallow reflecting along normal direction
        if axis == self.normal_axis:
            if reflection_only:
                return self.updated_copy(
                    values=self.values * symmetry,
                    normal_pos=2 * center - self.normal_pos,
                    deep=False,
                )
            else:
                raise DataError(
                    "Reflection in the normal direction to the grid is prohibited unless 'reflection_only=True'."
                )

        tan_dims = [0, 1, 2]
        tan_dims.remove(self.normal_axis)
        tan_axis = tan_dims.index(axis)

        return super().reflect(
            axis=tan_axis, center=center, reflection_only=reflection_only, symmetry=symmetry
        )

    """ Interpolation """

    def _spatial_interp(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        z: float | ArrayLike,
        fill_value: float | Literal["extrapolate"] | None = None,
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        ignore_normal_pos: bool = True,
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
    ) -> XrDataArray:
        """Interpolate data at provided x, y, and z. Note that data is assumed to be invariant along
        the dataset's normal direction.

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
        ignore_normal_pos : bool = True
            (Depreciated) Assume data is invariant along the normal direction to the grid plane.
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

        if not ignore_normal_pos:
            log.warning(
                "Parameter 'ignore_normal_pos' is depreciated. It is always assumed that data "
                "contained in 'TriangularGridDataset' is invariant in the normal direction. "
                "That is, 'ignore_normal_pos=True' is used."
            )

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        xyz = [x, y, z]
        xyz[self.normal_axis] = [self.normal_pos]
        interp_inplane = super()._spatial_interp(
            **dict(zip("xyz", xyz)),
            fill_value=fill_value,
            use_vtk=use_vtk,
            method=method,
            max_samples_per_step=max_samples_per_step,
            max_cells_per_step=max_cells_per_step,
        )
        interp_broadcasted = np.broadcast_to(
            interp_inplane,
            [len(np.atleast_1d(comp)) for comp in [x, y, z]] + self._non_spatial_shape,
        )

        coords_dict = {"x": x, "y": y, "z": z}
        coords_dict.update(self._non_spatial_coords_dict)

        if len(self._non_spatial_coords_dict) == 0:
            return SpatialDataArray(interp_broadcasted, coords=coords_dict, name=self.values.name)
        else:
            return XrDataArray(interp_broadcasted, coords=coords_dict, name=self.values.name)

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
        """2D-specific function to interpolate data at provided x, y, and z
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

        return self._interp_py_general(
            x=x,
            y=y,
            z=z,
            fill_value=fill_value,
            max_samples_per_step=max_samples_per_step,
            max_cells_per_step=max_cells_per_step,
            rel_tol=rel_tol,
            axis_ignore=self.normal_axis,
        )

    """ Data selection """

    @requires_vtk
    def sel(
        self,
        x: float | ArrayLike = None,
        y: float | ArrayLike = None,
        z: float | ArrayLike = None,
        method: Literal["None", "nearest", "pad", "ffill", "backfill", "bfill"] | None = None,
        **sel_kwargs: Any,
    ) -> XrDataArray:
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
        xarray.DataArray
            Extracted data.
        """

        xyz = [x, y, z]
        axes = [ind for ind, comp in enumerate(xyz) if comp is not None]
        num_provided = len(axes)

        if num_provided == 0 and len(sel_kwargs) == 0:
            raise DataError("At least one dimension for selection must be provided.")

        if self.normal_axis in axes:
            if xyz[self.normal_axis] != self.normal_pos:
                raise DataError(
                    f"No data for {'xyz'[self.normal_axis]} = {xyz[self.normal_axis]} (unstructured"
                    f" grid is defined at {'xyz'[self.normal_axis]} = {self.normal_pos})."
                )

            if num_provided < 3:
                num_provided -= 1
                axes.remove(self.normal_axis)

        self_after_non_spatial_sel = self._non_spatial_sel(method=method, **sel_kwargs)

        if num_provided == 1:
            axis = axes[0]
            return self_after_non_spatial_sel.plane_slice(axis=axis, pos=xyz[axis])

        if num_provided == 2:
            pos = [x, y, z]
            pos[self.normal_axis] = [self.normal_pos]
            return self_after_non_spatial_sel.interp(x=pos[0], y=pos[1], z=pos[2])

        if num_provided == 3:
            return self_after_non_spatial_sel.interp(x=x, y=y, z=z)

        return self_after_non_spatial_sel

    @requires_vtk
    def sel_inside(self, bounds: Bound) -> TriangularGridDataset:
        """Return a new ``TriangularGridDataset`` that contains the minimal amount data necessary to
        cover a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        TriangularGridDataset
            Extracted spatial data array.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``."
            )

        # expand along normal direction
        new_bounds = [list(bounds[0]), list(bounds[1])]

        new_bounds[0][self.normal_axis] = -inf
        new_bounds[1][self.normal_axis] = inf

        return super().sel_inside(new_bounds)

    def does_cover(self, bounds: Bound) -> bool:
        """Check whether data fully covers specified by ``bounds`` spatial region. If data contains
        only one point along a given direction, then it is assumed the data is constant along that
        direction and coverage is not checked.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        bool
            Full cover check outcome.
        """

        # expand along normal direction
        new_bounds = [list(bounds[0]), list(bounds[1])]

        new_bounds[0][self.normal_axis] = self.normal_pos
        new_bounds[1][self.normal_axis] = self.normal_pos

        return super().does_cover(new_bounds)

    """ Plotting """

    @property
    def _triangulation_obj(self) -> Triangulation:
        """Matplotlib triangular representation of the grid to use in plotting."""
        from matplotlib.tri import Triangulation

        return Triangulation(self.points[:, 0], self.points[:, 1], self.cells)

    @requires_vtk
    def _decimate(
        self, target_num_cells: int, geometry_only: bool = False
    ) -> TriangularGridDataset:
        """Return a decimated copy suitable for visualization.

        Uses VTK's ``vtkDecimatePro`` to reduce the number of triangles
        while preserving the overall shape and vertex data.

        Parameters
        ----------
        target_num_cells : int
            Desired (approximate) number of cells in the output mesh.
        geometry_only : bool = False
            When ``True`` skip marshaling point data into VTK, which avoids
            copying the full value tensor for grid-only plots.

        Returns
        -------
        TriangularGridDataset
            Decimated dataset.
        """
        num_cells = len(self.cells)
        if num_cells <= target_num_cells:
            return self

        target_reduction = 1.0 - target_num_cells / num_cells

        poly = vtk["mod"].vtkPolyData()
        poly.SetPoints(self._vtk_points)
        poly.SetPolys(self._vtk_cells)

        if not geometry_only:
            data_values = self.values.values
            if self.is_complex:
                data_values = data_values.view("(2,)float")
            if len(self._non_spatial_shape) > 0:
                data_values = data_values.reshape(
                    (len(self.points.values), (1 + self.is_complex) * self._num_fields)
                )
            point_data_vtk = vtk["numpy_to_vtk"](np.array(data_values, copy=True))
            point_data_vtk.SetName(self.name)
            poly.GetPointData().SetScalars(point_data_vtk)

        decimate = vtk["mod"].vtkDecimatePro()
        decimate.SetInputData(poly)
        decimate.SetTargetReduction(target_reduction)
        decimate.PreserveTopologyOn()
        decimate.Update()

        decimated = decimate.GetOutput()
        if decimated.GetNumberOfCells() == 0:
            return self

        if geometry_only:
            dummy = vtk["numpy_to_vtk"](np.zeros(decimated.GetNumberOfPoints()))
            decimated.GetPointData().SetScalars(dummy)

        return type(self)._from_vtk_obj(
            decimated,
            field=None if geometry_only else self._non_spatial_coords_dict,
            remove_degenerate_cells=True,
            remove_unused_points=True,
            values_type=IndexedDataArray if geometry_only else self._values_type,
            expect_complex=False if geometry_only else self.is_complex,
            ignore_invalid_cells=True,
        )

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        ax: Ax = None,
        field: bool = True,
        grid: bool = True,
        cbar: bool = True,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        shading: Literal["gourand", "flat"] = "gouraud",
        cbar_kwargs: dict | None = None,
        pcolor_kwargs: dict | None = None,
        max_cells: int | None = None,
    ) -> Ax:
        """Plot the data field and/or the unstructured grid.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        field : bool = True
            Whether to plot the data field.
        grid : bool = True
            Whether to plot the unstructured grid.
        cbar : bool = True
            Display colorbar (only if ``field == True``).
        cmap : str = "viridis"
            Color map to use for plotting.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        shading : Literal["gourand", "flat"] = "gourand"
            Type of shading to use when plotting the data field.
        cbar_kwargs : Dict = {}
            Additional parameters passed to colorbar object.
        pcolor_kwargs: Dict = {}
            Additional parameters passed to ax.tripcolor()
        max_cells : int = None
            When set to a positive integer, the mesh is decimated to approximately
            this many cells before plotting. This reduces memory usage and speeds up
            rendering for large meshes that would otherwise cause out-of-memory
            errors. Requires ``vtk``.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        if cbar_kwargs is None:
            cbar_kwargs = {}
        if pcolor_kwargs is None:
            pcolor_kwargs = {}
        if not (field or grid):
            raise DataError("Nothing to plot ('field == False', 'grid == False').")
        if max_cells is not None and max_cells <= 0:
            raise DataError("'max_cells' must be a positive integer.")

        if field and self._num_fields != 1:
            raise DataError(
                "Unstructured dataset contains more than 1 field. "
                "Use '.sel()' to select a single field from available dimensions "
                f"{self._non_spatial_coords_dict} before plotting."
            )

        plot_data = self
        if max_cells is not None and len(self.cells) > max_cells:
            try:
                plot_data = self._decimate(max_cells, geometry_only=not field)
                log.info(
                    f"Decimated mesh from {len(self.cells)} to {len(plot_data.cells)} "
                    "cells for plotting."
                )
            except Exception:
                log.warning(
                    "Mesh decimation failed (vtk may not be installed). Plotting full mesh."
                )

        triangulation = plot_data._triangulation_obj

        # plot data field if requested
        if field:
            plot_obj = ax.tripcolor(
                triangulation,
                plot_data.values.data.ravel(),
                shading=shading,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **pcolor_kwargs,
            )

            if cbar:
                from matplotlib import pyplot as plt

                label_kwargs = {}
                if "label" not in cbar_kwargs:
                    label_kwargs["label"] = self.values.name
                plt.colorbar(plot_obj, **cbar_kwargs, **label_kwargs)

        # plot grid if requested
        if grid:
            ax.triplot(
                triangulation,
                color=plot_params_grid.edgecolor,
                linewidth=plot_params_grid.linewidth,
            )

        # set labels and titles
        ax_labels = ["x", "y", "z"]
        normal_axis_name = ax_labels.pop(self.normal_axis)
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_title(f"{normal_axis_name} = {self.normal_pos}")
        return ax

    def get_cell_volumes(self) -> DataArray:
        """Get areas associated to each cell of the grid."""
        v0 = self.points[self.cells.sel(vertex_index=0)]
        e01 = self.points[self.cells.sel(vertex_index=1)] - v0
        e02 = self.points[self.cells.sel(vertex_index=2)] - v0
        areas = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]
        return 0.5 * np.abs(areas)
