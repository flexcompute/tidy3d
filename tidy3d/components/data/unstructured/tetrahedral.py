"""Defines tetrahedral grid datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import CellDataArray, IndexedDataArray, PointDataArray
from tidy3d.exceptions import DataError
from tidy3d.packaging import requires_vtk, vtk

from .base import UnstructuredGridDataset
from .triangular import TriangularGridDataset

if TYPE_CHECKING:
    from typing import Literal

    from pydantic import PositiveInt
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
    from xarray import DataArray
    from xarray import DataArray as XrDataArray

    from tidy3d.components.types import ArrayLike, Ax, Axis, Bound, Coordinate


class TetrahedralGridDataset(UnstructuredGridDataset):
    """Dataset for storing tetrahedral grid data. Data values are associated with the nodes of
    the grid.

    Note
    ----
    To use full functionality of unstructured datasets one must install ``vtk`` package (``pip
    install tidy3d[vtk]`` or ``pip install vtk``). Otherwise the functionality of unstructured
    datasets is limited to creation, writing to/loading from a file, and arithmetic manipulations.

    Example
    -------
    >>> import numpy as np
    >>> from tidy3d.components.data.data_array import PointDataArray, CellDataArray, IndexedDataArray
    >>>
    >>> tet_grid_points = PointDataArray(
    ...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ...     coords=dict(index=np.arange(4), axis=np.arange(3)),
    ... )
    >>>
    >>> tet_grid_cells = CellDataArray(
    ...     [[0, 1, 2, 3]],
    ...     coords=dict(cell_index=np.arange(1), vertex_index=np.arange(4)),
    ... )
    >>>
    >>> tet_grid_values = IndexedDataArray(
    ...     [1.0, 2.0, 3.0, 4.0], coords=dict(index=np.arange(4)),
    ... )
    >>>
    >>> tet_grid = TetrahedralGridDataset(
    ...     points=tet_grid_points,
    ...     cells=tet_grid_cells,
    ...     values=tet_grid_values,
    ... )
    """

    """ Fundametal parameters to set up based on grid dimensionality """

    @classmethod
    def _triangular_dataset_type(cls) -> type:
        """Corresponding class for triangular grid datasets. We need to know this when creating a triangular slice from a tetrahedral grid."""
        return TriangularGridDataset

    @classmethod
    def _point_dims(cls) -> PositiveInt:
        """Dimensionality of stored grid point coordinates."""
        return 3

    @classmethod
    def _cell_num_vertices(cls) -> PositiveInt:
        """Number of vertices in a cell."""
        return 4

    """ Convenience properties """

    @cached_property
    def _points_3d_array(self) -> Bound:
        """3D coordinates of grid points."""
        return self.points.data

    """ VTK interfacing """

    @classmethod
    @requires_vtk
    def _vtk_cell_type(cls) -> int:
        """VTK cell type to use in the VTK representation."""
        return vtk["mod"].VTK_TETRA

    @classmethod
    def _cell_types_numpy(cls, vtk_obj: vtkUnstructuredGrid) -> np.ndarray:
        """Return cell types as a numpy array with a compatibility fallback."""
        cell_types_array = vtk_obj.GetCellTypesArray()
        if cell_types_array is not None:
            return np.array(vtk["vtk_to_numpy"](cell_types_array), copy=True)

        # VTK may return ``None`` for the consolidated array while still exposing per-cell types.
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
        expect_complex: bool = False,
        ignore_invalid_cells: bool = False,
        warn_unused_points: bool = True,
    ) -> TetrahedralGridDataset:
        """Initialize from a vtkUnstructuredGrid instance."""

        # read point, cells, and values info from a vtk instance
        cells_numpy = np.array(
            vtk["vtk_to_numpy"](vtk_obj.GetCells().GetConnectivityArray()),
            copy=True,
        )
        points_numpy = np.array(vtk["vtk_to_numpy"](vtk_obj.GetPoints().GetData()), copy=True)
        values = cls._get_values_from_vtk(
            vtk_obj, len(points_numpy), field, values_type, expect_complex
        )

        # verify cell_types
        cells_types = cls._cell_types_numpy(vtk_obj)
        invalid_cells = cells_types != cls._vtk_cell_type()
        if any(invalid_cells):
            if ignore_invalid_cells:
                cell_offsets = np.array(
                    vtk["vtk_to_numpy"](vtk_obj.GetCells().GetOffsetsArray()),
                    copy=True,
                )
                valid_cell_offsets = cell_offsets[:-1][invalid_cells == 0]
                cells_numpy = cells_numpy[
                    np.ravel(
                        valid_cell_offsets[:, None]
                        + np.arange(cls._cell_num_vertices(), dtype=int)[None, :]
                    )
                ]
            else:
                raise DataError("Only tetrahedral 'vtkUnstructuredGrid' is currently supported")

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

    """ Grid operations """

    @requires_vtk
    def plane_slice(self, axis: Axis, pos: float) -> TriangularGridDataset:
        """Slice data with a plane and return the resulting :class:`.TriangularGridDataset`.

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.

        Returns
        -------
        TriangularGridDataset
            The resulting slice.
        """

        slice_vtk = self._plane_slice_raw(axis=axis, pos=pos)

        return self._triangular_dataset_type()._from_vtk_obj(
            slice_vtk,
            remove_degenerate_cells=True,
            remove_unused_points=True,
            field=self._non_spatial_coords_dict,
            values_type=self._values_type,
            expect_complex=self.is_complex,
        )

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
        """

        bounds = self.bounds
        start = list(pos)
        end = list(pos)

        start[axis] = bounds[0][axis]
        end[axis] = bounds[1][axis]

        # create cutting plane
        line = vtk["mod"].vtkLineSource()
        line.SetPoint1(start)
        line.SetPoint2(end)
        line.SetResolution(1)

        # this should be done using vtkProbeLineFilter
        # but for some reason it crashes Python
        # so, we use a workaround:
        # 1) extract cells that are intersected by line (to speed up further slicing)
        # 2) do plane slice along first direction
        # 3) do second plane slice along second direction

        prober = vtk["mod"].vtkExtractCellsAlongPolyLine()
        prober.SetSourceConnection(line.GetOutputPort())
        prober.SetInputData(self._vtk_obj)
        prober.Update()

        extracted_cells_vtk = prober.GetOutput()

        if extracted_cells_vtk.GetNumberOfPoints() == 0:
            raise DataError("Slicing line does not intersect the unstructured grid.")

        extracted_cells = self._from_vtk_obj_internal(extracted_cells_vtk)

        tan_dims = [0, 1, 2]
        tan_dims.remove(axis)

        # first plane slice
        plane_slice = extracted_cells.plane_slice(axis=tan_dims[0], pos=pos[tan_dims[0]])
        # second plane slice
        line_slice = plane_slice.plane_slice(axis=tan_dims[1], pos=pos[tan_dims[1]])

        return line_slice

    """ Interpolation """

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
        """3D-specific function to interpolate data at provided x, y, and z
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
            axis_ignore=None,
        )

    """ Data selection """

    def sel(
        self,
        x: float | ArrayLike = None,
        y: float | ArrayLike = None,
        z: float | ArrayLike = None,
        method: Literal["nearest", "pad", "ffill", "backfill", "bfill"] | None = None,
        **sel_kwargs: Any,
    ) -> TriangularGridDataset | XrDataArray:
        """Extract/interpolate data along one or more spatial or non-spatial directions.

        Must provide at least one argument among 'x', 'y', 'z' or non-spatial dimensions
        through additional arguments. Along spatial dimensions a suitable slicing of
        grid is applied (plane slice, line slice, or interpolation). Selection along
        non-spatial dimensions is forwarded to .sel() xarray function.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinate of the slice.
        y : Union[float, ArrayLike] = None
            y-coordinate of the slice.
        z : Union[float, ArrayLike] = None
            z-coordinate of the slice.
        method : Optional[Literal["nearest", "pad", "ffill", "backfill", "bfill"]] = None
            Method to use for inexact matches (applies to non-spatial dimensions only).
        **sel_kwargs : dict
            Keyword arguments to pass to the xarray sel() function.

        Returns
        -------
        Union[TriangularGridDataset, xarray.DataArray]
            Extracted data.
        """

        xyz = [x, y, z]
        axes = [ind for ind, comp in enumerate(xyz) if comp is not None]

        num_provided = len(axes)

        if num_provided < 3 and any(not np.isscalar(comp) for comp in xyz if comp is not None):
            raise DataError(
                "Providing x, y, or z as array is only allowed for interpolation. That is, when all"
                " three x, y, and z are provided or method '.interp()' is used explicitly."
            )

        if num_provided == 0 and len(sel_kwargs) == 0:
            raise DataError(
                "Must provide at least one dimension to select along "
                "(available: {self._non_spatial_dims + list('xyz')})."
            )

        self_after_non_spatial_sel = self._non_spatial_sel(method=method, **sel_kwargs)

        if num_provided == 1:
            axis = axes[0]
            return self_after_non_spatial_sel.plane_slice(axis=axis, pos=xyz[axis])

        if num_provided == 2:
            axis = 3 - axes[0] - axes[1]
            xyz[axis] = 0
            return self_after_non_spatial_sel.line_slice(axis=axis, pos=xyz)

        if num_provided == 3:
            return self_after_non_spatial_sel.interp(x=x, y=y, z=z)

        return self_after_non_spatial_sel

    def plot(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_cells: int | None = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot a 2D slice of the tetrahedral grid data.

        Exactly one of ``x``, ``y``, or ``z`` must be provided to select the
        slicing plane.  The slice produces a :class:`.TriangularGridDataset`
        whose ``.plot()`` method is then called with the remaining keyword
        arguments.

        Parameters
        ----------
        x : float = None
            Position of the slicing plane along the x-axis.
        y : float = None
            Position of the slicing plane along the y-axis.
        z : float = None
            Position of the slicing plane along the z-axis.
        max_cells : int = None
            When set, the triangular slice is decimated to approximately this many
            cells before plotting. Forwarded to :meth:`TriangularGridDataset.plot`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`TriangularGridDataset.plot`.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        sel_kwargs = {}
        if x is not None:
            sel_kwargs["x"] = x
        if y is not None:
            sel_kwargs["y"] = y
        if z is not None:
            sel_kwargs["z"] = z

        if len(sel_kwargs) != 1:
            raise DataError(
                "Exactly one of 'x', 'y', or 'z' must be provided to select a 2D slice "
                "for plotting."
            )

        axis = list("xyz").index(next(iter(sel_kwargs)))
        tri_data = self.sel(**sel_kwargs)
        ax = tri_data.plot(max_cells=max_cells, **kwargs)

        # Clip axis limits to the original tetrahedral domain bounds so that
        # triangles straddling the boundary don't stretch the plot extent.
        in_plane = [d for d in range(3) if d != axis]
        tet_bounds = self.bounds
        ax.set_xlim(tet_bounds[0][in_plane[0]], tet_bounds[1][in_plane[0]])
        ax.set_ylim(tet_bounds[0][in_plane[1]], tet_bounds[1][in_plane[1]])
        return ax

    def get_cell_volumes(self) -> DataArray:
        """Get the volumes associated to each cell in the grid"""
        v0 = self.points[self.cells.sel(vertex_index=0)]
        e01 = self.points[self.cells.sel(vertex_index=1)] - v0
        e02 = self.points[self.cells.sel(vertex_index=2)] - v0
        e03 = self.points[self.cells.sel(vertex_index=3)] - v0

        return np.abs(np.sum(np.cross(e01, e02) * e03, axis=1)) / 6
