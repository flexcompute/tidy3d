"""Defines tetrahedral grid datasets."""

from __future__ import annotations

from typing import Union

import numpy as np
import pydantic.v1 as pd
from xarray import DataArray as XrDataArray

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    CellDataArray,
    IndexedDataArray,
    PointDataArray,
)
from tidy3d.components.types import ArrayLike, Axis, Bound, Coordinate
from tidy3d.exceptions import DataError
from tidy3d.packaging import requires_vtk, vtk

from .base import UnstructuredGridDataset
from .triangular import TriangularGridDataset


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
    def _traingular_dataset_type(cls) -> type:
        """Corresponding class for triangular grid datasets. We need to know this when creating a triangular slice from a tetrahedral grid."""
        return TriangularGridDataset

    @classmethod
    def _point_dims(cls) -> pd.PositiveInt:
        """Dimensionality of stored grid point coordinates."""
        return 3

    @classmethod
    def _cell_num_vertices(cls) -> pd.PositiveInt:
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
    def _vtk_cell_type(cls):
        """VTK cell type to use in the VTK representation."""
        return vtk["mod"].VTK_TETRA

    @classmethod
    @requires_vtk
    def _from_vtk_obj(
        cls,
        vtk_obj,
        field=None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
        values_type=IndexedDataArray,
        expect_complex: bool = False,
    ) -> TetrahedralGridDataset:
        """Initialize from a vtkUnstructuredGrid instance."""

        # read point, cells, and values info from a vtk instance
        cells_numpy = vtk["vtk_to_numpy"](vtk_obj.GetCells().GetConnectivityArray())
        points_numpy = vtk["vtk_to_numpy"](vtk_obj.GetPoints().GetData())
        values = cls._get_values_from_vtk(
            vtk_obj, len(points_numpy), field, values_type, expect_complex
        )

        # verify cell_types
        cells_types = vtk["vtk_to_numpy"](vtk_obj.GetCellTypesArray())
        if not np.all(cells_types == cls._vtk_cell_type()):
            raise DataError("Only tetrahedral 'vtkUnstructuredGrid' is currently supported")

        # pack point and cell information into Tidy3D arrays
        num_cells = len(cells_numpy) // cls._cell_num_vertices()
        cells_numpy = np.reshape(cells_numpy, (num_cells, cls._cell_num_vertices()))

        cells = CellDataArray(
            cells_numpy,
            coords=dict(
                cell_index=np.arange(num_cells), vertex_index=np.arange(cls._cell_num_vertices())
            ),
        )

        points = PointDataArray(
            points_numpy,
            coords=dict(index=np.arange(len(points_numpy)), axis=np.arange(cls._point_dims())),
        )

        if remove_degenerate_cells:
            cells = cls._remove_degenerate_cells(cells=cells)

        if remove_unused_points:
            points, values, cells = cls._remove_unused_points(
                points=points, values=values, cells=cells
            )

        return cls(points=points, cells=cells, values=values)

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

        return self._traingular_dataset_type()._from_vtk_obj(
            slice_vtk,
            remove_degenerate_cells=True,
            remove_unused_points=True,
            field=self._values_coords_dict,
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

    @requires_vtk
    def sel(
        self,
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
        method=None,
        **sel_kwargs,
    ) -> Union[TriangularGridDataset, XrDataArray]:
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

    def get_cell_volumes(self):
        """Get the volumes associated to each cell in the grid"""
        v0 = self.points[self.cells.sel(vertex_index=0)]
        e01 = self.points[self.cells.sel(vertex_index=1)] - v0
        e02 = self.points[self.cells.sel(vertex_index=2)] - v0
        e03 = self.points[self.cells.sel(vertex_index=3)] - v0

        return np.abs(np.sum(np.cross(e01, e02) * e03, axis=1)) / 6
