"""Defines triangular grid datasets."""

from __future__ import annotations

from typing import Dict, Literal, Union

import numpy as np
import pydantic.v1 as pd

try:
    from matplotlib import pyplot as plt
    from matplotlib.tri import Triangulation
except ImportError:
    pass

from xarray import DataArray as XrDataArray

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    CellDataArray,
    IndexedDataArray,
    PointDataArray,
    SpatialDataArray,
)
from tidy3d.components.types import ArrayLike, Ax, Axis, Bound
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
)


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

    normal_axis: Axis = pd.Field(
        ...,
        title="Grid Axis",
        description="Orientation of the grid.",
    )

    normal_pos: float = pd.Field(
        ...,
        title="Position",
        description="Coordinate of the grid along the normal direction.",
    )

    """ Fundamental parameters to set up based on grid dimensionality """

    @classmethod
    def _point_dims(cls) -> pd.PositiveInt:
        """Dimensionality of stored grid point coordinates."""
        return 2

    @classmethod
    def _cell_num_vertices(cls) -> pd.PositiveInt:
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
    def _vtk_cell_type(cls):
        """VTK cell type to use in the VTK representation."""
        return vtk["mod"].VTK_TRIANGLE

    @classmethod
    @requires_vtk
    def _from_vtk_obj(
        cls,
        vtk_obj,
        field=None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
        values_type=IndexedDataArray,
        expect_complex=None,
    ):
        """Initialize from a vtkUnstructuredGrid instance."""

        # get points cells data from vtk object
        if isinstance(vtk_obj, vtk["mod"].vtkPolyData):
            cells_vtk = vtk_obj.GetPolys()
        elif isinstance(vtk_obj, vtk["mod"].vtkUnstructuredGrid):
            cells_vtk = vtk_obj.GetCells()

        cells_numpy = vtk["vtk_to_numpy"](cells_vtk.GetConnectivityArray())

        cell_offsets = vtk["vtk_to_numpy"](cells_vtk.GetOffsetsArray())
        if not np.all(np.diff(cell_offsets) == cls._cell_num_vertices()):
            raise DataError(
                "Only triangular 'vtkUnstructuredGrid' or 'vtkPolyData' can be converted into "
                "'TriangularGridDataset'."
            )

        points_numpy = vtk["vtk_to_numpy"](vtk_obj.GetPoints().GetData())

        # data values are read directly into Tidy3D array
        values = cls._get_values_from_vtk(
            vtk_obj, len(points_numpy), field, values_type, expect_complex
        )

        # detect zero size dimension
        bounds = np.max(points_numpy, axis=0) - np.min(points_numpy, axis=0)
        zero_dims = np.where(np.isclose(bounds, 0))[0]

        if len(zero_dims) != 1:
            raise DataError(
                f"Provided vtk grid does not represent a two dimensional grid. Found zero size dimensions are {zero_dims}."
            )

        normal_axis = zero_dims[0]
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
            coords=dict(
                cell_index=np.arange(num_cells), vertex_index=np.arange(cls._cell_num_vertices())
            ),
        )

        points = PointDataArray(
            points_2d_numpy,
            coords=dict(index=np.arange(len(points_numpy)), axis=np.arange(cls._point_dims())),
        )

        if remove_degenerate_cells:
            cells = cls._remove_degenerate_cells(cells=cells)

        if remove_unused_points:
            points, values, cells = cls._remove_unused_points(
                points=points, values=values, cells=cells
            )

        return cls(
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
        points_numpy = vtk["vtk_to_numpy"](slice_vtk.GetPoints().GetData())
        values = self._get_values_from_vtk(
            slice_vtk,
            len(points_numpy),
            field=self._values_coords_dict,
            values_type=self._values_type,
            expect_complex=self.is_complex,
        )

        # axis of the resulting line
        slice_axis = 3 - self.normal_axis - axis

        # assemble coords for DataArray
        coords = [None, None, None]
        coords[axis] = [pos]
        coords[self.normal_axis] = [self.normal_pos]
        coords[slice_axis] = points_numpy[:, slice_axis]
        coords_dict = dict(zip("xyz", coords))
        coords_dict.update(self._values_coords_dict)

        # reshape values from a 1d array into a 3d array
        new_shape = [1, 1, 1]
        new_shape[slice_axis] = len(values.index)
        new_shape = new_shape + list(np.shape(values.data))[1:]
        values_reshaped = np.reshape(values.data, new_shape)

        return XrDataArray(values_reshaped, coords=coords_dict, name=self.values.name).sortby(
            "xyz"[slice_axis]
        )

    @requires_vtk
    def reflect(
        self, axis: Axis, center: float, reflection_only: bool = False
    ) -> UnstructuredGridDataset:
        """Reflect unstructured data across the plane define by parameters ``axis`` and ``center``.
        By default the original data is preserved, setting ``reflection_only`` to ``True`` will
        produce only deflected data.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected data.

        Returns
        -------
        UnstructuredGridDataset
            Data after reflextion is performed.
        """

        # disallow reflecting along normal direction
        if axis == self.normal_axis:
            if reflection_only:
                return self.updated_copy(normal_pos=2 * center - self.normal_pos)
            else:
                raise DataError(
                    "Reflection in the normal direction to the grid is prohibited unless 'reflection_only=True'."
                )

        return super().reflect(axis=axis, center=center, reflection_only=reflection_only)

    """ Interpolation """

    def _spatial_interp(
        self,
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        z: Union[float, ArrayLike],
        fill_value: Union[float, Literal["extrapolate"]] = None,
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
            interp_inplane, [len(np.atleast_1d(comp)) for comp in [x, y, z]] + self._fields_shape
        )

        coords_dict = dict(x=x, y=y, z=z)
        coords_dict.update(self._values_coords_dict)

        if len(self._values_coords_dict) == 0:
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
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
        method: Literal["None", "nearest", "pad", "ffill", "backfill", "bfill"] = None,
        **sel_kwargs,
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

        if self.normal_axis in axes:
            if xyz[self.normal_axis] != self.normal_pos:
                raise DataError(
                    f"No data for {'xyz'[self.normal_axis]} = {xyz[self.normal_axis]} (unstructured"
                    f" grid is defined at {'xyz'[self.normal_axis]} = {self.normal_pos})."
                )

            if num_provided < 3:
                num_provided -= 1
                axes.remove(self.normal_axis)

        if num_provided == 0 and len(sel_kwargs) == 0:
            raise DataError("At least one dimension for selection must be provided.")

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
        return Triangulation(self.points[:, 0], self.points[:, 1], self.cells)

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        ax: Ax = None,
        field: bool = True,
        grid: bool = True,
        cbar: bool = True,
        cmap: str = "viridis",
        vmin: float = None,
        vmax: float = None,
        shading: Literal["gourand", "flat"] = "gouraud",
        cbar_kwargs: Dict = None,
        pcolor_kwargs: Dict = None,
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

        # plot data field if requested
        if field:
            if self._num_fields != 1:
                raise DataError(
                    "Unstructured dataset contains more than 1 field. "
                    "Use '.sel()' to select a single field from available dimensions "
                    f"{self._values_coords_dict} before plotting."
                )
            plot_obj = ax.tripcolor(
                self._triangulation_obj,
                self.values.data.ravel(),
                shading=shading,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **pcolor_kwargs,
            )

            if cbar:
                label_kwargs = {}
                if "label" not in cbar_kwargs:
                    label_kwargs["label"] = self.values.name
                plt.colorbar(plot_obj, **cbar_kwargs, **label_kwargs)

        # plot grid if requested
        if grid:
            ax.triplot(
                self._triangulation_obj,
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

    def get_cell_volumes(self):
        """Get areas associated to each cell of the grid."""
        v0 = self.points[self.cells.sel(vertex_index=0)]
        e01 = self.points[self.cells.sel(vertex_index=1)] - v0
        e02 = self.points[self.cells.sel(vertex_index=2)] - v0

        return 0.5 * np.abs(np.cross(e01, e02))
