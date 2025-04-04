"""Defines triangular grid datasets."""

from __future__ import annotations

from typing import Dict, Literal, Union

import numpy as np
import pydantic.v1 as pd

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from matplotlib import colormaps
from matplotlib.colors import Normalize
from xarray import DataArray as XrDataArray

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    CellDataArray,
    IndexedDataArrayTypes,
    PointDataArray,
)
from tidy3d.components.types import ArrayLike, Ax, Axis
from tidy3d.components.viz import add_ax_3d_if_none, equal_aspect
from tidy3d.exceptions import DataError, Tidy3dNotImplementedError
from tidy3d.packaging import requires_vtk, vtk

from .base import (
    UnstructuredDataset,
)


class TriangularSurfaceDataset(UnstructuredDataset):
    """Dataset for storing triangulated surface data. Data values are associated with the nodes of
    the mesh.

    Note
    ----
    To use full functionality of unstructured datasets one must install ``vtk`` package (``pip
    install tidy3d[vtk]`` or ``pip install vtk``). Otherwise the functionality of unstructured
    datasets is limited to creation, writing to/loading from a file, and arithmetic manipulations.

    Example
    -------
    >>> tri_grid_points = PointDataArray(
    ...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ...     coords=dict(index=np.arange(4), axis=np.arange(3)),
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
    >>> tri_grid = TriangularSurfaceDataset(
    ...     points=tri_grid_points,
    ...     cells=tri_grid_cells,
    ...     values=tri_grid_values,
    ... )
    """

    points: PointDataArray = pd.Field(
        ...,
        title="Surface Points",
        description="Coordinates of points composing the triangulated surface.",
    )

    values: IndexedDataArrayTypes = pd.Field(
        ...,
        title="Surface Values",
        description="Values stored at the surface points.",
    )

    cells: CellDataArray = pd.Field(
        ...,
        title="Surface Cells",
        description="Cells composing the triangulated surface specified as connections between surface "
        "points.",
    )

    """ Fundamental parameters to set up based on grid dimensionality """

    @classmethod
    def _point_dims(cls) -> pd.PositiveInt:
        """Dimensionality of stored surface point coordinates."""
        return 3

    @classmethod
    def _cell_num_vertices(cls) -> pd.PositiveInt:
        """Number of vertices in a cell."""
        return 3

    """ Convenience properties """

    @cached_property
    def _points_3d_array(self) -> ArrayLike:
        """3D representation of points."""
        return self.points.data

    """ VTK interfacing """

    @classmethod
    @requires_vtk
    def _vtk_cell_type(cls):
        """VTK cell type to use in the VTK representation."""
        return vtk["mod"].VTK_TRIANGLE

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

        raise Tidy3dNotImplementedError("Slicing of unstructured surfaces is not implemented yet.")

    """ Data selection """

    @requires_vtk
    def sel(
        self,
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
        method: Literal["None", "nearest", "pad", "ffill", "backfill", "bfill"] = None,
        **sel_kwargs,
    ) -> Union[TriangularSurfaceDataset, XrDataArray]:
        """Extract/interpolate data along one or more spatial or non-spatial directions.
        Currently works only for non-spatial dimensions through additional arguments.
        Selection along non-spatial dimensions is forwarded to
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
        Union[TriangularSurfaceDataset, xarray.DataArray]
            Extracted data.
        """

        if any(comp is not None for comp in [x, y, z]):
            raise Tidy3dNotImplementedError(
                "Surface datasets do not support selection along x, y, or z yet."
            )

        return self._non_spatial_sel(method=method, **sel_kwargs)

    def get_cell_volumes(self):
        """Get areas associated to each cell of the grid."""
        v0 = self.points[self.cells.sel(vertex_index=0)]
        e01 = self.points[self.cells.sel(vertex_index=1)] - v0
        e02 = self.points[self.cells.sel(vertex_index=2)] - v0

        return 0.5 * np.abs(np.cross(e01, e02))

    """ Plotting """

    @equal_aspect
    @add_ax_3d_if_none
    def plot(
        self,
        ax: Ax = None,
        field: bool = True,
        grid: bool = False,
        cbar: bool = True,
        cmap: str = "viridis",
        vmin: float = None,
        vmax: float = None,
        buffer: float = 0.1,
        cbar_kwargs: Dict = None,
    ) -> Ax:
        """Plot the surface mesh and/or associated data.

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
        buffer : float = 0.1
            Padding around the surface object relative to the diagonal length of the surface bounding box.
        cbar_kwargs : Dict = {}
            Additional parameters passed to colorbar object.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        if cbar_kwargs is None:
            cbar_kwargs = {}
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

        face_colors = None
        face_alpha = 0
        edge_colors = None
        if field:
            norm = Normalize()
            # np.linalg.norm(field, axis=1)
            values_avg = np.mean(self.values.data.ravel()[self.cells.data], axis=1)
            face_colors = colormaps[cmap](norm(values_avg))
            face_alpha = 1

        if grid:
            edge_colors = "k"

        plot_obj = ax.plot_trisurf(
            self.points.data[:, 0],
            self.points.data[:, 1],
            self.points.data[:, 2],
            triangles=self.cells.data,
            fc=face_colors,
            ec=edge_colors,
            alpha=face_alpha,
            # cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if field and cbar:
            label_kwargs = {}
            if "label" not in cbar_kwargs:
                label_kwargs["label"] = self.values.name
            plt.colorbar(plot_obj, **cbar_kwargs, **label_kwargs)

        # set buffer
        if buffer is not None:
            bounds = np.array(self.bounds)
            size = np.linalg.norm(bounds[1] - bounds[0])

            ax.set_xlim(bounds[0][0] - buffer * size, bounds[1][0] + buffer * size)
            ax.set_ylim(bounds[0][1] - buffer * size, bounds[1][1] + buffer * size)
            ax.set_zlim(bounds[0][2] - buffer * size, bounds[1][2] + buffer * size)

        # set labels and titles
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.set_title(f"{normal_axis_name} = {self.normal_pos}")
        return ax

    @equal_aspect
    @add_ax_3d_if_none
    def quiver(
        self,
        ax: Ax = None,
        dim: str = "axis",
        scale: float = 0.1,
        downsampling: int = 1,
        buffer: float = 0.1,
        color: str = "magnitude",
        cbar: bool = True,
        cmap: str = "Spectral",
        cbar_kwargs: Dict = None,
        quiver_kwargs: Dict = None,
    ) -> Ax:
        """Plot the associated data as quiver plot. Field ``values`` must have length 3 along
        the dimension representing x, y, and z components.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        dim : str = "axis"
            Dimension along which .
        scale : float = 0.1
            Size of arrows relative to the diagonal lentgh of the surface boundaing box.
        downsampling : int = 1
            Step for selecting points for plotting (1 for plotting all points).
        buffer : float = 0.1
            Padding around the surface object relative to the diagonal length of the surface bounding box.
        cbar : bool = True
            Display colorbar (only if ``field == True``).
        cmap : str = "Spectral"
            Color map to use for plotting.
        cbar_kwargs : Dict = {}
            Additional parameters passed to colorbar object.
        quiver_kwargs : Dict = {}
            Additional parameters passed to quiver plot function.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        if cbar_kwargs is None:
            cbar_kwargs = {}
        if quiver_kwargs is None:
            quiver_kwargs = {}

        # plot data field if requested
        if self._num_fields != 3:
            raise DataError(
                "Unstructured dataset must contain exactly 3 fields for quiver plotting. "
                "Use '.sel()' to select a single field from available dimensions "
                f"{self._values_coords_dict} before plotting."
            )

        # compute max magnitude of vecotr field
        mag = np.sqrt(self.values.dot(self.values.conj(), dim=dim).real)
        mag_max = np.max(mag)
        # compute max diagonal of dataset
        size = np.subtract(self.bounds[1], self.bounds[0])
        diag = np.sqrt(np.sum(size * size))
        # scaling factor
        scale_factor = scale * diag / mag_max
        u = self.values.sel(**{dim: 0}).real.data[::downsampling] * scale_factor.data
        v = self.values.sel(**{dim: 1}).real.data[::downsampling] * scale_factor.data
        w = self.values.sel(**{dim: 2}).real.data[::downsampling] * scale_factor.data

        if color == "magnitude":
            clr = plt.colormaps[cmap](1 - mag.data[::downsampling].ravel() / mag_max.data)
        else:
            clr = color
        plot_obj = ax.quiver(
            self.points.sel(axis=0).data[::downsampling],
            self.points.sel(axis=1).data[::downsampling],
            self.points.sel(axis=2).data[::downsampling],
            u.ravel(),
            v.ravel(),
            w.ravel(),
            color=clr,
            **quiver_kwargs,
        )

        if color == "magnitude" and cbar:
            label_kwargs = {}
            if "label" not in cbar_kwargs:
                label_kwargs["label"] = self.values.name
            plt.colorbar(plot_obj, **cbar_kwargs, **label_kwargs)

        # set buffer
        if buffer is not None:
            bounds = np.array(self.bounds)
            size = np.linalg.norm(bounds[1] - bounds[0])

            ax.set_xlim(bounds[0][0] - buffer * size, bounds[1][0] + buffer * size)
            ax.set_ylim(bounds[0][1] - buffer * size, bounds[1][1] + buffer * size)
            ax.set_zlim(bounds[0][2] - buffer * size, bounds[1][2] + buffer * size)

        # set labels and titles
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.set_title(f"{normal_axis_name} = {self.normal_pos}")
        return ax
