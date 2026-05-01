"""Defines triangular grid datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    CellDataArray,
    IndexedDataArrayTypes,
    PointDataArray,
)
from tidy3d.components.viz import add_plotter_if_none
from tidy3d.exceptions import DataError, Tidy3dNotImplementedError
from tidy3d.packaging import pyvista, requires_pyvista, requires_vtk, vtk

from .base import (
    DEFAULT_MAX_CELLS_PER_STEP,
    DEFAULT_MAX_SAMPLES_PER_STEP,
    DEFAULT_TOLERANCE_CELL_FINDING,
    UnstructuredDataset,
)

if TYPE_CHECKING:
    from typing import Any, Literal

    from pydantic import PositiveInt
    from xarray import DataArray as XrDataArray

    from tidy3d.components.types import ArrayLike, Axis


class TriangularSurfaceDataset(UnstructuredDataset):
    """Dataset for storing triangulated surface data. Data values are associated with the nodes of
    the mesh.

    Note
    ----
    To use full functionality of unstructured datasets one must install ``vtk`` package (``pip
    install tidy3d[vtk]`` or ``pip install vtk``). For visualization, ``pyvista`` is recommended
    (``pip install pyvista``). Otherwise the functionality of unstructured
    datasets is limited to creation, writing to/loading from a file, and arithmetic manipulations.

    Example
    -------
    >>> import numpy as np
    >>> from tidy3d.components.data.data_array import PointDataArray, CellDataArray, IndexedDataArray
    >>>
    >>> # Create a simple triangulated surface
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
    >>>
    >>> # Visualize the surface (change show=False to show=True to display the plot)
    >>> _ = tri_grid.plot(show=False)
    >>>
    >>> # Customize the visualization (change show=False to show=True to display the plot)
    >>> _ = tri_grid.plot(cmap='plasma', grid=True, grid_color='white', show=False)
    >>>
    >>> # For vector fields
    >>> vector_values = IndexedDataArray(
    ...     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
    ...     coords=dict(index=np.arange(4), axis=np.arange(3)),
    ... )
    >>> vector_grid = tri_grid.updated_copy(values=vector_values)
    >>>
    >>> # Plot as arrow field (change show=False to show=True to display the plot)
    >>> _ = vector_grid.quiver(scale=0.2, show=False)
    """

    points: PointDataArray = Field(
        ...,
        title="Surface Points",
        description="Coordinates of points composing the triangulated surface.",
    )

    values: IndexedDataArrayTypes = Field(
        ...,
        title="Surface Values",
        description="Values stored at the surface points.",
    )

    cells: CellDataArray = Field(
        ...,
        title="Surface Cells",
        description="Cells composing the triangulated surface specified as connections between surface "
        "points.",
    )

    """ Fundamental parameters to set up based on grid dimensionality """

    @classmethod
    def _point_dims(cls) -> PositiveInt:
        """Dimensionality of stored surface point coordinates."""
        return 3

    @classmethod
    def _cell_num_vertices(cls) -> PositiveInt:
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
    def _vtk_cell_type(cls) -> int:
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
        """Interpolate data along spatial dimensions at provided x, y, and z."""
        raise Tidy3dNotImplementedError(
            "Spatial interpolation from unstructured surfaces is not implemented yet."
        )

    """ Data selection """

    def sel(
        self,
        x: float | ArrayLike = None,
        y: float | ArrayLike = None,
        z: float | ArrayLike = None,
        method: Literal["None", "nearest", "pad", "ffill", "backfill", "bfill"] | None = None,
        **sel_kwargs: Any,
    ) -> TriangularSurfaceDataset | XrDataArray:
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

    def get_cell_volumes(self) -> XrDataArray:
        """Get areas associated to each cell of the grid."""
        v0 = self.points[self.cells.sel(vertex_index=0)]
        e01 = self.points[self.cells.sel(vertex_index=1)] - v0
        e02 = self.points[self.cells.sel(vertex_index=2)] - v0

        return 0.5 * np.linalg.norm(np.cross(e01, e02), axis=1)

    """ Plotting """

    @requires_pyvista
    @add_plotter_if_none
    def plot(
        self,
        plotter: Any = None,
        field: bool = True,
        grid: bool = False,
        cbar: bool = True,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        grid_color: str = "black",
        grid_width: float = 1.0,
        opacity: float = 1.0,
        show: bool = True,
        windowed: bool | None = None,
        window_size: tuple = (800, 600),
        **mesh_kwargs: Any,
    ) -> Any:
        """Plot the surface mesh and/or associated data using PyVista.

        Parameters
        ----------
        plotter : pyvista.Plotter = None
            PyVista plotter to add the mesh to. If not specified, one is created.
        field : bool = True
            Whether to plot the data field.
        grid : bool = False
            Whether to show the mesh edges/grid lines.
        cbar : bool = True
            Display scalar bar (only if ``field == True``).
        cmap : str = "viridis"
            Color map to use for plotting.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``,
            inferred from the data.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``,
            inferred from the data.
        grid_color : str = "black"
            Color of grid lines when ``grid == True``.
        grid_width : float = 1.0
            Width of grid lines when ``grid == True``.
        opacity : float = 1.0
            Opacity of the mesh (0=transparent, 1=opaque).
        show : bool = True
            Whether to display the plot immediately. If False, returns plotter for
            further customization.
        windowed : bool = None
            Whether to display in an external window. If None (default), automatically
            detects environment (inline in notebooks, windowed otherwise). Set to True
            to force external window even when running in a notebook, which provides
            better interactivity and performance.
        window_size : tuple = (800, 600)
            Size of the window (width, height) when creating new plotter.
        **mesh_kwargs
            Additional keyword arguments passed to plotter.add_mesh().

        Returns
        -------
        pyvista.Plotter
            The plotter object with the mesh added. Returns None if ``show=True`` and
            plotter was auto-created.

        Raises
        ------
        DataError
            If nothing to plot or if multiple fields present without selection.
        """

        if not (field or grid):
            raise DataError("Nothing to plot ('field == False', 'grid == False').")

        # Validate field selection
        if field and self._num_fields != 1:
            raise DataError(
                "Unstructured dataset contains more than 1 field. "
                "Use '.sel()' to select a single field from available dimensions "
                f"{self._non_spatial_coords_dict} before plotting."
            )

        # Check for complex values
        if field and self.is_complex:
            raise DataError(
                "Cannot plot complex-valued data directly. "
                "Please select a real component using '.real', '.imag', or '.abs' before plotting."
            )

        # Get PyVista module
        pv = pyvista["mod"]
        values_name = self.values.name if self.values.name else "values"

        # Create PyVista PolyData directly from points and cells
        # Faces format: [3, v0, v1, v2, 3, v3, v4, v5, ...] where 3 is the number of vertices
        faces = np.hstack([np.full((len(self.cells), 1), 3, dtype=int), self.cells.data])
        mesh = pv.PolyData(self.points.data, faces.ravel())

        # Setup scalar field if needed
        scalars = None
        if field:
            # Add scalar values to the mesh
            data_values = self.values.values
            if len(self._non_spatial_shape) > 0:
                data_values = data_values.reshape(len(self.points.values), self._num_fields)
            mesh[values_name] = data_values
            scalars = values_name

        # Add mesh to plotter
        plotter.add_mesh(
            mesh,
            scalars=scalars if field else None,
            cmap=cmap,
            clim=(vmin, vmax) if (vmin is not None or vmax is not None) else None,
            show_edges=grid,
            edge_color=grid_color,
            line_width=grid_width,
            opacity=opacity,
            show_scalar_bar=False,  # We'll manually add scalar bar if cbar=True
            **mesh_kwargs,
        )

        # Add scalar bar
        if field and cbar:
            scalar_bar_args = {
                "title": self.values.name or "Value",
                "n_labels": 5,
                "italic": False,
                "fmt": "%.2e",
                "font_family": "arial",
            }
            plotter.add_scalar_bar(**scalar_bar_args)

        # Set axis labels
        plotter.add_axes(xlabel="x", ylabel="y", zlabel="z")

        return plotter

    @requires_pyvista
    @add_plotter_if_none
    def quiver(
        self,
        plotter: Any = None,
        dim: str = "axis",
        scale: float = 0.1,
        downsampling: int = 1,
        color: str = "magnitude",
        cbar: bool = True,
        cmap: str = "Spectral",
        show: bool = True,
        windowed: bool | None = None,
        window_size: tuple = (800, 600),
        **arrow_kwargs: Any,
    ) -> Any:
        """Plot the associated data as vector field using arrows.

        Field ``values`` must have length 3 along the dimension representing x, y, and z components.

        Parameters
        ----------
        plotter : pyvista.Plotter = None
            PyVista plotter to add the arrows to. If not specified, one is created.
        dim : str = "axis"
            Dimension along which x, y, z components are stored.
        scale : float = 0.1
            Size of arrows relative to the diagonal length of the surface bounding box.
        downsampling : int = 1
            Step for selecting points for plotting (1 for plotting all points).
        color : str = "magnitude"
            How to color arrows. If "magnitude", colors by vector magnitude using cmap.
            Otherwise, should be a valid color string (e.g., 'red', 'blue', '#FF0000').
        cbar : bool = True
            Display scalar bar (only if ``color == "magnitude"``).
        cmap : str = "Spectral"
            Color map to use when coloring by magnitude.
        show : bool = True
            Whether to display the plot immediately.
        windowed : bool = None
            Whether to display in an external window. If None (default), automatically
            detects environment (inline in notebooks, windowed otherwise). Set to True
            to force external window.
        window_size : tuple = (800, 600)
            Size of the window (width, height) when creating new plotter.
        **arrow_kwargs
            Additional keyword arguments passed to plotter.add_mesh() for the arrow glyphs.

        Returns
        -------
        pyvista.Plotter
            The plotter object with arrows added. Returns None if ``show=True`` and
            plotter was auto-created.

        Raises
        ------
        DataError
            If dataset doesn't contain exactly 3 fields for vector components.
        """

        # Validate vector field
        if self._num_fields != 3:
            raise DataError(
                "Unstructured dataset must contain exactly 3 fields for quiver plotting. "
                "Use '.sel()' to select fields from available dimensions "
                f"{self._non_spatial_coords_dict} before plotting."
            )

        # Extract downsampled points
        points = self.points.data[::downsampling]

        # Extract vector components
        u = self.values.sel(**{dim: 0}).real.data[::downsampling]
        v = self.values.sel(**{dim: 1}).real.data[::downsampling]
        w = self.values.sel(**{dim: 2}).real.data[::downsampling]
        vectors = np.column_stack([u, v, w])

        # Compute magnitude
        mag = np.linalg.norm(vectors, axis=1)
        mag_max = np.max(mag)

        # Compute scaling factor
        bounds = np.array(self.bounds)
        size = np.subtract(bounds[1], bounds[0])
        diag = np.linalg.norm(size)
        scale_factor = scale * diag / mag_max if mag_max > 0 else scale * diag

        # Get PyVista module
        pv = pyvista["mod"]

        # Create point cloud with scaled vectors
        point_cloud = pv.PolyData(points)
        point_cloud["vectors"] = vectors

        # Create arrow glyphs
        arrows = point_cloud.glyph(
            orient="vectors",
            scale=True,  # We already scaled the vectors
            factor=scale_factor,
            geom=pv.Arrow(),
        )

        # Add magnitude data for coloring
        if color == "magnitude":
            # Each point generates multiple vertices for the arrow geometry
            # We need to repeat the magnitude for all vertices of each arrow
            n_points_per_arrow = arrows.n_points // len(points)
            arrows["magnitude"] = np.repeat(mag, n_points_per_arrow)

            plotter.add_mesh(arrows, scalars="magnitude", cmap=cmap, **arrow_kwargs)

            if cbar:
                scalar_bar_args = {
                    "title": self.values.name or "Magnitude",
                    "n_labels": 5,
                    "italic": False,
                    "fmt": "%.2e",
                    "font_family": "arial",
                }
                plotter.add_scalar_bar(**scalar_bar_args)
        else:
            plotter.add_mesh(arrows, color=color, **arrow_kwargs)

        # Set axis labels
        plotter.add_axes(xlabel="x", ylabel="y", zlabel="z")

        return plotter
