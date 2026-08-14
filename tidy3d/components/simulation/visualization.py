"""Two- and three-dimensional plotting helpers for Yee-grid simulations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, get_args

import autograd.numpy as np

from tidy3d.components.boundary import (
    ABCBoundary,
    BlochBoundary,
    ModeABCBoundary,
    PECBoundary,
    PMCBoundary,
)
from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.scene import Scene
from tidy3d.components.structure import Structure
from tidy3d.components.types import PermittivityComponent
from tidy3d.components.viz import (
    PlotParams,
    add_ax_if_none,
    equal_aspect,
    plot_params_abc,
    plot_params_bloch,
    plot_params_min_grid_size,
    plot_params_override_structures,
    plot_params_pec,
    plot_params_pmc,
    plot_params_pml,
    plot_sim_3d,
)
from tidy3d.log import log

if TYPE_CHECKING:
    from typing import Literal

    from tidy3d.components.boundary import BoundaryEdgeType
    from tidy3d.components.types import Ax


@equal_aspect
@add_ax_if_none
def plot_absorbers(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    hlim: tuple[float, float] | None = None,
    vlim: tuple[float, float] | None = None,
    alpha: float | None = None,
    ax: Ax = None,
    shifted: bool = False,
) -> Ax:
    """Plot each of simulation's port absorbers on a plane defined by one nonzero x,y,z coordinate.

    Parameters
    ----------
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane.
    hlim : Tuple[float, float] = None
        The x range if plotting on xy or xz planes, y range if plotting on yz plane.
    vlim : Tuple[float, float] = None
        The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
    alpha : float = None
        Opacity of the absorbers, If ``None`` uses Tidy3d default.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.
    """
    bounds = self.bounds
    absorbers_to_plot = self._shifted_internal_absorbers if shifted else self.internal_absorbers
    for absorber in absorbers_to_plot:
        ax = absorber.plot(x=x, y=y, z=z, alpha=alpha, ax=ax, sim_bounds=bounds)
    ax = Scene._set_plot_bounds(
        bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )
    # Add the default axis labels, tick labels, and title
    ax = Box.add_ax_labels_and_title(ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units)
    return ax


@equal_aspect
@add_ax_if_none
def plot(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    ax: Ax = None,
    source_alpha: float | None = None,
    monitor_alpha: float | None = None,
    lumped_element_alpha: float | None = None,
    absorber_alpha: float | None = None,
    absorber_actual_placement: bool = False,
    hlim: tuple[float, float] | None = None,
    vlim: tuple[float, float] | None = None,
    fill_structures: bool = True,
    **patch_kwargs: Any,
) -> Ax:
    """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

    Parameters
    ----------
    fill_structures : bool = True
        Whether to fill structures with color or just draw outlines.
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane.
    source_alpha : float = None
        Opacity of the sources. If ``None``, uses Tidy3d default.
    monitor_alpha : float = None
        Opacity of the monitors. If ``None``, uses Tidy3d default.
    lumped_element_alpha : float = None
        Opacity of the lumped elements. If ``None``, uses Tidy3d default.
    absorber_alpha : float = None
        Opacity of the port absorbers. If ``None``, uses Tidy3d default.
    absorber_actual_placement : bool = False
        Use the exact placement of port absorbers which take into account their ``shift`` values.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.
    hlim : tuple[float, float] = None
        The x range if plotting on xy or xz planes, y range if plotting on yz plane.
    vlim : tuple[float, float] = None
        The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.

    See Also
    ---------

    **Notebooks**
        * `Visualizing geometries in Tidy3D: Plotting Materials <../../notebooks/VizSimulation.html#Plotting-Materials>`_

    """
    hlim, vlim = Scene._get_plot_lims(
        bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )

    ax = self.scene.plot(
        x=x,
        y=y,
        z=z,
        ax=ax,
        hlim=hlim,
        vlim=vlim,
        fill_structures=fill_structures,
    )

    ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
    ax = self.plot_absorbers(
        ax=ax,
        x=x,
        y=y,
        z=z,
        hlim=hlim,
        vlim=vlim,
        alpha=absorber_alpha,
        shifted=absorber_actual_placement,
    )
    ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
    ax = self.plot_lumped_elements(
        ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=lumped_element_alpha
    )
    ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
    ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
    ax = Scene._set_plot_bounds(
        bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )
    ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)

    return ax


@equal_aspect
@add_ax_if_none
def plot_eps(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    freq: float | None = None,
    alpha: float | None = None,
    source_alpha: float | None = None,
    monitor_alpha: float | None = None,
    lumped_element_alpha: float | None = None,
    absorber_alpha: float | None = None,
    absorber_actual_placement: bool = False,
    hlim: tuple[float, float] | None = None,
    vlim: tuple[float, float] | None = None,
    ax: Ax = None,
    eps_component: PermittivityComponent | None = None,
    eps_lim: tuple[float | None, float | None] = (None, None),
) -> Ax:
    """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
    The permittivity is plotted in grayscale based on its value at the specified frequency.

    Parameters
    ----------
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane.
    freq : float = None
        Frequency to evaluate the relative permittivity of all mediums.
        If not specified, the central frequency of sources in the simulation will be used.
        If sources have different central frequencies, the relative permittivity will be evaluated
        at infinite frequency.
    alpha : float = None
        Opacity of the structures being plotted.
        Defaults to the structure default alpha.
    source_alpha : float = None
        Opacity of the sources. If ``None``, uses Tidy3d default.
    monitor_alpha : float = None
        Opacity of the monitors. If ``None``, uses Tidy3d default.
    lumped_element_alpha : float = None
        Opacity of the lumped elements. If ``None``, uses Tidy3d default.
    absorber_alpha : float = None
        Opacity of the port absorbers. If ``None``, uses Tidy3d default.
    absorber_actual_placement : bool = False
        Use the exact placement of port absorbers which take into account their ``shift`` values.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.
    hlim : tuple[float, float] = None
        The x range if plotting on xy or xz planes, y range if plotting on yz plane.
    vlim : tuple[float, float] = None
        The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
    eps_component : Optional[PermittivityComponent] = None
        Component of the permittivity tensor to plot for anisotropic materials,
        e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
        Defaults to ``None``, which returns the average of the diagonal values.
    eps_lim : Tuple[float, float] = None
        Custom limits for eps coloring.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.

    See Also
    ---------

    **Notebooks**
        * `Visualizing geometries in Tidy3D: Plotting Permittivity <../../notebooks/VizSimulation.html#Plotting-Permittivity>`_
    """

    # check that eps_component is one of the allowed values, otherwise raise an error
    if eps_component is not None:
        if eps_component not in get_args(PermittivityComponent):
            raise ValueError(
                f"eps_component '{eps_component}' is not supported. "
                "eps_component must be one of the following values:"
                "'xx', 'yy', 'zz', 'xy', 'yx', 'xz', 'zx', 'yz', 'zy', or 'None'"
            )

    hlim, vlim = Scene._get_plot_lims(
        bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )

    ax = self.plot_structures_eps(
        freq=freq,
        cbar=True,
        alpha=alpha,
        ax=ax,
        x=x,
        y=y,
        z=z,
        hlim=hlim,
        vlim=vlim,
        eps_component=eps_component,
        eps_lim=eps_lim,
    )
    ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
    ax = self.plot_absorbers(
        ax=ax,
        x=x,
        y=y,
        z=z,
        hlim=hlim,
        vlim=vlim,
        alpha=absorber_alpha,
        shifted=absorber_actual_placement,
    )
    ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
    ax = self.plot_lumped_elements(
        ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=lumped_element_alpha
    )
    ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
    ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
    ax = Scene._set_plot_bounds(
        bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )
    ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
    return ax


@equal_aspect
@add_ax_if_none
def plot_structures_eps(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    freq: float | None = None,
    alpha: float | None = None,
    cbar: bool = True,
    reverse: bool = False,
    ax: Ax = None,
    hlim: tuple[float, float] | None = None,
    vlim: tuple[float, float] | None = None,
    eps_component: PermittivityComponent | None = None,
    eps_lim: tuple[float | None, float | None] = (None, None),
) -> Ax:
    """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
    The permittivity is plotted in grayscale based on its value at the specified frequency.

    Parameters
    ----------
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane.
    freq : float = None
        Frequency to evaluate the relative permittivity of all mediums.
        If not specified, the central frequency of sources in the simulation will be used.
        If sources have different central frequencies, the relative permittivity will be evaluated
        at infinite frequency.
    reverse : bool = False
        If ``False``, the highest permittivity is plotted in black.
        If ``True``, it is plotteed in white (suitable for black backgrounds).
    cbar : bool = True
        Whether to plot a colorbar for the relative permittivity.
    alpha : float = None
        Opacity of the structures being plotted.
        Defaults to the structure default alpha.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.
    hlim : tuple[float, float] = None
        The x range if plotting on xy or xz planes, y range if plotting on yz plane.
    vlim : tuple[float, float] = None
        The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
    eps_component : Optional[PermittivityComponent] = None
        Component of the permittivity tensor to plot for anisotropic materials,
        e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
        Defaults to ``None``, which returns the average of the diagonal values.
    eps_lim : Tuple[float, float] = None
        Custom limits for eps coloring.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.
    """

    hlim, vlim = Scene._get_plot_lims(
        bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )
    if freq is None:
        freq0s = [source.source_time._freq0 for source in self.sources]
        if freq0s and all(math.isclose(freq0, freq0s[0]) for freq0 in freq0s):
            freq = freq0s[0]
        else:
            freq = np.inf
            log.warning(
                "An appropriate frequency could not be determined when plotting the permittivity. "
                "The permittivity will be evaluated at infinite frequency. Please supply a value "
                "for `freq` to plot at a finite frequency. ",
                capture=False,
            )
    return self.scene.plot_structures_eps(
        freq=freq,
        cbar=cbar,
        alpha=alpha,
        ax=ax,
        x=x,
        y=y,
        z=z,
        hlim=hlim,
        vlim=vlim,
        grid=self.grid,
        reverse=reverse,
        eps_component=eps_component,
        eps_lim=eps_lim,
    )


@equal_aspect
@add_ax_if_none
def plot_pml(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    hlim: tuple[float, float] | None = None,
    vlim: tuple[float, float] | None = None,
    ax: Ax = None,
) -> Ax:
    """Plot each of simulation's absorbing boundaries
    on a plane defined by one nonzero x,y,z coordinate.

    Parameters
    ----------
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane
    hlim : tuple[float, float] = None
        The x range if plotting on xy or xz planes, y range if plotting on yz plane.
    vlim : tuple[float, float] = None
        The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.
    """
    normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
    pml_boxes = self._make_pml_boxes(normal_axis=normal_axis)
    for pml_box in pml_boxes:
        pml_box.plot(x=x, y=y, z=z, ax=ax, **plot_params_pml.to_kwargs())
    ax = Scene._set_plot_bounds(
        bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )
    # Add the default axis labels, tick labels, and title
    ax = Box.add_ax_labels_and_title(ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units)
    return ax


@equal_aspect
@add_ax_if_none
def plot_lumped_elements(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    hlim: tuple[float, float] | None = None,
    vlim: tuple[float, float] | None = None,
    alpha: float | None = None,
    ax: Ax = None,
) -> Ax:
    """Plot each of simulation's lumped elements on a plane defined by one
    nonzero x,y,z coordinate.

    Parameters
    ----------
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane.
    hlim : tuple[float, float] = None
        The x range if plotting on xy or xz planes, y range if plotting on yz plane.
    vlim : tuple[float, float] = None
        The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
    alpha : float = None
        Opacity of the lumped element, If ``None`` uses Tidy3d default.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.
    """
    bounds = self.bounds
    for element in self.lumped_elements:
        kwargs = element.plot_params.include_kwargs(alpha=alpha).to_kwargs()
        ax = element.to_geometry().plot(x=x, y=y, z=z, ax=ax, sim_bounds=bounds, **kwargs)
    ax = Scene._set_plot_bounds(
        bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )
    return ax


@add_ax_if_none
def plot_grid(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    ax: Ax = None,
    hlim: tuple[float, float] | None = None,
    vlim: tuple[float, float] | None = None,
    override_structures_alpha: float = 1,
    snapping_points_alpha: float = 1,
    finest_grid_region_alpha: float = 0,
    **kwargs: Any,
) -> Ax:
    """Plot the cell boundaries as lines on a plane defined by one nonzero x,y,z coordinate.

    Parameters
    ----------
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane.
    hlim : tuple[float, float] = None
        The x range if plotting on xy or xz planes, y range if plotting on yz plane.
    vlim : tuple[float, float] = None
        The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
    override_structures_alpha : float = 1
        Opacity of the override structures.
    snapping_points_alpha : float = 1
        Opacity of the snapping points.
    finest_grid_region_alpha : float = 0
        Opacity of the shaded regions highlighting finest grid regions. Defaults to ``0``
        (off); pass a nonzero value to opt in to drawing these regions.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.
    **kwargs
        Optional keyword arguments passed to the matplotlib ``LineCollection``.
        For details on accepted values, refer to
        `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.
    """
    import matplotlib as mpl
    from matplotlib.collections import PatchCollection

    kwargs.setdefault("linewidth", 0.2)
    kwargs.setdefault("colors", "black")
    kwargs.setdefault("colors_internal", "darkmagenta")
    kwargs.setdefault("dashes", (10, 10))
    kwargs.setdefault("override_linestyle", ":")
    kwargs.setdefault("snapping_linestyle", "--")
    cell_boundaries = self.grid.boundaries
    axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
    _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
    boundaries_x = cell_boundaries.model_dump()["xyz"[axis_x]]
    boundaries_y = cell_boundaries.model_dump()["xyz"[axis_y]]

    if self.size[axis_x] > 0:
        for b in boundaries_x:
            ax.axvline(x=b, linewidth=kwargs["linewidth"], color=kwargs["colors"])

    if self.size[axis_y] > 0:
        for b in boundaries_y:
            ax.axhline(y=b, linewidth=kwargs["linewidth"], color=kwargs["colors"])

    # Plot bounding boxes of override structures
    plot_params = [
        plot_params_override_structures.include_kwargs(
            linewidth=4 * kwargs["linewidth"],
            edgecolor=kwargs["colors"],
            alpha=override_structures_alpha,
        ),
    ] * 3
    plot_params[0] = plot_params[0].include_kwargs(edgecolor=kwargs["colors_internal"])

    if self.grid_spec.auto_grid_used:
        # Internal and external override structures are visualized with different colors,
        # so let's not sort them together.
        all_override_structures = [
            Structure._sort_structures(structures, self.scene.structure_priority_mode)
            for structures in [
                self.internal_override_structures,
                self.grid_spec.external_override_structures,
            ]
        ]

        for structures, plot_param in zip(all_override_structures, plot_params):
            rects = []
            for structure in structures:
                bounds = list(zip(*structure.geometry.bounds))
                _, ((xmin, xmax), (ymin, ymax)) = structure.geometry.pop_axis(bounds, axis=axis)
                xmin, xmax, ymin, ymax = (self._evaluate_inf(v) for v in (xmin, xmax, ymin, ymax))
                rects.append(
                    mpl.patches.Rectangle(
                        xy=(xmin, ymin),
                        width=(xmax - xmin),
                        height=(ymax - ymin),
                    )
                )
            if rects:
                pc_kwargs = plot_param.to_kwargs()
                if not pc_kwargs.pop("fill", True):
                    pc_kwargs["facecolor"] = "none"
                pc = PatchCollection(
                    rects,
                    linestyle=kwargs["override_linestyle"],
                    **pc_kwargs,
                )
                ax.add_collection(pc)

    # Plot snapping points
    for points, plot_param in zip(
        [
            self.internal_snapping_points,
            self.grid_spec.snapping_points,
            self._gap_meshing_snapping_lines,
        ],
        plot_params,
    ):
        scatter_xs = []
        scatter_ys = []
        for point in points:
            _, (x_point, y_point) = Geometry.pop_axis(point, axis=axis)
            if x_point is None and y_point is None:
                continue
            if x_point is None:
                ax.axhline(
                    y=self._evaluate_inf(y_point),
                    linewidth=4 * kwargs["linewidth"],
                    color=plot_param.edgecolor,
                    alpha=snapping_points_alpha,
                    linestyle=kwargs["snapping_linestyle"],
                    dashes=kwargs["dashes"],
                )
                continue
            if y_point is None:
                ax.axvline(
                    x=self._evaluate_inf(x_point),
                    linewidth=4 * kwargs["linewidth"],
                    color=plot_param.edgecolor,
                    alpha=snapping_points_alpha,
                    linestyle=kwargs["snapping_linestyle"],
                    dashes=kwargs["dashes"],
                )
                continue
            scatter_xs.append(self._evaluate_inf(x_point))
            scatter_ys.append(self._evaluate_inf(y_point))
        if scatter_xs:
            ax.scatter(
                scatter_xs, scatter_ys, color=plot_param.edgecolor, alpha=snapping_points_alpha
            )

    ax = Scene._set_plot_bounds(
        bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
    )

    # Plot shaded regions for minimal grid cell sizes
    if finest_grid_region_alpha > 0:
        min_size_locs = self.grid.fine_mesh_info
        dim_names = ["x", "y", "z"]
        dim_x = dim_names[axis_x]
        dim_y = dim_names[axis_y]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        plot_params = plot_params_min_grid_size.include_kwargs(alpha=finest_grid_region_alpha)

        for (dim, location), size in min_size_locs.items():
            # Only plot patches for dimensions in the current plane
            if dim == dim_x:
                # Vertical patch (constant x)
                rect = mpl.patches.Rectangle(
                    xy=(location - size / 2, ylim[0]),
                    width=size,
                    height=ylim[1] - ylim[0],
                    **plot_params.to_kwargs(),
                )
                ax.add_patch(rect)
            elif dim == dim_y:
                # Horizontal patch (constant y)
                rect = mpl.patches.Rectangle(
                    xy=(xlim[0], location - size / 2),
                    width=xlim[1] - xlim[0],
                    height=size,
                    **plot_params.to_kwargs(),
                )
                ax.add_patch(rect)

    # Add the default axis labels, tick labels, and title
    ax = Box.add_ax_labels_and_title(ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units)
    return ax


@equal_aspect
@add_ax_if_none
def plot_boundaries(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    ax: Ax = None,
    **kwargs: Any,
) -> Ax:
    """Plot the simulation boundary conditions as lines on a plane
       defined by one nonzero x,y,z coordinate.

    Parameters
    ----------
    x : float = None
        position of plane in x direction, only one of x, y, z must be specified to define plane.
    y : float = None
        position of plane in y direction, only one of x, y, z must be specified to define plane.
    z : float = None
        position of plane in z direction, only one of x, y, z must be specified to define plane.
    ax : matplotlib.axes._subplots.Axes = None
        Matplotlib axes to plot on, if not specified, one is created.
    **kwargs
        Optional keyword arguments passed to the matplotlib ``LineCollection``.
        For details on accepted values, refer to
        `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

    Returns
    -------
    matplotlib.axes._subplots.Axes
        The supplied or created matplotlib axes.
    """
    import matplotlib as mpl

    def set_plot_params(
        boundary_edge: ABCBoundary | ModeABCBoundary | BoundaryEdgeType,
        lim: float,
        side: Literal[-1, 1],
        thickness: float,
    ) -> tuple[PlotParams, float]:
        """Return the line plot properties such as color and opacity based on the boundary"""
        if isinstance(boundary_edge, PECBoundary):
            plot_params = plot_params_pec.copy(deep=True)
        elif isinstance(boundary_edge, PMCBoundary):
            plot_params = plot_params_pmc.copy(deep=True)
        elif isinstance(boundary_edge, BlochBoundary):
            plot_params = plot_params_bloch.copy(deep=True)
        elif isinstance(boundary_edge, ABCBoundary | ModeABCBoundary):
            plot_params = plot_params_abc.copy(deep=True)
        else:
            plot_params = PlotParams(alpha=0)

        # expand axis limit so that the axis ticks and labels aren't covered
        new_lim = lim
        if plot_params.alpha != 0:
            if side == -1:
                new_lim = lim - thickness
            elif side == 1:
                new_lim = lim + thickness

        return plot_params, new_lim

    boundaries = self.boundary_spec.to_list

    normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
    _, (dim_u, dim_v) = self.pop_axis([0, 1, 2], axis=normal_axis)

    umin, umax = ax.get_xlim()
    vmin, vmax = ax.get_ylim()

    size_factor = 1.0 / 35.0
    thickness_u = (umax - umin) * size_factor
    thickness_v = (vmax - vmin) * size_factor

    # boundary along the u axis, minus side
    plot_params, ulim_minus = set_plot_params(boundaries[dim_u][0], umin, -1, thickness_u)
    rect = mpl.patches.Rectangle(
        xy=(umin - thickness_u, vmin),
        width=thickness_u,
        height=(vmax - vmin),
        **plot_params.to_kwargs(),
        **kwargs,
    )
    ax.add_patch(rect)

    # boundary along the u axis, plus side
    plot_params, ulim_plus = set_plot_params(boundaries[dim_u][1], umax, 1, thickness_u)
    rect = mpl.patches.Rectangle(
        xy=(umax, vmin),
        width=thickness_u,
        height=(vmax - vmin),
        **plot_params.to_kwargs(),
        **kwargs,
    )
    ax.add_patch(rect)

    # boundary along the v axis, minus side
    plot_params, vlim_minus = set_plot_params(boundaries[dim_v][0], vmin, -1, thickness_v)
    rect = mpl.patches.Rectangle(
        xy=(umin, vmin - thickness_v),
        width=(umax - umin),
        height=thickness_v,
        **plot_params.to_kwargs(),
        **kwargs,
    )
    ax.add_patch(rect)

    # boundary along the v axis, plus side
    plot_params, vlim_plus = set_plot_params(boundaries[dim_v][1], vmax, 1, thickness_v)
    rect = mpl.patches.Rectangle(
        xy=(umin, vmax),
        width=(umax - umin),
        height=thickness_v,
        **plot_params.to_kwargs(),
        **kwargs,
    )
    ax.add_patch(rect)

    # ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
    ax.set_xlim([ulim_minus, ulim_plus])
    ax.set_ylim([vlim_minus, vlim_plus])
    # Add the default axis labels, tick labels, and title
    ax = Box.add_ax_labels_and_title(ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units)
    return ax


def plot_3d(self: Any, width: int = 800, height: int = 800) -> None:
    """Render 3D plot of ``Simulation`` (in jupyter notebook only).
    Parameters
    ----------
    width : float = 800
        width of the 3d view dom's size
    height : float = 800
        height of the 3d view dom's size

    """
    return plot_sim_3d(self, width=width, height=height)
