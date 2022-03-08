# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any
from functools import wraps

import matplotlib.pylab as plt
from pydantic import BaseModel
import plotly.graph_objects as go
import numpy as np

from .types import Ax

""" Constants """

# add this around extents of plots
PLOT_BUFFER = 0.3

ARROW_COLOR_MONITOR = "orange"
ARROW_COLOR_SOURCE = "green"
ARROW_COLOR_POLARIZATION = "brown"
ARROW_ALPHA = 0.8


# this times the min of axis height and width gives the arrow length
ARROW_LENGTH_FACTOR = 0.4

# this times ARROW_LENGTH gives width
ARROW_WIDTH_FACTOR = 0.4


""" Decorators """


def make_ax() -> Ax:
    """makes an empty `ax`."""
    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """Decorates `plot(*args, **kwargs, ax=None)` function.
    if ax=None in the function call, creates an ax and feeds it to rest of function.
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> Ax:
        """New plot function using a generated ax if None."""
        if kwargs.get("ax") is None:
            ax = make_ax()
            kwargs["ax"] = ax
        return plot(*args, **kwargs)

    return _plot


def equal_aspect(plot):
    """Decorates a plotting function returning a matplotlib axes.
    Ensures the aspect ratio of the returned axes is set to equal.
    Useful for 2D plots, like sim.plot() or sim_data.plot_fields()
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> Ax:
        """New plot function with equal aspect ratio axes returned."""
        ax = plot(*args, **kwargs)
        ax.set_aspect("equal")
        return ax

    return _plot


""" plot parameters """


class PlotParams(BaseModel):
    """Stores plotting parameters / specifications for a given model."""

    alpha: Any = None
    edgecolor: Any = None
    facecolor: Any = None
    fill: bool = True
    hatch: str = None
    lw: int = 1


# stores color of simulation.structures for given index in simulation.medium_map
MEDIUM_CMAP = [
    "#689DBC",
    "#D0698E",
    "#5E6EAD",
    "#C6224E",
    "#BDB3E2",
    "#9EC3E0",
    "#616161",
    "#877EBC",
]


""" Plotly. """

# buffer beyond sim.bounds_pml for plotting infinite shapes.
BUFFER = 100


def plotly_bounds(sim, normal_axis):
    """get X, Y limits for plotly figure."""
    rmin, rmax = sim.bounds_pml
    rmin = np.array(rmin)
    rmax = np.array(rmax)
    _, (xmin, ymin) = sim.pop_axis(rmin, axis=normal_axis)
    _, (xmax, ymax) = sim.pop_axis(rmax, axis=normal_axis)
    return (xmin, xmax), (ymin, ymax)


def process_shape(shape, sim, normal_axis):
    """Return xs, ys for given shapely shape."""
    xs, ys = shape.exterior.coords.xy
    xs = xs.tolist()
    ys = ys.tolist()
    (xmin, xmax), (ymin, ymax) = plotly_bounds(sim=sim, normal_axis=normal_axis)
    xs = [xmin - BUFFER if np.isneginf(x) else x for x in xs]
    xs = [xmax + BUFFER if np.isposinf(x) else x for x in xs]
    ys = [ymin - BUFFER if np.isneginf(y) else y for y in ys]
    ys = [ymax + BUFFER if np.isposinf(y) else y for y in ys]
    return xs, ys


def plotly_shape(
    fig, shape, sim, normal_axis, plot_params, name=""
):  # pylint:disable=too-many-arguments
    """Plot a shape to a figure."""
    xs, ys = process_shape(shape=shape, sim=sim, normal_axis=normal_axis)
    plotly_trace = go.Scatter(
        x=xs,
        y=ys,
        fill="toself",
        fillcolor=plot_params.facecolor,
        line=dict(width=plot_params.lw, color=plot_params.facecolor),
        marker=dict(size=0.0001, line=dict(width=0)),
        name=name,
        opacity=plot_params.alpha,
    )
    fig.add_trace(plotly_trace)
    return fig


def plotly_structures(fig, sim, x=None, y=None, z=None):
    """Plot all structures in simulation on plotly fig."""

    normal_axis, _ = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    for struct in sim.structures:

        shapes = struct.geometry.intersections(x=x, y=y, z=z)
        mat_index = sim.medium_map[struct.medium]
        structure_name = struct.medium.name if struct.medium.name else f"medium[{mat_index}]"

        plot_params = sim.get_structure_plot_params(medium=struct.medium)
        plot_params.lw = 4

        for shape in shapes:
            fig = plotly_shape(
                fig=fig,
                shape=shape,
                sim=sim,
                normal_axis=normal_axis,
                plot_params=plot_params,
                name=structure_name,
            )

    return fig


def plotly_sources(fig, sim, x=None, y=None, z=None):  # pylint:disable=too-many-locals
    """Plot all sources in simulation on plotly fig."""

    normal_axis, _ = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    for source_index, source in enumerate(sim.sources):

        for shape in source.geometry.intersections(x=x, y=y, z=z):
            fig = plotly_shape(
                fig=fig,
                shape=shape,
                sim=sim,
                normal_axis=normal_axis,
                plot_params=source.plot_params,
                name=source.name if source.name else f"sources[{source_index}]",
            )

    return fig


def plotly_monitors(fig, sim, x=None, y=None, z=None):
    """Plot all monitors in simulation on plotly fig."""

    normal_axis, _ = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    for monitor in sim.monitors:

        for shape in monitor.geometry.intersections(x=x, y=y, z=z):
            fig = plotly_shape(
                fig=fig,
                shape=shape,
                sim=sim,
                normal_axis=normal_axis,
                plot_params=monitor.plot_params,
                name=f'monitor: "{monitor.name}"',
            )

    return fig


def plotly_pml(fig, sim, x=None, y=None, z=None):
    """Plot all pml layers in simulation on plotly fig."""

    normal_axis, _ = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    for pml_axis, (pml, dl) in enumerate(zip(sim.pml_layers, sim.grid_size)):

        if pml is None or pml.num_layers == 0 or pml_axis == normal_axis:
            continue

        if isinstance(dl, float):
            dl_min_max = (dl, dl)
        else:
            dl_min_max = (dl[0], dl[-1])

        for sign, dl_edge in zip((-1, 1), dl_min_max):

            pml_box = sim.make_pml_box(
                pml_axis=pml_axis, pml_height=pml.num_layers * dl_edge, sign=sign
            )

            for shape in pml_box.intersections(x=x, y=y, z=z):

                fig = plotly_shape(
                    fig=fig,
                    shape=shape,
                    sim=sim,
                    normal_axis=normal_axis,
                    plot_params=pml_box.plot_params,
                    name="PML",
                )

    return fig


def plotly_symmetry(fig, sim, x=None, y=None, z=None):
    """Plot all symmertries in simulation on plotly fig."""

    normal_axis, _ = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    for sym_axis, sym_value in enumerate(sim.symmetry):

        if sym_value == 0 or sym_axis == normal_axis:
            continue

        sym_box = sim.make_symmetry_box(sym_axis=sym_axis, sym_value=sym_value)
        for shape in sym_box.intersections(x=x, y=y, z=z):
            fig = plotly_shape(
                fig=fig,
                shape=shape,
                sim=sim,
                normal_axis=normal_axis,
                plot_params=sym_box.plot_params,
                name=f"{'xyz'[sym_axis]}-axis symmetry ({('+' if sym_value > 0 else '-')}1)",
            )

    return fig


def plotly_resize(fig, sim, normal_axis, width_pixels=500):
    """Set the lmits and make equal aspect."""
    (xmin, xmax), (ymin, ymax) = plotly_bounds(sim=sim, normal_axis=normal_axis)

    fig.update_xaxes(range=[xmin, xmax])
    fig.update_yaxes(range=[ymin, ymax])

    width = xmax - xmin
    height = ymax - ymin

    fig.update_layout(width=float(width_pixels), height=float(width_pixels) * height / width)
    return fig


def remove_redundant_labels(fig):
    """Remove label entries that show up more than once."""
    seen = []
    for trace in fig["data"]:
        name = trace["name"]
        if name not in seen:
            seen.append(name)
        else:
            trace["showlegend"] = False
    return fig


def plotly_cleanup(fig, sim, x=None, y=None, z=None):
    """Finish plotting simulation cross section using plotly."""

    normal_axis, pos = sim.parse_xyz_kwargs(x=x, y=y, z=z)

    fig = plotly_resize(fig=fig, sim=sim, normal_axis=normal_axis)
    _, (xlabel, ylabel) = sim.pop_axis("xyz", axis=normal_axis)

    fig.update_layout(
        title=f'{"xyz"[normal_axis]} = {pos:.2f}',
        xaxis_title=f"{xlabel} (um)",
        yaxis_title=f"{ylabel} (um)",
        legend_title="Contents",
    )

    fig = remove_redundant_labels(fig=fig)

    return fig


def plotly_sim(sim, x=None, y=None, z=None):
    """Make a plotly plot."""

    fig = go.Figure()

    fig = plotly_structures(fig=fig, sim=sim, x=x, y=y, z=z)
    fig = plotly_sources(fig=fig, sim=sim, x=x, y=y, z=z)
    fig = plotly_monitors(fig=fig, sim=sim, x=x, y=y, z=z)
    fig = plotly_pml(fig=fig, sim=sim, x=x, y=y, z=z)
    fig = plotly_symmetry(fig=fig, sim=sim, x=x, y=y, z=z)
    fig = plotly_cleanup(fig=fig, sim=sim, x=x, y=y, z=z)

    return fig
