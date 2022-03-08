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

# pylint:disable=too-many-locals, too-many-branches, too-many-statements
def plotly_sim(sim, x=None, y=None, z=None):
    """Make a plotly plot."""

    fig = go.Figure()
    axis, pos = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    rmin, rmax = sim.bounds_pml
    rmin = np.array(rmin)
    rmax = np.array(rmax)
    _, (xmin, ymin) = sim.pop_axis(rmin, axis=axis)
    _, (xmax, ymax) = sim.pop_axis(rmax, axis=axis)

    for struct in sim.structures:
        geo = struct.geometry
        shapes = geo.intersections(x=x, y=y, z=z)
        mat_index = sim.medium_map[struct.medium]

        plot_params = sim.get_structure_plot_params(medium=struct.medium)
        color = plot_params.facecolor

        for shape in shapes:
            xs, ys = shape.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()
            xs = [xmin - BUFFER if np.isneginf(x) else x for x in xs]
            xs = [xmax + BUFFER if np.isposinf(x) else x for x in xs]
            ys = [ymin - BUFFER if np.isneginf(y) else y for y in ys]
            ys = [ymax + BUFFER if np.isposinf(y) else y for y in ys]
            plotly_trace = go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=color,
                line=dict(width=0),
                marker=dict(size=0.0001, line=dict(width=0)),
                name=struct.medium.name if struct.medium.name else f"medium[{mat_index}]",
            )
            fig.add_trace(plotly_trace)

    for i, source in enumerate(sim.sources):
        geo = source.geometry
        shapes = geo.intersections(x=x, y=y, z=z)
        plot_params = source.plot_params
        color = plot_params.facecolor
        opacity = plot_params.alpha

        for shape in shapes:
            xs, ys = shape.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()

            xs = [xmin - BUFFER if np.isneginf(x) else x for x in xs]
            xs = [xmax + BUFFER if np.isposinf(x) else x for x in xs]
            ys = [ymin - BUFFER if np.isneginf(y) else y for y in ys]
            ys = [ymax + BUFFER if np.isposinf(y) else y for y in ys]

            plotly_trace = go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=color,
                line=dict(width=4, color=color),
                marker=dict(size=0.0001, line=dict(width=0)),
                name=source.name if source.name else f"sources[{i}]",
                opacity=opacity,
            )
            fig.add_trace(plotly_trace)

    for monitor in sim.monitors:
        geo = monitor.geometry
        shapes = geo.intersections(x=x, y=y, z=z)
        plot_params = monitor.plot_params
        color = plot_params.facecolor
        opacity = plot_params.alpha

        for shape in shapes:
            xs, ys = shape.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()

            xs = [xmin - BUFFER if np.isneginf(x) else x for x in xs]
            xs = [xmax + BUFFER if np.isposinf(x) else x for x in xs]
            ys = [ymin - BUFFER if np.isneginf(y) else y for y in ys]
            ys = [ymax + BUFFER if np.isposinf(y) else y for y in ys]

            plotly_trace = go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=color,
                line=dict(width=4, color=color),
                marker=dict(size=0.0001, line=dict(width=0)),
                name=f'monitor: "{monitor.name}"',
                opacity=opacity,
            )
            fig.add_trace(plotly_trace)

    normal_axis, _ = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    for pml_axis, (pml, dl) in enumerate(zip(sim.pml_layers, sim.grid_size)):
        if pml is None or pml.num_layers == 0 or pml_axis == normal_axis:
            continue
        if isinstance(dl, float):
            dl_min = dl_max = dl
        else:
            dl_min = dl[0]
            dl_max = dl[-1]

        for sign, dl_edge in zip((-1, 1), (dl_min, dl_max)):
            pml_height = pml.num_layers * dl_edge
            pml_box = sim.make_pml_box(pml_axis=pml_axis, pml_height=pml_height, sign=sign)
            shapes = pml_box.intersections(x=x, y=y, z=z)
            color = pml_box.plot_params.facecolor
            opacity = pml_box.plot_params.alpha

            for shape in shapes:
                xs, ys = shape.exterior.coords.xy
                xs = xs.tolist()
                ys = ys.tolist()

                xs = [xmin - BUFFER if np.isneginf(x) else x for x in xs]
                xs = [xmax + BUFFER if np.isposinf(x) else x for x in xs]
                ys = [ymin - BUFFER if np.isneginf(y) else y for y in ys]
                ys = [ymax + BUFFER if np.isposinf(y) else y for y in ys]

                plotly_trace = go.Scatter(
                    x=xs,
                    y=ys,
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=1, color=color),
                    marker=dict(size=0.0001, line=dict(width=0)),
                    name="PML",
                    opacity=opacity,
                )
                fig.add_trace(plotly_trace)

    for sym_axis, sym_value in enumerate(sim.symmetry):
        if sym_value == 0 or sym_axis == normal_axis:
            continue
        sym_box = sim.make_symmetry_box(sym_axis=sym_axis, sym_value=sym_value)
        shapes = sym_box.intersections(x=x, y=y, z=z)
        color = sym_box.plot_params.facecolor
        opacity = pml_box.plot_params.alpha

        for shape in shapes:

            xs, ys = shape.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()

            xs = [xmin - BUFFER if np.isneginf(x) else x for x in xs]
            xs = [xmax + BUFFER if np.isposinf(x) else x for x in xs]
            ys = [ymin - BUFFER if np.isneginf(y) else y for y in ys]
            ys = [ymax + BUFFER if np.isposinf(y) else y for y in ys]

            sym_sign = '+' if sym_value > 0 else '-'

            plotly_trace = go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=color,
                line=dict(width=1, color=color),
                marker=dict(size=0.0001, line=dict(width=0)),
                name=f"{'xyz'[sym_axis]}-axis symmetry ({sym_sign}1)",
                opacity=opacity,
            )
            fig.add_trace(plotly_trace)

    fig.update_xaxes(range=[xmin, xmax])
    fig.update_yaxes(range=[ymin, ymax])
    width = xmax - xmin
    height = ymax - ymin
    min_dim = min(width, height)
    DESIRED_SIZE_PIXELS = 500.0
    if min_dim < DESIRED_SIZE_PIXELS:
        scale = DESIRED_SIZE_PIXELS / min_dim + 0.01
        width *= scale
        height *= scale
    fig.update_layout(width=width, height=height)

    _, (xlabel, ylabel) = sim.pop_axis("xyz", axis=axis)
    fig.update_layout(
        title=f'{"xyz"[axis]} = {pos:.2f}',
        xaxis_title=f"{xlabel} (um)",
        yaxis_title=f"{ylabel} (um)",
        legend_title="Contents",
    )

    seen = []
    for trace in fig["data"]:
        name = trace["name"]
        if name not in seen:
            seen.append(name)
        else:
            trace["showlegend"] = False

    return fig
