# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any
from abc import abstractmethod
from functools import wraps

import matplotlib.pylab as plt
from pydantic import BaseModel
import plotly.graph_objects as go
import numpy as np

from .types import Ax
from ..constants import pec_val

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

# arrow width cannot be larger than this factor times the max of axis height and width
MAX_ARROW_WIDTH_FACTOR = 0.02


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


""" Utilities for default plotting parameters."""


class PatchParams(BaseModel):
    """Datastructure holding default parameters for plotting matplotlib.patches.
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html for explanation.
    """

    alpha: Any = None
    edgecolor: Any = None
    facecolor: Any = None
    fill: bool = True
    hatch: str = None
    lw: int = 1


class PatchParamSwitcher(BaseModel):
    """Class used for updating plot kwargs based on :class:`PatchParams` values."""

    def update_params(self, **plot_params):
        """return dictionary of plot params updated with fields user supplied **plot_params dict."""

        # update self's plot params with the the supplied plot parameters and return non none ones.
        default_plot_params = self.get_plot_params()
        default_plot_params_dict = default_plot_params.dict().copy()
        default_plot_params_dict.update(plot_params)

        # get rid of pairs with value of None as they will mess up plots down the line.
        return {key: val for key, val in default_plot_params_dict.items() if val is not None}

    @abstractmethod
    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args.  Implement in subclasses."""


class GeoParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Geometry`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(edgecolor=None, facecolor="cornflowerblue")


class SourceParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Source`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.4, facecolor="limegreen", edgecolor="limegreen", lw=3)


class MonitorParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Monitor`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.4, facecolor="orange", edgecolor="orange", lw=3)


class StructMediumParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Structures` in ``Simulation.plot_structures``."""

    medium: Any
    medium_map: dict

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        mat_index = self.medium_map[self.medium]
        mat_cmap = [
            "#689DBC",
            "#D0698E",
            "#5E6EAD",
            "#C6224E",
            "#BDB3E2",
            "#9EC3E0",
            "#616161",
            "#877EBC",
        ]

        if mat_index == 0:
            facecolor = "white"
        else:
            facecolor = mat_cmap[(mat_index - 1) % len(mat_cmap)]
        if self.medium.name == "PEC":
            return PatchParams(facecolor="gold", edgecolor="k", lw=1)
        return PatchParams(facecolor=facecolor, edgecolor=facecolor, lw=0)


class StructEpsParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Structures` in `td.Simulation.plot_structures_eps`."""

    eps: float
    eps_max: float
    eps_min: float
    reverse: bool = True

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        eps_min = min(self.eps_min, 1)
        delta_eps = self.eps - eps_min
        delta_eps_max = self.eps_max - eps_min + 1e-5
        color = delta_eps / delta_eps_max
        if self.reverse:
            color = 1 - color
        if self.eps == pec_val:
            return PatchParams(facecolor="gold", edgecolor="k", lw=1)
        return PatchParams(facecolor=str(color), edgecolor=str(color), lw=0)


class PMLParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`AbsorberSpec` (PML)."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.7, facecolor="gray", edgecolor="gray", hatch="x")


class SymParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Simulation.symmetry`."""

    sym_value: int

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""

        if self.sym_value == 1:
            sym_color = "lightsteelblue"
            return PatchParams(alpha=0.6, facecolor=sym_color, edgecolor=sym_color, hatch="++")
        if self.sym_value == -1:
            sym_color = "rosybrown"
            return PatchParams(alpha=0.6, facecolor=sym_color, edgecolor=sym_color, hatch="--")
        raise ValueError(f"sym_value of {self.sym_value} not recognized, must be 1 or -1.")


class SimDataGeoParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.SimulationData`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.4, edgecolor="black")


# pylint:disable=too-many-locals, too-many-branches, too-many-statements
def plotly_sim(sim, x=None, y=None, z=None):
    """Make a plotly plot."""

    fig = go.Figure()
    axis, pos = sim.parse_xyz_kwargs(x=x, y=y, z=z)
    rmin, rmax = sim.bounds
    rmin = np.array(rmin)
    rmax = np.array(rmax)

    for struct in sim.structures:
        geo = struct.geometry
        shapes = geo.intersections(x=x, y=y, z=z)
        mat_index = sim.medium_map[struct.medium]

        params = StructMediumParams(medium=struct.medium, medium_map=sim.medium_map)
        color = params.get_plot_params().facecolor

        for shape in shapes:
            xs, ys = shape.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()

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
        params = SourceParams()
        color = params.get_plot_params().facecolor
        opacity = params.get_plot_params().alpha

        for shape in shapes:
            xs, ys = shape.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()

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
        params = MonitorParams()
        color = params.get_plot_params().facecolor
        opacity = params.get_plot_params().alpha

        for shape in shapes:
            xs, ys = shape.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()

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

    for dir_index, (pml, dl) in enumerate(zip(sim.pml_layers, sim.grid_size)):
        if pml is None:
            continue
        if isinstance(dl, float):
            dl_min = dl_max = dl
        else:
            dl_min = dl[0]
            dl_max = dl[-1]
        for sign, dl_edge in zip((-1, 1), (dl_min, dl_max)):
            pml_thick = pml.num_layers * dl_edge
            pml_size = 2 * np.array(sim.size)
            pml_size[dir_index] = pml_thick
            if sign == 1:
                rmax[dir_index] += pml_thick
            else:
                rmin[dir_index] -= pml_thick
            pml_center = np.array(sim.center) + sign * np.array(sim.size) / 2
            pml_center[dir_index] += sign * pml_thick / 2
            pml_box = sim.geometry
            pml_box.center = pml_center.tolist()
            pml_box.size = pml_size.tolist()
            shapes = pml_box.intersections(x=x, y=y, z=z)
            params = PMLParams()
            color = params.get_plot_params().facecolor
            opacity = params.get_plot_params().alpha

            for shape in shapes:
                xs, ys = shape.exterior.coords.xy
                xs = xs.tolist()
                ys = ys.tolist()

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

    _, (xmin, ymin) = sim.pop_axis(rmin, axis=axis)
    _, (xmax, ymax) = sim.pop_axis(rmax, axis=axis)

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
