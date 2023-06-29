# pylint: disable=invalid-name
""" utilities for plotting """
from __future__ import annotations

from typing import Any
from functools import wraps
import random
import time

import matplotlib.pylab as plt
from matplotlib.patches import PathPatch, ArrowStyle
from matplotlib.path import Path
from numpy import array, concatenate, ones
import pydantic as pd

from .types import Ax
from .base import Tidy3dBaseModel
from ..exceptions import SetupError

""" Constants """

# add this around extents of plots
PLOT_BUFFER = 0.3

ARROW_COLOR_MONITOR = "orange"
ARROW_COLOR_SOURCE = "green"
ARROW_COLOR_POLARIZATION = "brown"
ARROW_ALPHA = 0.8


# Arrow length in inches
ARROW_LENGTH = 0.3


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


class PlotParams(Tidy3dBaseModel):
    """Stores plotting parameters / specifications for a given model."""

    alpha: Any = pd.Field(1.0, title="Opacity")
    edgecolor: Any = pd.Field(None, title="Edge Color", alias="ec")
    facecolor: Any = pd.Field(None, title="Face Color", alias="fc")
    fill: bool = pd.Field(True, title="Is Filled")
    hatch: str = pd.Field(None, title="Hatch Style")
    linewidth: pd.NonNegativeFloat = pd.Field(1, title="Line Width", alias="lw")

    def include_kwargs(self, **kwargs) -> PlotParams:
        """Update the plot params with supplied kwargs."""
        update_dict = {
            key: value
            for key, value in kwargs.items()
            if key not in ("type",) and value is not None and key in self.__fields__
        }
        return self.copy(update=update_dict)

    def to_kwargs(self) -> dict:
        """Export the plot parameters as kwargs dict that can be supplied to plot function."""
        kwarg_dict = self.dict()
        for ignore_key in ("type",):
            kwarg_dict.pop(ignore_key)
        return kwarg_dict


# defaults for different tidy3d objects
plot_params_geometry = PlotParams()
plot_params_structure = PlotParams()
plot_params_source = PlotParams(alpha=0.4, facecolor="limegreen", edgecolor="limegreen", lw=3)
plot_params_monitor = PlotParams(alpha=0.4, facecolor="orange", edgecolor="orange", lw=3)
plot_params_pml = PlotParams(alpha=0.7, facecolor="gray", edgecolor="gray", hatch="x")
plot_params_pec = PlotParams(alpha=1.0, facecolor="gold", edgecolor="black")
plot_params_pmc = PlotParams(alpha=1.0, facecolor="lightsteelblue", edgecolor="black")
plot_params_bloch = PlotParams(alpha=1.0, facecolor="orchid", edgecolor="black")
plot_params_symmetry = PlotParams(edgecolor="gray", facecolor="gray", alpha=0.6)
plot_params_override_structures = PlotParams(linewidth=0.4, edgecolor="black", fill=False)

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

# colormap for structure's permittivity in plot_eps
STRUCTURE_EPS_CMAP = "gist_yarg"

# default arrow style
arrow_style = ArrowStyle.Simple(head_length=12, head_width=9, tail_width=4)


"""=================================================================================================
Descartes modified from https://pypi.org/project/descartes/ for Shapely >= 1.8.0

Copyright Flexcompute 2022

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


class Polygon:
    """Adapt Shapely polygons to a common interface"""

    def __init__(self, context):
        if isinstance(context, dict):
            self.context = context["coordinates"]
        else:
            self.context = context

    @property
    def exterior(self):
        """Get polygon exterior."""
        return getattr(self.context, "exterior", None) or self.context[0]

    @property
    def interiors(self):
        """Get polygon interiors."""
        value = getattr(self.context, "interiors", None)
        if value is None:
            value = self.context[1:]
        return value


def polygon_path(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(obj):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(obj, "coords", None) or obj)
        vals = ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    ptype = polygon.geom_type
    if ptype == "Polygon":
        polygon = [Polygon(polygon)]
    elif ptype == "MultiPolygon":
        polygon = [Polygon(p) for p in polygon.geoms]

    vertices = concatenate(
        [
            concatenate(
                [array(t.exterior.coords)[:, :2]] + [array(r.coords)[:, :2] for r in t.interiors]
            )
            for t in polygon
        ]
    )
    codes = concatenate(
        [concatenate([coding(t.exterior)] + [coding(r) for r in t.interiors]) for t in polygon]
    )

    return Path(vertices, codes)


def polygon_patch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example
    -------
    >>> b = Point(0, 0).buffer(1.0) # doctest: +SKIP
    >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5) # doctest: +SKIP
    >>> axis.add_patch(patch) # doctest: +SKIP

    """
    return PathPatch(polygon_path(polygon), **kwargs)


"""End descartes modification
================================================================================================="""


def plot_sim_3d(sim) -> None:
    """Make 3D display of simulation in ipyython notebook."""

    try:
        # pylint:disable=import-outside-toplevel
        from IPython.display import display, HTML
    except ImportError as e:
        raise SetupError(
            "3D plotting requires ipython to be installed "
            "and the code to be running on a jupyter notebook."
        ) from e

    uuid = str(int(time.time() * 1000)) + str(random.randint(0, 100000))
    # pylint:disable=protected-access
    js_code = f"""
    window.postMessageToViewer{uuid} = event => {{
        if(event.data.type === 'viewer'&&event.data.uuid==='{uuid}'){{
            document.getElementById('simulation-viewer{uuid}').contentWindow.postMessage({{ type: 'jupyter', uuid:'{uuid}', value:{sim._json_string}}}, '*')
        }}
    }};
    window.addEventListener(
        'message',
        window.postMessageToViewer{uuid},
        false
    );
    """
    viewer_url = (
        "https://feature-simulation-viewer.d3a9gfg7glllfq.amplifyapp.com/simulation-viewer?uuid="
        + str(uuid)
    )
    html_code = f"""
    <iframe id="simulation-viewer{uuid}" src={viewer_url} width="800" height="800"></iframe>
    <script>
        {js_code}
    </script>
    """

    return display(HTML(html_code))
