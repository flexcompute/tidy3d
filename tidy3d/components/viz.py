"""utilities for plotting"""

from __future__ import annotations

from functools import wraps
from html import escape
from typing import Any, Dict, Optional

import pydantic.v1 as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import is_color_like
    from matplotlib.patches import ArrowStyle, PathPatch
    from matplotlib.path import Path

    # default arrow style
    arrow_style = ArrowStyle.Simple(head_length=12, head_width=9, tail_width=4)
except ImportError:
    arrow_style = None

from numpy import array, concatenate, inf, ones

from ..constants import UnitScaling
from ..exceptions import SetupError, Tidy3dKeyError
from .base import Tidy3dBaseModel
from .types import Ax, Axis, LengthUnit

""" Constants """

# add this around extents of plots
PLOT_BUFFER = 0.3

ARROW_COLOR_MONITOR = "orange"
ARROW_COLOR_SOURCE = "green"
ARROW_COLOR_POLARIZATION = "brown"
ARROW_ALPHA = 0.8

# Arrow length in inches
ARROW_LENGTH = 0.3

FLEXCOMPUTE_COLORS = {
    "brand_green": 0x00643C,
    "brand_tan": 0xB8A18B,
    "brand_blue": 0x6DB5DD,
    "brand_purple": 0x8851AD,
    "brand_black": 0x000000,
}

""" Decorators """


def make_ax() -> Ax:
    """makes an empty ``ax``."""
    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """Decorates ``plot(*args, **kwargs, ax=None)`` function.
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


class AbstractPlotParams(Tidy3dBaseModel):
    """Abstract class for storing plotting parameters.
    Corresponds with select properties of ``matplotlib.artist.Artist``.
    """

    alpha: Any = pd.Field(1.0, title="Opacity")
    zorder: float = pd.Field(None, title="Display Order")

    def include_kwargs(self, **kwargs) -> AbstractPlotParams:
        """Update the plot params with supplied kwargs."""
        update_dict = {
            key: value
            for key, value in kwargs.items()
            if key not in ("type",) and value is not None and key in self.__fields__
        }
        return self.copy(update=update_dict)

    def override_with_viz_spec(self, viz_spec) -> AbstractPlotParams:
        """Override plot params with supplied VisualizationSpec."""
        return self.include_kwargs(**dict(viz_spec))

    def to_kwargs(self) -> dict:
        """Export the plot parameters as kwargs dict that can be supplied to plot function."""
        kwarg_dict = self.dict()
        for ignore_key in ("type", "attrs"):
            kwarg_dict.pop(ignore_key)
        return kwarg_dict


class PathPlotParams(AbstractPlotParams):
    """Stores plotting parameters / specifications for a path.
    Corresponds with select properties of ``matplotlib.lines.Line2D``.
    """

    color: Any = pd.Field(None, title="Color", alias="c")
    linewidth: pd.NonNegativeFloat = pd.Field(2, title="Line Width", alias="lw")
    linestyle: str = pd.Field("--", title="Line Style", alias="ls")
    marker: Any = pd.Field("o", title="Marker Style")
    markeredgecolor: Any = pd.Field(None, title="Marker Edge Color", alias="mec")
    markerfacecolor: Any = pd.Field(None, title="Marker Face Color", alias="mfc")
    markersize: pd.NonNegativeFloat = pd.Field(10, title="Marker Size", alias="ms")


class PlotParams(AbstractPlotParams):
    """Stores plotting parameters / specifications for a given model.
    Corresponds with select properties of ``matplotlib.patches.Patch``.
    """

    edgecolor: Any = pd.Field(None, title="Edge Color", alias="ec")
    facecolor: Any = pd.Field(None, title="Face Color", alias="fc")
    fill: bool = pd.Field(True, title="Is Filled")
    hatch: str = pd.Field(None, title="Hatch Style")
    linewidth: pd.NonNegativeFloat = pd.Field(1, title="Line Width", alias="lw")


# defaults for different tidy3d objects
plot_params_geometry = PlotParams()
plot_params_structure = PlotParams()
plot_params_source = PlotParams(alpha=0.4, facecolor="limegreen", edgecolor="limegreen", lw=3)
plot_params_monitor = PlotParams(alpha=0.4, facecolor="orange", edgecolor="orange", lw=3)
plot_params_pml = PlotParams(alpha=0.7, facecolor="gray", edgecolor="gray", hatch="x", zorder=inf)
plot_params_pec = PlotParams(alpha=1.0, facecolor="gold", edgecolor="black", zorder=inf)
plot_params_pmc = PlotParams(alpha=1.0, facecolor="lightsteelblue", edgecolor="black", zorder=inf)
plot_params_bloch = PlotParams(alpha=1.0, facecolor="orchid", edgecolor="black", zorder=inf)
plot_params_symmetry = PlotParams(edgecolor="gray", facecolor="gray", alpha=0.6, zorder=inf)
plot_params_override_structures = PlotParams(
    linewidth=0.4, edgecolor="black", fill=False, zorder=inf
)
plot_params_fluid = PlotParams(facecolor="white", edgecolor="lightsteelblue", lw=0.4, hatch="xx")
plot_params_grid = PlotParams(edgecolor="black", lw=0.2)
plot_params_lumped_element = PlotParams(
    alpha=0.4, facecolor="mediumblue", edgecolor="mediumblue", lw=3
)

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
STRUCTURE_EPS_CMAP_R = "gist_yarg_r"
STRUCTURE_HEAT_COND_CMAP = "gist_yarg"


def is_valid_color(value: str) -> str:
    if not is_color_like(value):
        raise pd.ValidationError(f"{value} is not a valid plotting color")

    return value


class VisualizationSpec(Tidy3dBaseModel):
    """Defines specification for visualization when used with plotting functions."""

    facecolor: str = pd.Field(
        "",
        title="Face color",
        description="Color applied to the faces in visualization.",
    )

    edgecolor: Optional[str] = pd.Field(
        "",
        title="Edge color",
        description="Color applied to the edges in visualization.",
    )

    alpha: Optional[pd.confloat(ge=0.0, le=1.0)] = pd.Field(
        1.0,
        title="Opacity",
        description="Opacity/alpha value in plotting between 0 and 1.",
    )

    @pd.validator("facecolor", always=True)
    def validate_color(value: str) -> str:
        return is_valid_color(value)

    @pd.validator("edgecolor", always=True)
    def validate_and_copy_color(value: str, values: Dict[str, Any]) -> str:
        if (value == "") and "facecolor" in values:
            return is_valid_color(values["facecolor"])

        return is_valid_color(value)


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

    The ``polygon`` may be a Shapely or GeoJSON-like object with or without holes.
    The ``kwargs`` are those supported by the matplotlib.patches.Polygon class
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


def plot_sim_3d(sim, width=800, height=800) -> None:
    """Make 3D display of simulation in ipyython notebook."""

    try:
        from IPython.display import HTML, display
    except ImportError as e:
        raise SetupError(
            "3D plotting requires ipython to be installed "
            "and the code to be running on a jupyter notebook."
        ) from e

    from base64 import b64encode
    from io import BytesIO

    buffer = BytesIO()
    sim.to_hdf5_gz(buffer)
    buffer.seek(0)
    base64 = b64encode(buffer.read()).decode("utf-8")
    js_code = """
        /**
        * Simulation Viewer Injector
        *
        * Monitors the document for elements being added in the form:
        *
        *    <div class="simulation-viewer" data-width="800" data-height="800" data-simulation="{...}" />
        *
        * This script will then inject an iframe to the viewer application, and pass it the simulation data
        * via the postMessage API on request. The script may be safely included multiple times, with only the
        * configuration of the first started script (e.g. viewer URL) applying.
        *
        */
        (function() {
            const TARGET_CLASS = "simulation-viewer";
            const ACTIVE_CLASS = "simulation-viewer-active";
            const VIEWER_URL = "https://tidy3d.simulation.cloud/simulation-viewer";

            class SimulationViewerInjector {
                constructor() {
                    for (var node of document.getElementsByClassName(TARGET_CLASS)) {
                        this.injectViewer(node);
                    }

                    // Monitor for newly added nodes to the DOM
                    this.observer = new MutationObserver(this.onMutations.bind(this));
                    this.observer.observe(document.body, {childList: true, subtree: true});
                }

                onMutations(mutations) {
                    for (var mutation of mutations) {
                        if (mutation.type === 'childList') {
                            /**
                            * Have found that adding the element does not reliably trigger the mutation observer.
                            * It may be the case that setting content with innerHTML does not trigger.
                            *
                            * It seems to be sufficient to re-scan the document for un-activated viewers
                            * whenever an event occurs, as Jupyter triggers multiple events on cell evaluation.
                            */
                            var viewers = document.getElementsByClassName(TARGET_CLASS);
                            for (var node of viewers) {
                                this.injectViewer(node);
                            }
                        }
                    }
                }

                injectViewer(node) {
                    // (re-)check that this is a valid simulation container and has not already been injected
                    if (node.classList.contains(TARGET_CLASS) && !node.classList.contains(ACTIVE_CLASS)) {
                        // Mark node as injected, to prevent re-runs
                        node.classList.add(ACTIVE_CLASS);

                        var uuid;
                        if (window.crypto && window.crypto.randomUUID) {
                            uuid = window.crypto.randomUUID();
                        } else {
                            uuid = "" + Math.random();
                        }

                        var frame = document.createElement("iframe");
                        frame.width = node.dataset.width || 800;
                        frame.height = node.dataset.height || 800;
                        frame.style.cssText = `width:${frame.width}px;height:${frame.height}px;max-width:none;border:0;display:block`
                        frame.src = VIEWER_URL + "?uuid=" + uuid;

                        var postMessageToViewer;
                        postMessageToViewer = event => {
                            if(event.data.type === 'viewer' && event.data.uuid===uuid){
                                frame.contentWindow.postMessage({ type: 'jupyter', uuid, value: node.dataset.simulation, fileType: 'hdf5'}, '*');

                                // Run once only
                                window.removeEventListener('message', postMessageToViewer);
                            }
                        };
                        window.addEventListener(
                            'message',
                            postMessageToViewer,
                            false
                        );

                        node.appendChild(frame);
                    }
                }
            }

            if (!window.simulationViewerInjector) {
                window.simulationViewerInjector = new SimulationViewerInjector();
            }
        })();
    """
    html_code = f"""
    <div class="simulation-viewer" data-width="{escape(str(width))}" data-height="{escape(str(height))}" data-simulation="{escape(base64)}" ></div>
    <script>
        {js_code}
    </script>
    """

    return display(HTML(html_code))


def set_default_labels_and_title(
    axis_labels: tuple[str, str],
    axis: Axis,
    position: float,
    ax: Ax,
    plot_length_units: LengthUnit = None,
) -> Ax:
    """Adds axis labels and title to plots involving spatial dimensions.
    When the ``plot_length_units`` are specified, the plot axes are scaled, and
    the title and axis labels include the desired units.
    """
    xlabel = axis_labels[0]
    ylabel = axis_labels[1]
    if plot_length_units is not None:
        if plot_length_units not in UnitScaling:
            raise Tidy3dKeyError(
                f"Provided units '{plot_length_units}' are not supported. "
                f"Please choose one of '{LengthUnit}'."
            )
        ax.set_xlabel(f"{xlabel} ({plot_length_units})")
        ax.set_ylabel(f"{ylabel} ({plot_length_units})")
        # Formatter to help plot in arbitrary units
        scale_factor = UnitScaling[plot_length_units]
        formatter = ticker.FuncFormatter(lambda y, _: f"{y * scale_factor:.2f}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_title(
            f"cross section at {'xyz'[axis]}={position * scale_factor:.2f} ({plot_length_units})"
        )
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")
    return ax
