"""Utilities for plotly plotting."""
from functools import wraps

import plotly.graph_objects as go

from ...components.viz import PlotParams

# type of a plotly figure
PlotlyFig = go.Figure

# display properties of a simulation boundary in plotly
plot_params_sim_boundary = PlotParams(linewidth=0, facecolor="rgba(0,0,0,0.05)", edgecolor="black")


def make_fig() -> PlotlyFig:
    """makes an empty plotly figure."""
    return go.Figure()


def add_fig_if_none(plotly):
    """Decorates `plot(*args, **kwargs, ax=None)` function.
    if fig=None in the function call, creates a plotly fig and feeds it to rest of function.
    """

    @wraps(plotly)
    def _plotly(*args, **kwargs) -> PlotlyFig:
        """New plot function using a generated ax if None."""
        if kwargs.get("fig") is None:
            fig = make_fig()
            kwargs["fig"] = fig
        return plotly(*args, **kwargs)

    return _plotly


def equal_aspect_plotly(plotly):
    """Decorates a plotting function returning a matplotlib axes.
    Ensures the aspect ratio of the returned axes is set to equal.
    Useful for 2D plots, like sim.plotly()
    """

    @wraps(plotly)
    def _plotly(*args, **kwargs) -> PlotlyFig:
        """New plot function with equal aspect ratio axes returned."""
        fig = plotly(*args, **kwargs)
        # fig.update_yaxes(
        #     scaleanchor="x",
        #     scaleratio=1,
        # )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        return fig

    return _plotly
