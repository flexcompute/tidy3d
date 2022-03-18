"""Adds plotly plotting to regular tidy3d objects."""
import pydantic as pd

from ...components.base import Tidy3dBaseModel
from ...components.simulation import Simulation
from ...components.geometry import Geometry
from ...components.structure import Structure


def plotly_shape(
    shape: ShapelyGeo,
    plot_params: PlotParams,
    fig: PlotlyFig,
    name: str = None,
) -> PlotlyFig:
    """Plot a shape to a figure."""
    _shape = Geometry.evaluate_inf_shape(shape)
    xs, ys = Geometry._get_shape_coords(shape=shape)
    plotly_trace = go.Scatter(
        x=xs,
        y=ys,
        fill="toself",
        fillcolor=plot_params.facecolor,
        line=dict(width=plot_params.linewidth, color=plot_params.facecolor),
        marker=dict(size=0.0001, line=dict(width=0)),
        name=name,
        opacity=plot_params.alpha,
    )
    fig.add_trace(plotly_trace)
    return fig

class SimulationPlotly(Simulation, Plotly):
    
    class 