from abc import ABC, abstractmethod
from dash import dcc
import sys

sys.path.append("../../../")

from tidy3d.components.base import Tidy3dBaseModel

"""
How the components work.

1. The app is initialized in `App`.
2. For each UI component in the view:
    a) construct UI component using SimulationData object contents.
    b) call `make_component(app)`, which
        i. generates the layout for the component.
        ii. adds any callback functions to `app` to make it interactive.
        iii. returns the layout.
    c) add the layout from this component to the app.

If a figure is needed, the component can create it with `fig = self.make_figure()`
This will use the internal state to contruct kwargs for a call to `fig = self.plotly(**kwargs)`

"""


class UIComponent(Tidy3dBaseModel):
    @abstractmethod
    def make_component(self, app) -> dcc.Tab:
        """Creates the dash component for this montor data."""

    @abstractmethod
    def make_figure(self) -> "Plotly.Figure":
        """Creates the dash component for this montor data."""

    @abstractmethod
    def plotly(self, **kwargs) -> "Plotly.Figure":
        """Make a plotly figure using all of the optional kwargs."""
