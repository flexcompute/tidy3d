from abc import ABC, abstractmethod
from dash import dcc
import sys

sys.path.append("../../../")

from tidy3d.components.base import Tidy3dBaseModel

class UIComponent(Tidy3dBaseModel):

    @abstractmethod
    def make_figure(self) -> 'Plotly.Figure':
        """Creates the dash component for this montor data."""

    @abstractmethod
    def make_component(self, app) -> dcc.Tab:
        """Creates the dash component for this montor data."""
