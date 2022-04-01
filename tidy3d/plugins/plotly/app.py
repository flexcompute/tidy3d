"""Makes an app."""
from abc import ABC, abstractmethod
from typing import List, Union, Any
from typing_extensions import Literal

from jupyter_dash import JupyterDash
from dash import Dash, dcc, html
import pydantic as pd

import sys

sys.path.append("../../tidy3d")
import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from .simulation import SimulationPlotly
from .data import DataPlotly

APP_MODE = Literal["python", "jupyter", "jupyterlab"]
DEFAULT_MODE = "jupyterlab"
DASH_APP = "Dash App"


class App(Tidy3dBaseModel, ABC):
    """Basic dash app template: initializes, makes layout, and fires up a server."""

    mode: APP_MODE = pd.Field(
        DEFAULT_MODE,
        title="App Mode",
        description='Run app differently based on `mode` in `"python"`, `"jupyter"`, `"jupyterlab"`',
    )

    def _initialize_app(self) -> DASH_APP:
        """Creates an app based on specs."""
        if "jupyter" in self.mode.lower():
            return JupyterDash(__name__)
        elif "python" in self.mode.lower():
            return Dash(__name__)
        else:
            raise NotImplementedError(f"doesnt support mode={mode}")

    @abstractmethod
    def _make_layout(self, app: DASH_APP) -> "Dash Layout":
        """Creates the layout for the app."""
        raise NotImplementedError("Must implement in subclass.")

    def _make_app(self) -> DASH_APP:
        """Initialize everything and make the plotly app."""
        app = self._initialize_app()
        app.layout = self._make_layout(app)
        return app

    def run(self, debug: bool = False) -> None:
        """Starts running the app based on specs."""

        app = self._make_app()

        if self.mode == "jupyterlab":
            app.run_server(
                mode="jupyterlab",
                port=8090,
                dev_tools_ui=True,
                dev_tools_hot_reload=True,
                threaded=True,
                debug=debug,
            )
        elif "python" in self.mode.lower():
            app.run_server(debug=debug, port=8090)
        else:
            raise NotImplementedError(f"doesnt support mode={mode}")


class SimulationDataApp(App):
    """App for viewing contents of a :class:`.SimulationData` instance."""

    sim_data: td.SimulationData = pd.Field(
        ..., title="Simulation data", description="A :class:`.SimulationData` instance to view."
    )

    def _make_layout(self, app: DASH_APP) -> "Dash Layout":
        """Creates the layout for the app."""

        layout = dcc.Tabs([])

        # simulation
        sim_plotly = SimulationPlotly(simulation=self.sim_data.simulation)
        component = sim_plotly.make_component(app)
        layout.children += [component]

        # monitors
        for monitor_name, monitor_data in self.sim_data.monitor_data.items():
            data_plotly = DataPlotly.from_monitor_data(
                monitor_data=monitor_data, monitor_name=monitor_name
            )
            if data_plotly is None:
                continue
            component = data_plotly.make_component(app)
            layout.children += [component]

        # log
        component = dcc.Tab(
            [
                html.Div([html.H1(f"Solver Log")]),
                html.Div([html.Code(self.sim_data.log, style={"whiteSpace": "pre-wrap"})]),
            ],
            label="log",
        )
        layout.children += [component]

        return layout

    @classmethod
    def from_file(cls, path: str, mode: APP_MODE = DEFAULT_MODE):
        """Load the SimulationDataApp from a tidy3d data file in .hdf5 format."""
        sim_data = td.SimulationData.from_file(path)
        sim_data_normalized = sim_data.normalize()
        return cls(sim_data=sim_data_normalized, mode=mode)


class SimulationApp(App):
    """TODO: App for viewing and editing a :class:`.Simulation`."""

    simulation: td.Simulation = pd.Field(
        ..., title="Simulation", description="A Simulation instance to view."
    )

    def _make_layout(self, app: DASH_APP) -> "Dash Layout":
        """Creates the layout for the app."""
        return cc.Tabs([])

    @classmethod
    def from_file(cls, path: str, mode: APP_MODE = DEFAULT_MODE):
        """Load the SimulationApp from a tidy3d Simulation file in .json or .yaml format."""
        simulation = td.Simulation.from_file(path)
        return cls(simulation=simulation, mode=mode)
