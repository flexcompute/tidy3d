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
from .data import FieldDataPlotly

APP_MODE = Literal["python", "jupyter", "jupyterlab"]
DEFAULT_MODE = "jupyterlab"
DASH_APP = "Dash App"


class App(Tidy3dBaseModel, ABC):
    """Generic app."""

    mode: APP_MODE = pd.Field(
        DEFAULT_MODE,
        title="App Mode",
        description="If running in jupyter notebook, should be 'jupyter'.",
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

        # Simulation Plot.
        # >>> Wrap in something
        sim_plotly = SimulationPlotly(simulation=self.sim_data.simulation)
        layout.children += [
            dcc.Tab([html.Div([dcc.Graph(figure=sim_plotly.plotly(x=0))])], label="Simulation")
        ]
        # <<<

        # Monitor Data
        # >>> Wrap in something
        for monitor_name, monitor_data in self.sim_data.monitor_data.items():
            label = f'monitor: "{monitor_name}"'
            data_plotly = FieldDataPlotly(data=monitor_data)
            layout.children += [
                dcc.Tab(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    figure=data_plotly.plotly(
                                        x=0, field="Ex", freq=float(monitor_data.Ex.f), val="abs"
                                    )
                                )
                            ]
                        )
                    ],
                    label=label,
                )
            ]
        # <<<

        # log
        layout.children += [
            dcc.Tab([html.Code(self.sim_data.log, style={"whiteSpace": "pre-wrap"})], label="log")
        ]

        return layout

    @classmethod
    def from_file(cls, path: str, mode: APP_MODE = DEFAULT_MODE):
        """Load the SimulationDataApp from a tidy3d data file in .hdf5 format."""
        sim_data = td.SimulationData.from_file(path)
        return cls(sim_data=sim_data, mode=mode)


class SimulationApp(App):
    """TODO: App for viewing and editing a Simulation."""

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
