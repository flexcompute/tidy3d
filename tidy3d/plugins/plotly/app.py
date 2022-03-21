from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import pydantic as pd

from .geo import SimulationPlotly
from ...components.simulation import Simulation

from ...components.base import Tidy3dBaseModel
from ...components.simulation import Simulation
from ...components.geometry import Box, Sphere
from ...components.medium import Medium
from ...components.structure import Structure
from ...components.source import PlaneWave, GaussianPulse
from ...components.monitor import FluxMonitor
from ...constants import inf

def plot_sim(sim, x=None, y=None, z=None):
    sim_plotly = SimulationPlotly(simulation=sim)
    return sim_plotly.plotly(x=x, y=y, z=z)

class Tidy3dApp(Tidy3dBaseModel):

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation to generate the plots."
    )

    def start(self):
        # visit http://127.0.0.1:8050/ in your web browser.

        app = Dash(__name__),

        figx = plot_sim(self.simulation, x=0)
        figy = plot_sim(self.simulation, y=0)
        figz = plot_sim(self.simulation, z=0)

        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.simulation.bounds

        colors = {
            'background': '#111111',
            'text': '#7FDBFF'
        }


        app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
            html.H1(
                children='Hello Dash',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),

            html.Div(children='Dash: A web application framework for your data.', style={
                'textAlign': 'center',
                'color': colors['text']
            }),

            dcc.Slider(
                min=xmin,
                max=xmax,
                value=0,
                id='x_position'
            ),

            dcc.Graph(figure=figx, id='plot_x'),

            dcc.Slider(
                min=ymin,
                max=ymax,
                value=0,
                id='y_position'
            ),

            dcc.Graph(figure=figy, id='plot_y'),

            dcc.Slider(
                min=zmin,
                max=zmax,
                value=0,
                id='z_position'
            ),
            dcc.Graph(figure=figz, id='plot_z'),
        ])

        @app.callback(
            Output(component_id='plot_x', component_property='figure'),
            Input(component_id='x_position', component_property='value')
        )
        def update_x_pos(input_value):
            return plot_sim(sim=self.simulation, x=float(input_value))

        @app.callback(
            Output(component_id='plot_y', component_property='figure'),
            Input(component_id='y_position', component_property='value')
        )
        def update_y_pos(input_value):
            return plot_sim(sim=self.simulation, y=float(input_value))

        @app.callback(
            Output(component_id='plot_z', component_property='figure'),
            Input(component_id='z_position', component_property='value')
        )
        def update_z_pos(input_value):
            return plot_sim(sim=self.simulation, z=float(input_value))

        app.run_server(debug=True)
