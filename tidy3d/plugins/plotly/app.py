from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

from .geo import SimulationPlotly
from ...components.simulation import Simulation

def make_app(simulation:Simulation):

    sim = SimulationPlotly(simulation=simulation)
    figx = sim.plotly(x=0)
    figy = sim.plotly(y=0)
    figz = sim.plotly(z=0)

    # visit http://127.0.0.1:8050/ in your web browser.

    app = Dash(__name__)

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

        dcc.Graph(figure=figx),
        dcc.Graph(figure=figy),
        dcc.Graph(figure=figz),
    ])
    return app


from ...components.simulation import Simulation
from ...components.geometry import Box, Sphere
from ...components.medium import Medium
from ...components.structure import Structure
from ...components.source import PlaneWave, GaussianPulse
from ...components.monitor import FluxMonitor
from ...constants import inf

sim = Simulation(
    size=(5,5,5),
    grid_size=(.1, .1, .1),
    run_time=1e-12,
    structures=[
        Structure(
            geometry=Box(size=(1,1,1), center=(2, 0, 0)),
            medium=Medium(permittivity=2.0)
        ),
        Structure(
            geometry=Box(size=(1,1,1), center=(-2, 0, 0)),
            medium=Medium(permittivity=3.0)
        ),
        Structure(
            geometry=Sphere(radius=1),
            medium=Medium(permittivity=4.0)
        ),                
    ],
    sources=[
        PlaneWave(
            center=(0, 2, 0), 
            size=(inf, 0, inf),
            source_time=GaussianPulse(freq0=200e12, fwidth=100e12),
            pol_angle=0,
            direction="-"
        )
    ],
    monitors=[
        FluxMonitor(
            center=(0,-2,0),
            size=(inf, 0, inf),
            freqs=[200e12],
            name='flux'
        )
    ]
)

app = make_app(simulation=sim)
# app.run_server(debug=True)