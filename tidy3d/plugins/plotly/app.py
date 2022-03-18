from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

from .geo import SimulationPlotly
from ...components.simulation import Simulation

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

def plot_sim(sim, x=None, y=None, z=None):
    sim_plotly = SimulationPlotly(simulation=sim)
    return sim_plotly.plotly(x=x, y=y, z=z)

def make_app(simulation:Simulation):

    # visit http://127.0.0.1:8050/ in your web browser.

    figx = plot_sim(simulation, x=0)
    figy = plot_sim(simulation, y=0)
    figz = plot_sim(simulation, z=0)

    (xmin, ymin, zmin), (xmax, ymax, zmax) = simulation.bounds

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

        # html.Div([
        #     "X position: ",
        #     dcc.Input(id='x_position', value="0", type="text")
        # ]),

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
    return app

app = make_app(simulation=sim)

@app.callback(
    Output(component_id='plot_x', component_property='figure'),
    Input(component_id='x_position', component_property='value')
)
def update_x_pos(input_value):
    return plot_sim(sim=sim, x=float(input_value))

@app.callback(
    Output(component_id='plot_y', component_property='figure'),
    Input(component_id='y_position', component_property='value')
)
def update_y_pos(input_value):
    return plot_sim(sim=sim, y=float(input_value))

@app.callback(
    Output(component_id='plot_z', component_property='figure'),
    Input(component_id='z_position', component_property='value')
)
def update_z_pos(input_value):
    return plot_sim(sim=sim, z=float(input_value))



# app.run_server(debug=True)