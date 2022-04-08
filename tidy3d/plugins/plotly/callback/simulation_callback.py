from dash import callback, Output, Input, dcc

from tidy3d.plugins.plotly import SimulationPlotly
from tidy3d.plugins.plotly.data import DataPlotly
from tidy3d.plugins.plotly.store import get_simulation_data, get_simulation_plotly


@callback(
    Output("container", "children"),
    [Input("store", "data")],
)
def display_simulation_data_app(store):
    data_app = get_simulation_data(store)
    layout = dcc.Tabs([])
    component = SimulationPlotly(simulation=data_app.simulation).make_component()
    layout.children += [component]

    for monitor_name, monitor_data in data_app.monitor_data.items():
        data_plotly = DataPlotly.from_monitor_data(
            monitor_data=monitor_data, monitor_name=monitor_name
        )
        if data_plotly is None:
            continue
        component = data_plotly.make_component()
        layout.children += [component]
    return layout


@callback(
    Output("simulation_plot", "figure"),
    [
        Input("simulation_cs_axis_dropdown", "value"),
        Input("simulation_cs_slider", "value"),
        Input("store", "data"),
    ],
)
def set_fig_from_xyz_sliderbar(cs_axis_string, cs_val, store):
    sim_plotly = get_simulation_plotly(store)
    sim_plotly.cs_axis = ["x", "y", "z"].index(cs_axis_string)
    sim_plotly.cs_val = float(cs_val)

    return sim_plotly.make_figure()


# set the xyz slider back to the average if the axis changes.
@callback(
    Output("simulation_cs_slider", "value"),
    Input("simulation_cs_axis_dropdown", "value"),
    Input("store", "data"),
)
def reset_slider_position(value_cs_axis, store):
    sim_plotly = get_simulation_plotly(store)
    sim_plotly.cs_axis = ["x", "y", "z"].index(value_cs_axis)
    _, (xyz_min, xyz_max) = sim_plotly.xyz_label_bounds
    sim_plotly.cs_val = float((xyz_min + xyz_max) / 2.0)
    return sim_plotly.cs_val


@callback(
    Output("simulation_cs_slider", "min"),
    Input("simulation_cs_axis_dropdown", "value"),
    Input("store", "data"),
)
def set_min(cs_axis_string, store):
    sim_plotly = get_simulation_plotly(store)
    sim_plotly.cs_axis = ["x", "y", "z"].index(cs_axis_string)
    _, (xyz_min, _) = sim_plotly.xyz_label_bounds
    return xyz_min


@callback(
    Output("simulation_cs_slider", "max"),
    Input("simulation_cs_axis_dropdown", "value"),
    Input("store", "data"),
)
def set_max(cs_axis_string, store):
    sim_plotly = get_simulation_plotly(store)
    sim_plotly.cs_axis = ["x", "y", "z"].index(cs_axis_string)
    _, (_, xyz_max) = sim_plotly.xyz_label_bounds
    return xyz_max
