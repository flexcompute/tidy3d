""" link what happens in the inputs to what gets displayed in the figure """

from dash import callback, Output, Input

from ..store import get_store


@callback(
    Output("simulation_plot", "figure"),
    [
        Input("simulation_cs_axis_dropdown", "value"),
        Input("simulation_cs_slider", "value"),
        Input("store", "data"),
    ],
)
def set_fig_from_xyz_sliderbar(cs_axis_string, cs_val, store):
    """set the figure from the xyz slider bar"""
    sim_plotly = get_store().get_simulation_plotly(store)
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
    """set the xyz slider back to the average if the axis changes."""
    sim_plotly = get_store().get_simulation_plotly(store)
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
    """set the min slider value"""
    sim_plotly = get_store().get_simulation_plotly(store)
    sim_plotly.cs_axis = ["x", "y", "z"].index(cs_axis_string)
    _, (xyz_min, _) = sim_plotly.xyz_label_bounds
    return xyz_min


@callback(
    Output("simulation_cs_slider", "max"),
    Input("simulation_cs_axis_dropdown", "value"),
    Input("store", "data"),
)
def set_max(cs_axis_string, store):
    """set the max slider value"""
    sim_plotly = get_store().get_simulation_plotly(store)
    sim_plotly.cs_axis = ["x", "y", "z"].index(cs_axis_string)
    _, (_, xyz_max) = sim_plotly.xyz_label_bounds
    return xyz_max
