""" link what happens in the inputs to what gets displayed in the figure """

import numpy as np
from dash import callback, Output, Input, MATCH, State

from ..store import get_store


@callback(
    Output({"type": "FieldData_figure", "name": MATCH}, "figure"),
    [
        Input({"type": "FieldData_field_dropdown", "name": MATCH}, "value"),
        Input({"type": "FieldData_val_dropdown", "name": MATCH}, "value"),
        Input({"type": "FieldData_cs_axis_dropdown", "name": MATCH}, "value"),
        Input({"type": "FieldData_cs_slider", "name": MATCH}, "value"),
        Input({"type": "FieldData_ft_slider", "name": MATCH}, "value"),
        Input("store", "data"),
    ],
    State({"type": "FieldData_figure", "name": MATCH}, "id"),
)
def set_field(  # pylint:disable=too-many-arguments
    value_field,
    value_val,
    value_cs_axis,
    value_cs,
    value_ft,
    store,
    state_id,
    value_mode_ind=None,
):
    """set the field to be displayed"""
    data_plotly = get_store().get_data_plotly_by_name(store, state_id["name"])
    data_plotly.field_val = str(value_field)
    data_plotly.val = str(value_val)
    data_plotly.cs_axis = ["x", "y", "z"].index(value_cs_axis)
    data_plotly.cs_val = float(value_cs)
    data_plotly.ft_val = float(value_ft)
    data_plotly.mode_ind_val = int(value_mode_ind) if value_mode_ind is not None else None
    return data_plotly.make_figure()


# set the minimum of the xyz sliderbar depending on the cross-section axis
@callback(
    Output({"type": "FieldData_cs_slider", "name": MATCH}, "min"),
    Input({"type": "FieldData_cs_axis_dropdown", "name": MATCH}, "value"),
    Input("store", "data"),
    State({"type": "FieldData_figure", "name": MATCH}, "id"),
)
def set_min(value_cs_axis, store, state_id):
    """set the minimum of the sliderbar"""
    data_plotly = get_store().get_data_plotly_by_name(store, state_id["name"])
    data_plotly.cs_axis = ["x", "y", "z"].index(value_cs_axis)
    _, xyz_coords = data_plotly.xyz_label_coords
    return xyz_coords[0]


# set the xyz slider back to the average if the axis changes.
@callback(
    Output({"type": "FieldData_cs_slider", "name": MATCH}, "value"),
    Input({"type": "FieldData_cs_axis_dropdown", "name": MATCH}, "value"),
    Input("store", "data"),
    State({"type": "FieldData_figure", "name": MATCH}, "id"),
)
def reset_slider_position(value_cs_axis, store, state_id):
    """reset the sliderbar position"""
    data_plotly = get_store().get_data_plotly_by_name(store, state_id["name"])
    data_plotly.cs_axis = ["x", "y", "z"].index(value_cs_axis)
    _, xyz_coords = data_plotly.xyz_label_coords
    data_plotly.cs_val = float(np.mean(xyz_coords))
    return data_plotly.cs_val


# set the maximum of the xyz sliderbar depending on the cross-section axis
@callback(
    Output({"type": "FieldData_cs_slider", "name": MATCH}, "max"),
    Input({"type": "FieldData_cs_axis_dropdown", "name": MATCH}, "value"),
    Input("store", "data"),
    State({"type": "FieldData_figure", "name": MATCH}, "id"),
)
def set_max(value_cs_axis, store, state_id):
    """set the maximum of the sliderbar"""
    data_plotly = get_store().get_data_plotly_by_name(store, state_id["name"])
    data_plotly.cs_axis = ["x", "y", "z"].index(value_cs_axis)
    _, xyz_coords = data_plotly.xyz_label_coords
    return xyz_coords[-1]
