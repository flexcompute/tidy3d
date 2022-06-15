""" link what happens in the inputs to what gets displayed in the figure """
from dash import callback, Output, MATCH, Input, State

from ..store import get_store


@callback(
    Output({"type": "ModeData_figure", "name": MATCH}, "figure"),
    [
        Input({"type": "ModeData_amps_or_neff_dropdown", "name": MATCH}, "value"),
        Input({"type": "ModeData_val_dropdown", "name": MATCH}, "value"),
        Input({"type": "ModeData_dir_dropdown", "name": MATCH}, "value"),
        Input({"type": "ModeData_mode_index_selector", "name": MATCH}, "value"),
        Input("store", "data"),
    ],
    State({"type": "ModeData_figure", "name": MATCH}, "id"),
)
def set_field(value_amps_or_neff, value_val, value_dir, value_mode_ind, store, state_id):
    """set the field to plot"""
    # pylint: disable=too-many-arguments
    data_plotly = get_store().get_data_plotly_by_name(store, state_id["name"])
    data_plotly.amps_or_neff = str(value_amps_or_neff)
    data_plotly.val = str(value_val)
    data_plotly.dir_val = str(value_dir)

    data_plotly.mode_ind_val = int(value_mode_ind) if value_mode_ind is not None else None
    return data_plotly.make_figure()


#
@callback(
    Output({"type": "ModeData_dir_dropdown_header", "name": MATCH}, "hidden"),
    [
        Input({"type": "ModeData_amps_or_neff_dropdown", "name": MATCH}, "value"),
        Input("store", "data"),
    ],
    State({"type": "ModeData_figure", "name": MATCH}, "id"),
)
def set_dir_header_visibilty(value_amps_or_neff, store, state_id):
    """set the visibility of the dropdown"""
    data_plotly = get_store().get_data_plotly_by_name(store, state_id["name"])
    data_plotly.amps_or_neff = str(value_amps_or_neff)
    return data_plotly.dir_dropdown_hidden


#
@callback(
    Output({"type": "ModeData_dir_dropdown_div", "name": MATCH}, "hidden"),
    [
        Input({"type": "ModeData_amps_or_neff_dropdown", "name": MATCH}, "value"),
        Input("store", "data"),
    ],
    State({"type": "ModeData_figure", "name": MATCH}, "id"),
)
def set_dir_dropdown_visibilty(value_amps_or_neff, store, state_id):
    """set the visibility of the dropdown"""
    data_plotly = get_store().get_data_plotly_by_name(store, state_id["name"])
    data_plotly.amps_or_neff = str(value_amps_or_neff)
    return data_plotly.dir_dropdown_hidden
