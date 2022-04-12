""" link what happens in the inputs to what gets displayed in the figure """

import urllib

from dash import callback, Output, Input, dcc, html

from ..simulation import SimulationPlotly
from ..data import DataPlotly
from ..store import get_store


@callback(
    Output("store", "data"),
    [Input("url", "search")],
)
def get_task_id(search):
    """get the task id from the url"""
    parsed = urllib.parse.urlparse(search)
    parsed_dict = urllib.parse.parse_qs(parsed.query)
    store = {}
    if parsed_dict.get("task_id"):
        store["task_id"] = parsed_dict.get("task_id")[0]
    if parsed_dict.get("pre_signed_url"):
        store["pre_signed_url"] = parsed_dict.get("pre_signed_url")[0]
    return store


@callback(
    Output("container", "children"),
    Output("loading", "children"),
    [Input("store", "data")],
)
def display_simulation_data_app(store) -> dcc.Tabs:
    """display the simulation data in the app"""
    layout = dcc.Tabs([])
    try:
        data_app = get_store().get_simulation_data(store)
        component = SimulationPlotly(simulation=data_app.simulation).make_component()
        layout.children += [component]
    except Exception as e:  # pylint:disable=broad-except
        return html.Div([html.H1("Fail to generate plot"), html.H2(f"Error: {e}")])

    for monitor_name, monitor_data in data_app.monitor_data.items():
        data_plotly = DataPlotly.from_monitor_data(
            monitor_data=monitor_data, monitor_name=monitor_name
        )
        if data_plotly is None:
            continue
        component = data_plotly.make_component()
        layout.children += [component]

    layout.children += [
        dcc.Tab(
            [
                html.Div([html.H1("Solver Log")]),
                html.Div([html.Code(data_app.log, style={"whiteSpace": "pre-wrap"})]),
            ],
            label="log",
        )
    ]
    return layout, ""
