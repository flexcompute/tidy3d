import urllib

import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
from callback import *

app = dash.Dash(__name__, suppress_callback_exceptions=True)
VALID_USERNAME_PASSWORD_PAIRS = {"hello": "world!"}


def stateless_layout():
    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="store"),
            html.Div(id="container"),
        ]
    )


app.layout = stateless_layout


@app.callback(
    Output("store", "data"),
    [Input("url", "search")],
)
def get_task_id(search):
    parsed = urllib.parse.urlparse(search)
    parsed_dict = urllib.parse.parse_qs(parsed.query)
    store = {}
    if parsed_dict.get("task_id"):
        store["task_id"] = parsed_dict.get("task_id")[0]
    return store


if __name__ == "__main__":
    app.run_server(debug=True)
