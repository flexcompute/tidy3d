from abc import ABC, abstractmethod
from typing import Union
from typing_extensions import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, Output, Input

from .component import UIComponent

import sys
sys.path.append("../../../")

from tidy3d.components.data import FluxData, FluxTimeData, FieldData, Tidy3dData, FieldTimeData
from tidy3d.components.geometry import Geometry
from tidy3d.components.types import Axis
from tidy3d.log import Tidy3dKeyError


class DataPlotly(UIComponent):

    monitor_name : str

    @property
    def label(self) -> str:
        return f'monitor: {self.monitor_name}'
    
    @property
    def id(self) -> str:
        return f'monitor_{self.monitor_name}'

    @staticmethod
    def sel_by_val(data, val):
        if "re" in val.lower():
            return data.real
        if "im" in val.lower():
            return data.imag
        if "abs" in val.lower():
            return abs(data)

    def append_monitor_name(self, value):
        """makes the ids unique for this element."""
        return f'{value}_{self.monitor_name}'

    @classmethod
    def from_monitor_data(self, monitor_name: str, monitor_data):

        MAP = {
            FluxData: FluxDataPlotly,
            FluxTimeData: FluxTimeDataPlotly,
            FieldData: FieldDataPlotly,
            FieldTimeData: FieldTimeDataPlotly,
        }

        plotly_type = MAP.get(type(monitor_data))
        if not plotly_type:
            raise Tidy3dKeyError(f"could not find the monitor data type: {type(monitor_data)}.")

        return plotly_type(data=monitor_data, monitor_name=monitor_name)


class FluxDataPlotly(DataPlotly):
    """Flux in frequency domain."""

    data: FluxData

    def make_figure(self):
        """ Generate plotly figure from the current state of self."""

    def make_component(self, app) -> dcc.Tab:
        """Creates the dash component for this montor data."""

    def plotly(self):
        return px.line(x=self.data.data.coords["f"], y=self.data.data.values)


class FluxTimeDataPlotly(DataPlotly):
    """Flux in time domain."""

    data: FluxTimeData

    def make_figure(self):
        """ Generate plotly figure from the current state of self."""

    def make_component(self, app) -> dcc.Tab:
        """Creates the dash component for this montor data."""

    def plotly(self):
        return px.line(x=self.data.data.coords["t"], y=self.data.data.values)


class FieldDataPlotly(DataPlotly):
    """Flux in frequency domain."""

    data: FieldData
    field_val: str = 'Ex'
    cs_axis: Axis = 0
    cs_val: float = None
    val: str = 'abs'

    def make_figure(self):
        """ Generate plotly figure from the current state of self."""
        scalar_field_data = self.data.data_dict[self.field_val]
        xyz_string = 'xyz'[self.cs_axis]
        if self.cs_val is None:
            self.cs_val =  np.mean(scalar_field_data.data.coords[xyz_string].values)
        f0 = float(scalar_field_data.f[0])
        plotly_kwargs = {xyz_string:self.cs_val, 'field':self.field_val, 'freq':f0, 'val':self.val}
        return self.plotly(**plotly_kwargs)

    def make_component(self, app):
        """Creates the dash component."""

        component = dcc.Tab(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id=self.append_monitor_name('figure'),
                                    figure=self.make_figure(),
                                )
                            ], style={'padding': 10, 'flex': 1}
                        ),
                        html.Div(
                            [
                                html.H1(f'Viewing data for FieldMonitor: {self.monitor_name}'),
                                html.H2(f'Field component.'),
                                dcc.Dropdown(
                                    options=list(self.data.data_dict.keys()),
                                    value=self.field_val,
                                    id=self.append_monitor_name('field_dropdown'),
                                ),
                                html.H2(f'Value to plot.'),
                                dcc.Dropdown(
                                    options=['real', 'imag', 'abs'],
                                    value=self.val,
                                    id=self.append_monitor_name('val_dropdown'),
                                ),            
                                html.Br(),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            options=['x', 'y', 'z'],
                                            value='xyz'[self.cs_axis],
                                            id=self.append_monitor_name('cs_axis_dropdown'),
                                        ),
                                        dcc.Slider(
                                            min=self.data.data_dict[self.field_val].data.coords['xyz'[self.cs_axis]].values.tolist()[0],
                                            max=self.data.data_dict[self.field_val].data.coords['xyz'[self.cs_axis]].values.tolist()[-1],
                                            value=np.mean(self.data.data_dict[self.field_val].data.coords['xyz'[self.cs_axis]].values),
                                            id=self.append_monitor_name('cs_slider'),
                                        ),
                                    ],
                                )
                            ], style={'padding': 10, 'flex': 1}
                        ),
                    ], style={'display': 'flex', 'flex-direction': 'row'},
                )
            ],
            label=f'monitor: "{self.monitor_name}"'
        )

        @app.callback(
            Output(self.append_monitor_name('figure'), 'figure'),
            [
                Input(self.append_monitor_name('field_dropdown'), 'value'),
                Input(self.append_monitor_name('val_dropdown'), 'value'),
                Input(self.append_monitor_name('cs_axis_dropdown'), 'value'),
                Input(self.append_monitor_name('cs_slider'), 'value'),
            ]
        )
        def set_field(value_field, value_val, value_cs_axis, value_cs):
            self.field_val = str(value_field)
            self.val = str(value_val)
            self.cs_axis = ['x', 'y', 'z'].index(value_cs_axis)
            self.cs_val = float(value_cs)
            print('hi')
            fig = self.make_figure()
            return fig

        @app.callback(
            Output(self.append_monitor_name('cs_slider'), 'min'),
            Input(self.append_monitor_name('cs_axis_dropdown'), 'value'),
        )
        def set_min(value_cs_axis):
            self.cs_axis = ['x', 'y', 'z'].index(value_cs_axis)
            return self.data.data_dict[self.field_val].data.coords['xyz'[self.cs_axis]].values.tolist()[0]

        @app.callback(
            Output(self.append_monitor_name('cs_slider'), 'max'),
            Input(self.append_monitor_name('cs_axis_dropdown'), 'value'),
        )
        def set_max(value_cs_axis):#, value_f):
            self.cs_axis = ['x', 'y', 'z'].index(value_cs_axis)
            return self.data.data_dict[self.field_val].data.coords['xyz'[self.cs_axis]].values.tolist()[-1]

        return component


    def plotly(
        self, field: str, freq: float, val: Literal["real", "imag", "abs"], x=None, y=None, z=None
    ):
        """Creates the plotly figure given some parameters."""

        axis, position = Geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        scalar_field_data = self.data.data_dict[field]
        sel_freq = scalar_field_data.data.sel(f=freq)
        xyz_labels = ["x", "y", "z"]
        xyz_kwargs = {xyz_labels.pop(axis): position}
        sel_xyz = sel_freq.interp(**xyz_kwargs)
        sel_val = self.sel_by_val(data=sel_xyz, val=val)
        d1 = sel_val.coords[xyz_labels[0]]
        d2 = sel_val.coords[xyz_labels[1]]
        fig = go.Figure(
            data=go.Heatmap(
                x=d1,
                y=d2,
                z=sel_val.values,
                transpose=True,
                type="heatmap",
                colorscale="magma" if val in "abs" in val else "RdBu",
            )
        )
        fig.update_layout(title=f'{val}[{field}({"xyz"[axis]}={position:.2e}, f={freq:.2e})]')
        return fig

class FieldTimeDataPlotly(DataPlotly):
    """Flux in frequency domain."""

    data: FieldTimeData

    def make_figure(self):
        """ Generate plotly figure from the current state of self."""

    def make_component(self, app) -> dcc.Tab:
        """Creates the dash component for this montor data."""

    def plotly(
        self, field: str, time: float, val: Literal["real", "imag", "abs"], x=None, y=None, z=None
    ):
        axis, position = Geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        scalar_field_data = self.data.data_dict[field]
        sel_freq = scalar_field_data.data.sel(t=time)
        xyz_labels = ["x", "y", "z"]
        xyz_kwargs = {xyz_labels.pop(axis): position}
        sel_xyz = sel_freq.interp(**xyz_kwargs)
        sel_val = self.sel_by_val(data=sel_xyz, val=val)
        d1 = sel_val.coords[xyz_labels[0]]
        d2 = sel_val.coords[xyz_labels[1]]
        fig = go.Figure(
            data=go.Heatmap(
                x=d1,
                y=d2,
                z=sel_val.values,
                transpose=True,
                type="heatmap",
                colorscale="magma" if val in "abs" in val else "RdBu",
            )
        )
        fig.update_layout(title=f'{val}[{field}({"xyz"[axis]}={position:.2e}, t={time:.2e})]')
        return fig