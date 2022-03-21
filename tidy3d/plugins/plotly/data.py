from abc import ABC
from typing import Union
from typing_extensions import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from ...components.data import FluxData, FluxTimeData, FieldData
from ...components.base import Tidy3dBaseModel

class DataPlotly(Tidy3dBaseModel):

    @staticmethod
    def sel_by_val(data, val):
        if 're' in val.lower():
            return data.real
        if 'im' in val.lower():
            return data.imag
        if 'abs' in val.lower():
            return abs(data)

class FluxDataPlotly(DataPlotly):
    """Flux in frequency domain."""
    data : FluxData

    def plotly(self):
        return px.line(x=self.data.data.coords['f'], y=self.data.data.values)

class FluxTimeDataPlotly(DataPlotly):
    """Flux in time domain."""
    data : FluxTimeData

    def plotly(self):
        return px.line(x=self.data.data.coords['t'], y=self.data.data.values)

class FieldDataPlotly(DataPlotly):
    """Flux in frequency domain."""
    data : FieldData

    def plotly(self, field:str, freq:float, val:Literal['real', 'imag', 'abs']):
        scalar_field_data = self.data.data_dict[field]
        sel_freq = scalar_field_data.data.sel(f=freq)
        sel_val = self.sel_by_val(data=sel_freq, val=val)
        x=sel_val.coords['x'],
        y=sel_val.coords['y'],
        z=sel_val.coords['z'],
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        x = xx.flatten()
        y = yy.flatten()
        z = zz.flatten()
        value=sel_val.values.flatten()
        fig = go.Figure(data=go.Volume(
            x=x,
            y=y,
            z=z,
            value=value,
            isomin=0.1,
            isomax=0.8,
            opacity=0.1, # needs to be small to see through all surfaces
            surface_count=17, # needs to be a large number for good volume rendering
        ))
        return fig


# if __name__ == '__main__':
# import numpy as np
# import tidy3d as td
# f = np.linspace(2e14, 3e14, 1001)
# values = np.random.random((len(f),))
# data = td.FluxData(values=values, f=f)
# from tidy3d.plugins.plotly import FluxDataPlotly
# dpy = FluxDataPlotly(data=data)
# fig = dpy.plotly()
# fig.show()