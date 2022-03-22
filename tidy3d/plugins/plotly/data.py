from abc import ABC
from typing import Union
from typing_extensions import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import sys;
sys.path.append('../../../')

from tidy3d.components.data import FluxData, FluxTimeData, FieldData
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry import Geometry
# from ...components.data import FluxData, FluxTimeData, FieldData
# from ...components.base import Tidy3dBaseModel

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
    
    def plotly(self, field:str, freq:float, val:Literal['real', 'imag', 'abs'], x=None, y=None, z=None):
        axis, position = Geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        scalar_field_data = self.data.data_dict[field]
        sel_freq = scalar_field_data.data.sel(f=freq)
        xyz_labels = ['x', 'y', 'z']
        xyz_kwargs = {xyz_labels.pop(axis): position}
        sel_xyz = sel_freq.interp(**xyz_kwargs)
        sel_val = self.sel_by_val(data=sel_xyz, val=val)
        d1=sel_val.coords[xyz_labels[0]]
        d2=sel_val.coords[xyz_labels[1]]
        fig = go.Figure(
            data=go.Heatmap(
                x = d1,
                y = d2,
                z = sel_val.values,
                transpose=True,
                type='heatmap',
                colorscale='magma' if val in 'abs' in val else 'RdBu'
            )
        )
        fig.update_layout(title=f'{val}[{field}({"xyz"[axis]}={position:.2e}, f={freq:.2e})]')
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