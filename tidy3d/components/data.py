import pydantic
from abc import ABC, abstractmethod
import xarray as xr
import numpy as np

# import holoviews as hv
# hv.extension('bokeh')

from .simulation import Simulation
from .monitor import Monitor, FluxMonitor, FieldMonitor, ModeMonitor
from .types import Dict

class Tidy3dData(pydantic.BaseModel):
	""" base class for data associated with a specific task."""

	class Config:
		""" sets config for all Tidy3dBaseModel objects """
		validate_all = True              # validate default values too
		extra = 'forbid'                 # forbid extra kwargs not specified in model
		validate_assignment = True       # validate when attributes are set after initialization
		arbitrary_types_allowed = True   # allow us to specify a type for an arg that is an arbitrary class (np.ndarray)
		allow_mutation = False           # dont allow one to change the data

class MonitorData(xr.DataArray, ABC):
	pass
	# @abstractmethod
	# def visualize(self, simulation):
	# 	pass


class FieldData(MonitorData):
	pass
	# def visualize(self):
	#     hv_ds = hv.Dataset(self)    
	#     image = hv_ds.to(hv.Image, kdims=["xs", "ys"], dynamic=True)
	#     return image.options(cmap='RdBu', colorbar=True)

class FluxData(MonitorData):
	pass
	
	# def visualize(self):
	#     hv_ds = hv.Dataset(self) 
	#     image = hv_ds.to(hv.Curve, 'freqs')
	#     return image

class ModeData(MonitorData):
	pass
	
	# def visualize(self):
	#     hv_ds = hv.Dataset(self)
	#     image = hv_ds.to(hv.Curve, "freqs", dynamic=True)
	#     return image

class SimulationData(Tidy3dData):

	simulation: Simulation
	monitor_data: Dict[str, MonitorData]

# maps monitor type to corresponding data type
monitor_data_map = {
	FieldMonitor: FieldData,
	FluxMonitor: FluxData,
	ModeMonitor: ModeData}
