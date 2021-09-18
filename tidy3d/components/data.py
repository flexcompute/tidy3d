import pydantic
from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
import holoviews as hv

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
	__slots__ = ()  # need this for xarray subclassing

	def sampler_label(self):
		return 'freqs' if 'freqs' in self.coords else 'times'

	@abstractmethod
	def visualize(self, simulation):
		pass


class FieldData(MonitorData):
	__slots__ = ()  # need this for xarray subclassing

	def visualize(self):
		hv_ds = hv.Dataset(self.copy())
		image = hv_ds.to(hv.Image, kdims=["xs", "ys"], dynamic=True)
		return image.options(cmap='RdBu', colorbar=True, aspect='equal')

class FluxData(MonitorData):
	__slots__ = ()  # need this for xarray subclassing
	
	def visualize(self):
		hv.extension('bokeh')		
		hv_ds = hv.Dataset(self.copy())
		image = hv_ds.to(hv.Curve, self.sampler_label())
		return image

class ModeData(MonitorData):
	__slots__ = ()  # need this for xarray subclassing
	
	def visualize(self):
		hv_ds = hv.Dataset(self.copy())
		image = hv_ds.to(hv.Curve, self.sampler_label(), dynamic=True)
		return image

class SimulationData(Tidy3dData):

	simulation: Simulation
	monitor_data: Dict[str, MonitorData]

# maps monitor type to corresponding data type
monitor_data_map = {
	FieldMonitor: FieldData,
	FluxMonitor: FluxData,
	ModeMonitor: ModeData}
