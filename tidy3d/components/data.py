import pydantic
from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
import holoviews as hv
import os
import json

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
		""" make interactive plot"""
		pass

	def export(self, path: str) -> None:
		""" Export MonitorData to hdf5 file """
		self.to_netcdf(path=path, engine='h5netcdf', invalid_netcdf=True)

	@classmethod
	def load(cls, path: str):
		""" Load MonitorData from hdf5 file """
		data_array = xr.open_dataarray(path, engine='h5netcdf')
		return cls(data_array)

class FieldData(MonitorData):
	__slots__ = ()  # need this for xarray subclassing

	def visualize(self):
		""" make interactive plot"""
		hv_ds = hv.Dataset(self.copy())
		image = hv_ds.to(hv.Image, kdims=["xs", "ys"], dynamic=True)
		return image.options(cmap='RdBu', colorbar=True, aspect='equal')

class FluxData(MonitorData):
	__slots__ = ()  # need this for xarray subclassing
	
	def visualize(self):
		""" make interactive plot"""
		hv.extension('bokeh')		
		hv_ds = hv.Dataset(self.copy())
		image = hv_ds.to(hv.Curve, self.sampler_label())
		return image

class ModeData(MonitorData):
	__slots__ = ()  # need this for xarray subclassing
	
	def visualize(self):
		""" make interactive plot"""
		hv_ds = hv.Dataset(self.copy())
		image = hv_ds.to(hv.Curve, self.sampler_label(), dynamic=True)
		return image

# maps monitor type to corresponding data type
monitor_data_map = {
	FieldMonitor: FieldData,
	FluxMonitor: FluxData,
	ModeMonitor: ModeData}

class SimulationData(Tidy3dData):

	simulation: Simulation
	monitor_data: Dict[str, MonitorData]

	def export(self, path_base: str, sim_fname: str='simulation.json') -> None:
		""" Export all data to files"""

		# write simulation.json to file
		sim_path = os.path.join(path_base, sim_fname)
		get_mon_path = lambda name: os.path.join(path_base, name)
		self.simulation.export(sim_path)

		# write monitor data to file
		for mon_name, mon_data in self.monitor_data.items():
			mon_path = os.path.join(path_base, f"monitor_data_{mon_name}.hdf5")
			mon_data.export(path=mon_path)

	@classmethod
	def load(cls, path_base: str, sim_fname='simulation.json'):
		""" Load SimulationData from files"""

		# load simulation
		simulation_json_path = os.path.join(path_base, sim_fname)
		simulation = Simulation.load(simulation_json_path)

		# load monitor data
		monitor_data = {}
		for f in os.listdir(path_base):

			# for all monitor data files
			if 'monitor_data' in f:

				# open the dataset as an xr.dataArray and get its name
				data_path = os.path.join(path_base, f)
				data_array = xr.open_dataarray(data_path, engine='h5netcdf')
				mon_name = data_array.name

				# from the name, lookup the monitor type and get monitor data type
				mon_type = type(simulation.monitors[mon_name])
				mon_data_type = monitor_data_map[mon_type]

				# create a MonitorData instance from the MonitorData type and DataArray
				mon_data = mon_data_type(data_array)
				monitor_data[mon_name] = mon_data

		return cls(simulation=simulation, monitor_data=monitor_data)


