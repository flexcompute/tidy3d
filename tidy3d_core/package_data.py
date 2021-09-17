import numpy as np
import xarray as xr

from typing import Dict, Tuple

import sys
sys.path.append('../')
from tidy3d.components import Simulation, Monitor, FieldMonitor, FluxMonitor, ModeMonitor
from tidy3d.components.monitor import Sampler, FreqSampler
from tidy3d.components.types import GridSize
from tidy3d.plugins.data_analyzer import SimulationData, FieldData, FluxData, ModeData

# RESERVED_KEYS = ('dir', 'fs', 'ts', 'field', 'dir', 'xs', 'ys', 'mode_index', 'component')

""" Loads fake simulation data into various monitors and returns a big SimulationData object """

def unpack_grid_size(grid_size: GridSize) -> Tuple[float, float, float]:
	""" if grid_size is supplied as single float, convert to tuple of floats """
	if isinstance(grid_size, float):
		return tuple(3*[grid_size])
	return grid_size

def make_coordinates_1d(cmin: float, cmax: float, dl: float) -> np.ndarray:
	""" returns an endpoint-inclusive array of points between cmin and cmax with spacing dl """
	return np.arange(cmin, cmax + 1e-8, dl)

def make_coordinates_3d(center, size, grid_size):
	""" gets the x,y,z coordinates of a box with center, size and discretization of grid_size """
	rmin = np.array([c - s/2.0 for (c, s) in zip(center, size)])
	rmax = np.array([c + s/2.0 for (c, s) in zip(center, size)])
	grid_size = unpack_grid_size(grid_size)
	xs, ys, zs = [make_coordinates_1d(cmin, cmax, dl) for (cmin, cmax, dl) in zip(rmin, rmax, grid_size)]
	return xs, ys, zs

def parse_sampler(sampler: Sampler) -> Tuple[str, np.ndarray]:
	""" gets the correct coordinate labels and values for a sampler """
	if isinstance(sampler, FreqSampler):
		return 'freqs', sampler.freqs
	else:
		return 'ts', sampler.times

def create_xarray(sim: Simulation) -> Dict[str, xr.DataArray]:
	""" return dictionary of {monitor name : xarray DaraArray} for simulation."""
	monitor_data = {}
	for (name, monitor) in sim.monitors.items():
		mon_data = make_monitor_data(sim.grid_size, monitor)
		mon_data.name = name # needed for holoviews
		monitor_data[name] = mon_data
	return monitor_data

def make_monitor_data(grid_size: GridSize, monitor: Monitor) -> xr.DataArray:
	""" calls different monitor DataArray creation function depending on monitor type."""
	if isinstance(monitor, FieldMonitor):
		return make_field_monitor_data(monitor, grid_size)
	elif isinstance(monitor, FluxMonitor):
		return make_flux_monitor_data(monitor)
	elif isinstance(monitor, ModeMonitor):
		return make_mode_monitor_data(monitor)

def make_fake_field(xs, ys, zs, sample_values) -> np.ndarray:
	""" constructs an artificial electromagnetic field data based loosely on dipole radiation."""
	xx, yy, zz = np.meshgrid(xs, ys, zs)
	rr = np.sqrt(xx**2 + yy**2 + zz**2)[..., None]
	ks = sample_values
	ikr = 1j * rr * ks
	scalar = np.exp(ikr) / rr**2
	scalar_x = xx[..., None] * scalar
	scalar_y = yy[..., None] * scalar
	scalar_z = zz[..., None] * scalar
	E = np.stack((scalar_x, scalar_y, scalar_z))
	H = np.stack((scalar_z-scalar_y, scalar_x-scalar_z, scalar_y-scalar_x))
	return np.stack((E, H))

def make_field_monitor_data(monitor: FieldMonitor, grid_size: GridSize) -> xr.DataArray:
	""" Converts field monitor into xr.DataArray """

	# construct the coordinates of the monitor data points
	xs, ys, zs = make_coordinates_3d(monitor.center, monitor.size, grid_size)
	Nx, Ny, Nz = len(xs), len(ys), len(zs)

	# parse information about sampler
	sampler = monitor.sampler
	Ns = len(sampler)
	sample_string, sample_values = parse_sampler(sampler)

	# make fake field data from dipole radiation
	EH_data = make_fake_field(xs, ys, zs, sample_values)

	# construct xarray out of data and coords
	return xr.DataArray(
		data=EH_data,
		coords={
			'field': ['E', 'H'],
			'component': ['x', 'y', 'z'],
			'xs': xs,
			'ys': ys,
			'zs': zs,
			sample_string: sample_values,
		})

def make_fake_flux(sample_values: np.ndarray) -> np.ndarray:
	""" create a fake flux spectrum."""
	oscillation = np.cos(sample_values)
	envelope = np.ones_like(sample_values)
	modulation = 0.3
	return modulation * oscillation + (envelope - modulation)

def make_flux_monitor_data(monitor: FluxMonitor) -> xr.DataArray:
	""" make flux data array from monitor """
	sampler = monitor.sampler
	sample_string, sample_values = parse_sampler(sampler)
	Ns = len(sampler)
	flux = make_fake_flux(sample_values)
	return xr.DataArray(
		data=flux,
		coords={
			sample_string: sample_values,
		})

def make_fake_mode(freqs: np.ndarray, Nm: int) -> np.ndarray:
	""" create a fake flux spectrum."""
	oscillation = np.exp(1j * np.array(freqs))
	envelope = np.ones_like(freqs)
	modulations = np.arange(Nm)[..., None]
	trans = modulations * oscillation + (envelope - modulations)
	ref = 1 - trans
	return np.stack((trans, ref))

def make_mode_monitor_data(monitor: ModeMonitor) -> xr.DataArray:
	sampler = monitor.sampler
	sample_string, sample_values = parse_sampler(sampler)
	mode_amps = make_fake_mode(sample_values, len(monitor.modes))
	print(mode_amps.shape)
	return xr.DataArray(
		data=mode_amps,
		coords={
			'dir': ['+', '-'],
			'mode_index': np.arange(len(monitor.modes)),
			sample_string: sample_values,
		})

