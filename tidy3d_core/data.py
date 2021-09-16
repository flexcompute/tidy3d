import numpy as np

import sys
sys.path.append('../')
from tidy3d.components import Simulation, Monitor, FieldMonitor, FluxMonitor, ModeMonitor, FreqSampler
from tidy3d.components.types import GridSize
from tidy3d.plugins.data_analyzer import SimulationData, FieldData, FluxData, ModeData

""" Loads fake simulation data into various monitors and returns a big SimulationData object """

def unpack_grid_size(grid_size: GridSize):
	if isinstance(grid_size, float):
		dlx, dly, dlz = 3*[grid_size]
	else:
		dlx, dly, dlz = grid_size
	return dlx, dly, dlz

def make_coordinates_1d(rmin, rmax, grid_size):
	return np.arange(rmin, rmax + 1e-8, grid_size)

def make_coordinates_3d(center, size, grid_size):
	rmin = np.array([c - s/2.0 for (c, s) in zip(center, size)])
	rmax = np.array([c + s/2.0 for (c, s) in zip(center, size)])
	grid_size = unpack_grid_size(grid_size)
	xs, ys, zs = [make_coordinates_1d(cmin, cmax, dl) for (cmin, cmax, dl) in zip(rmin, rmax, grid_size)]
	return xs, ys, zs

""" Loads the JSON file into Simulation and prepares data for solver """

def create_data(sim: Simulation):
	monitor_data = {}
	for (name, monitor) in sim.monitors.items():
		print(f"working on {type(monitor)} named '{name}'")
		mon_data = make_monitor_data(sim.grid_size, monitor)
		monitor_data[name] = mon_data

	return SimulationData(
		task_id=1001,
		simulation=sim,
		data=monitor_data)

def make_monitor_data(grid_size: GridSize, monitor: Monitor):
	if isinstance(monitor, FieldMonitor):
		return make_field_monitor_data(grid_size, monitor)
	elif isinstance(monitor, FluxMonitor):
		return make_flux_monitor_data(monitor)
	elif isinstance(monitor, ModeMonitor):
		return make_mode_monitor_data(monitor)

def make_field_monitor_data(grid_size: GridSize, mon: FieldMonitor):
	xs, ys, zs = make_coordinates_3d(mon.center, mon.size, grid_size)
	Nx, Ny, Nz = len(xs), len(ys), len(zs)
	Ns = len(mon.sampler)
	const = 1.+1j if isinstance(mon.sampler, FreqSampler) else 1.0
	E = const * np.random.random((3, Nx, Ny, Nz, Ns))
	H = const * np.random.random((3, Nx, Ny, Nz, Ns))
	return FieldData(monitor=mon, xs=xs, ys=ys, zs=zs, E=E, H=H)

def make_flux_monitor_data(monitor: FluxMonitor):
	Ns = len(monitor.sampler)
	flux = np.random.random(Ns)
	return FluxData(monitor=monitor, flux=flux)

def make_mode_monitor_data(monitor: ModeMonitor):
	Ns = len(monitor.sampler)
	Nm = len(monitor.modes)
	mode_amps = (1 + 1j) * np.random.random((2, Nm, Ns))
	return ModeData(monitor=monitor, mode_amps=mode_amps)

