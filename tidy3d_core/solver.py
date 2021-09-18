import numpy as np

import sys
sys.path.append('../')
from tidy3d import Simulation
from tidy3d.components.monitor import FieldMonitor, FluxMonitor, ModeMonitor, Sampler, FreqSampler
from tidy3d.components.types import GridSize, Tuple

""" Creates fake data for the simulation and returns a monitor data dict containing all fields """

def solve(simulation: Simulation) -> dict[str, dict[str, np.ndarray]]:
	""" takes simulation and returns dictionary of dictionaries storing the data """

	data_dict = {}
	for name, monitor in simulation.monitors.items():
		sample_name, sample_values = unpack_sampler(monitor.sampler)
		if isinstance(monitor, FieldMonitor):
			xs, ys, zs = discretize_montor(simulation, monitor)
			data_array = make_fake_field_data(xs, ys, zs, sample_values)
			data_dict[name] = {
				'field': ['E', 'H'],
				'component': ['x', 'y', 'z'],
				'xs':xs,
				'ys':ys,
				'zs':zs,
				sample_name:sample_values,
				'data':data_array
			}
		elif isinstance(monitor, FluxMonitor):
			data_array = make_fake_flux_data(sample_values)
			data_dict[name] = {
				'data': data_array,
				sample_name:sample_values,
			}
		elif isinstance(monitor, ModeMonitor):
			num_modes = len(monitor.modes)
			data_array = make_fake_mode_data(sample_values, num_modes)
			data_dict[name] = {
				'direction': ['+', '-'],
				'data': data_array,
				'mode_index': np.arange(num_modes),
				sample_name:sample_values,
			}
	return data_dict

""" Convenience functions """

def unpack_grid_size(grid_size: GridSize) -> Tuple[float, float, float]:
	""" if grid_size is supplied as single float, convert to tuple of floats """
	if isinstance(grid_size, float):
		return tuple(3*[grid_size])
	return grid_size

def unpack_sampler(sampler: Sampler) -> Tuple[str, np.ndarray]:
	""" gets the correct coordinate labels and values for a sampler """
	if isinstance(sampler, FreqSampler):
		return 'freqs', sampler.freqs
	else:
		return 'times', sampler.times

""" Discretization functions """

def discretize_montor(simulation: Simulation, mon: FieldMonitor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	""" Discretizes spatial extent of a monitor """
	grid_size = simulation.grid_size
	center = mon.center
	size = mon.size
	xs, ys, zs = make_coordinates_3d(center, size, grid_size)
	return xs, ys, zs

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

""" Fake data constructures, to be replaced by actual solver"""

def make_fake_field_data(xs, ys, zs, sample_values) -> np.ndarray:
	""" constructs an artificial electromagnetic field data based loosely on dipole radiation."""
	xx, yy, zz = np.meshgrid(xs, ys, zs)
	rr = np.sqrt(xx**2 + yy**2 + zz**2)[..., None]
	ks = sample_values
	ikr = 1j * rr * ks
	scalar = np.exp(ikr) / (rr**2 + 1e-3)
	scalar_x = xx[..., None] * scalar
	scalar_y = yy[..., None] * scalar
	scalar_z = zz[..., None] * scalar
	E = np.stack((scalar_x, scalar_y, scalar_z))
	H = np.stack((scalar_z-scalar_y, scalar_x-scalar_z, scalar_y-scalar_x))
	return np.stack((E, H))

def make_fake_mode_data(freqs: np.ndarray, Nm: int) -> np.ndarray:
	""" create a fake flux spectrum."""
	oscillation = np.exp(1j * np.array(freqs))
	envelope = np.ones_like(freqs)
	modulations = np.arange(Nm)[..., None]
	trans = modulations * oscillation + (envelope - modulations)
	ref = 1 - trans
	return np.stack((trans, ref))

def make_fake_flux_data(sample_values: np.ndarray) -> np.ndarray:
	""" create a fake flux spectrum."""
	oscillation = np.cos(sample_values)
	envelope = np.ones_like(sample_values)
	modulation = 0.3
	return modulation * oscillation + (envelope - modulation)
