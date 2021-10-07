""" Generates data"""
from typing import Dict, Tuple
import sys

import numpy as np

sys.path.append("../")

from tidy3d import Simulation
from tidy3d.components.monitor import VectorFieldMonitor, AbstractFluxMonitor
from tidy3d.components.monitor import ModeMonitor, FreqMonitor, TimeMonitor, PermittivityMonitor
from tidy3d.components.types import GridSize, Tuple, Numpy

""" Creates fake data for the simulation and returns a monitor data dict containing all fields """

# maps monitor name to dictionary mapping data label to data value
MonitorDataDict = Dict[str, Numpy]
SolverDataDict = Dict[str, MonitorDataDict]

# note: "values" is a special key in the Monitor data dict, corresponds to the raw data, not coords


def solve(simulation: Simulation) -> SolverDataDict:
    """takes simulation and returns dictionary of dictionaries storing the data"""

    data_dict = {}
    for name, monitor in simulation.monitors.items():
        if isinstance(monitor, FreqMonitor):
            sampler_values = np.array(monitor.freqs)
            sampler_label = "f"
            value_fn = lambda x: x
        elif isinstance(monitor, TimeMonitor):
            sampler_values = np.array(monitor.times)
            sampler_label = "t"
            value_fn = np.real
        if isinstance(monitor, VectorFieldMonitor):
            x, y, z = discretize_monitor(simulation, monitor)
            data_array = make_fake_field_values(x, y, z, sampler_values)
            Nx, Ny, Nz = len(x), len(y), len(z)

            num_fields = data_array.shape[0]
            num_components = data_array.shape[1]
            x_expanded = x * np.ones((num_fields, num_components, Nx))
            y_expanded = y * np.ones((num_fields, num_components, Ny))
            z_expanded = z * np.ones((num_fields, num_components, Nz))
            data_dict[name] = {
                "x": x_expanded,
                "y": y_expanded,
                "z": z_expanded,
                "values": data_array,
                sampler_label: value_fn(sampler_values),
            }
            if isinstance(monitor, PermittivityMonitor):
                data_dict[name]["values"] = data_dict[name]["values"].sum(axis=0)
                x_expanded = x * np.ones((num_components, Nx))
                y_expanded = y * np.ones((num_components, Ny))
                z_expanded = z * np.ones((num_components, Nz))
                data_dict[name]["x"] = x_expanded
                data_dict[name]["y"] = y_expanded
                data_dict[name]["z"] = z_expanded
        elif isinstance(monitor, AbstractFluxMonitor):
            data_array = make_fake_flux_values(sampler_values)
            data_dict[name] = {"values": data_array, sampler_label: value_fn(sampler_values)}
        elif isinstance(monitor, ModeMonitor):
            num_modes = len(monitor.modes)
            data_array = make_fake_mode_values(sampler_values, num_modes)
            data_dict[name] = {
                "mode_index": np.arange(num_modes),
                "values": data_array,
                sampler_label: value_fn(sampler_values),
            }
        data_dict[name]["monitor_name"] = name
    return data_dict


""" Convenience functions """


def unpack_grid_size(grid_size: GridSize) -> Tuple[float, float, float]:
    """if grid_size is supplied as single float, convert to tuple of floats"""
    if isinstance(grid_size, float):
        return tuple(3 * [grid_size])
    return grid_size


""" Discretization functions """


def discretize_monitor(
    simulation: Simulation, mon: VectorFieldMonitor
) -> Tuple[Numpy, Numpy, Numpy]:
    """Discretizes spatial extent of a monitor"""
    grid_size = simulation.grid_size
    center = mon.center
    size = mon.size
    x, y, z = make_coordinates_3d(center, size, grid_size)
    return x, y, z


def make_coordinates_1d(cmin: float, cmax: float, dl: float) -> Numpy:
    """returns an endpoint-inclusive array of points between cmin and cmax with spacing dl"""
    return np.arange(cmin, cmax + 1e-8, dl)


def make_coordinates_3d(center, size, grid_size):
    """gets the x,y,z coordinates of a box with center, size and discretization of grid_size"""
    rmin = np.array([c - s / 2.0 for (c, s) in zip(center, size)])
    rmax = np.array([c + s / 2.0 for (c, s) in zip(center, size)])
    grid_size = unpack_grid_size(grid_size)
    x, y, z = [
        make_coordinates_1d(cmin, cmax, dl) for (cmin, cmax, dl) in zip(rmin, rmax, grid_size)
    ]
    return x, y, z


""" Fake data constructures, to be replaced by actual solver"""


def make_fake_field_values(x, y, z, sample_values) -> Numpy:
    """constructs an artificial electromagnetic field data based loosely on dipole radiation."""
    xx, yy, zz = np.meshgrid(x, y, z)
    rr = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)[..., None]
    ikr = 1j * rr * sample_values
    scalar = np.exp(ikr) / (rr ** 2 + 1e-3)
    scalar_x = xx[..., None] * scalar
    scalar_y = yy[..., None] * scalar
    scalar_z = zz[..., None] * scalar
    E = np.stack((scalar_x, scalar_y, scalar_z))
    H = np.stack((scalar_z - scalar_y, scalar_x - scalar_z, scalar_y - scalar_x))
    return np.stack((E, H))


def make_fake_mode_values(freqs: Numpy, num_modes: int) -> Numpy:
    """create a fake flux spectrum."""
    oscillation = np.exp(1j * np.array(freqs))
    envelope = np.ones_like(freqs)
    modulations = np.arange(num_modes)[..., None]
    trans = modulations * oscillation + (envelope - modulations)
    ref = 1 - trans
    return np.stack((trans, ref))


def make_fake_flux_values(sample_values: Numpy) -> Numpy:
    """create a fake flux spectrum."""
    oscillation = np.cos(sample_values)
    envelope = np.ones_like(sample_values)
    modulation = 0.3
    return modulation * oscillation + (envelope - modulation)
