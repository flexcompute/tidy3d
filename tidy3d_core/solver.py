import numpy as np
from typing import Dict, Tuple

import sys

sys.path.append("../")
from tidy3d import Simulation
from tidy3d.components.monitor import FieldMonitor, FluxMonitor, ModeMonitor, PermittivityMonitor
from tidy3d.components.monitor import Sampler, FreqSampler
from tidy3d.components.types import GridSize, Tuple

""" Creates fake data for the simulation and returns a monitor data dict containing all fields """

# maps monitor name to dictionary mapping data label to data value
MonitorDataDict = Dict[str, np.ndarray]
SolverDataDict = Dict[str, MonitorDataDict]

# note: "values" is a special key in the Monitor data dict, corresponds to the raw data, not coords (x,y,z, etc)


def solve(simulation: Simulation) -> SolverDataDict:
    """takes simulation and returns dictionary of dictionaries storing the data"""

    data_dict = {}
    for name, monitor in simulation.monitors.items():
        sampler_label, sampler_values = unpack_sampler(monitor.sampler)
        if isinstance(monitor, FieldMonitor):
            x, y, z = discretize_montor(simulation, monitor)
            data_array = make_fake_field_values(x, y, z, sampler_values)
            data_dict[name] = {
                "x": x,
                "y": y,
                "z": z,
                "values": data_array,
            }
        if isinstance(monitor, PermittivityMonitor):
            x, y, z = discretize_montor(simulation, monitor)
            data_array = make_fake_field_values(x, y, z, sampler_values)
            data_dict[name] = {
                "x": x,
                "y": y,
                "z": z,
                "values": np.sum(np.abs(data_array), axis=0),
            }
        elif isinstance(monitor, FluxMonitor):
            data_array = make_fake_flux_values(sampler_values)
            data_dict[name] = {
                "values": data_array,
            }
        elif isinstance(monitor, ModeMonitor):
            num_modes = len(monitor.modes)
            data_array = make_fake_mode_values(sampler_values, num_modes)
            data_dict[name] = {
                "mode_index": np.arange(num_modes),
                "values": data_array,
            }
        data_dict[name]["sampler_label"] = sampler_label
        data_dict[name]["sampler_values"] = sampler_values
    return data_dict


""" Convenience functions """


def unpack_grid_size(grid_size: GridSize) -> Tuple[float, float, float]:
    """if grid_size is supplied as single float, convert to tuple of floats"""
    if isinstance(grid_size, float):
        return tuple(3 * [grid_size])
    return grid_size


def unpack_sampler(sampler: Sampler) -> Tuple[str, np.ndarray]:
    """gets the correct coordinate labels and values for a sampler"""
    sampler_label = "f" if sampler._label == "freqs" else "t"
    sampler_values = sampler.dict()[sampler._label]
    return sampler_label, sampler_values


""" Discretization functions """


def discretize_montor(
    simulation: Simulation, mon: FieldMonitor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretizes spatial extent of a monitor"""
    grid_size = simulation.grid_size
    center = mon.center
    size = mon.size
    x, y, z = make_coordinates_3d(center, size, grid_size)
    return x, y, z


def make_coordinates_1d(cmin: float, cmax: float, dl: float) -> np.ndarray:
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


def make_fake_field_values(x, y, z, sample_values) -> np.ndarray:
    """constructs an artificial electromagnetic field data based loosely on dipole radiation."""
    xx, yy, zz = np.meshgrid(x, y, z)
    rr = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)[..., None]
    ks = sample_values
    ikr = 1j * rr * ks
    scalar = np.exp(ikr) / (rr ** 2 + 1e-3)
    scalar_x = xx[..., None] * scalar
    scalar_y = yy[..., None] * scalar
    scalar_z = zz[..., None] * scalar
    E = np.stack((scalar_x, scalar_y, scalar_z))
    H = np.stack((scalar_z - scalar_y, scalar_x - scalar_z, scalar_y - scalar_x))
    return np.stack((E, H))


def make_fake_mode_values(freqs: np.ndarray, Nm: int) -> np.ndarray:
    """create a fake flux spectrum."""
    oscillation = np.exp(1j * np.array(freqs))
    envelope = np.ones_like(freqs)
    modulations = np.arange(Nm)[..., None]
    trans = modulations * oscillation + (envelope - modulations)
    ref = 1 - trans
    return np.stack((trans, ref))


def make_fake_flux_values(sample_values: np.ndarray) -> np.ndarray:
    """create a fake flux spectrum."""
    oscillation = np.cos(sample_values)
    envelope = np.ones_like(sample_values)
    modulation = 0.3
    return modulation * oscillation + (envelope - modulation)
