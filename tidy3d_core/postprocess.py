import numpy as np

from typing import Dict, Tuple
import os
import h5py
import xarray as xr
import sys

sys.path.append("../")

from .solver import solve, SolverDataDict

from tidy3d import Simulation
from tidy3d.components.data import SimulationData, monitor_data_map
from tidy3d.components.data import FieldData, FluxData, ModeData
from tidy3d.components.monitor import Sampler

""" Loads solver raw data dictionary into forms to use later """


def load_solver_results(simulation: Simulation, solver_data_dict: SolverDataDict) -> SimulationData:
    """load the solver_data_dict and simulation into SimulationData"""

    # constuct monitor_data dictionary
    monitor_data = {}
    for name, monitor in simulation.monitors.items():

        # get the type of the MonitorData
        monitor_data_type = monitor_data_map[type(monitor)]

        # separate coordinates and data from the SolverDataDict
        monitor_data_dict = solver_data_dict[name]

        monitor_data[name] = monitor_data_type(**monitor_data_dict)

    return SimulationData(simulation=simulation, monitor_data=monitor_data)


def save_solver_results(path: str, sim: Simulation, solver_dict: SolverDataDict) -> None:
    """save the solver_data_dict and simulation json to file"""

    # create SimulationData object
    sim_data = load_solver_results(sim, solver_dict)

    # export the solver data to path
    sim_data.export(path)
    # potentially: export HTML for viz or other files here?
