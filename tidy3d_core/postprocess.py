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

""" Loads solver raw data dictionary into forms to use later """


def load_solver_results(simulation: Simulation, solver_data_dict: SolverDataDict) -> SimulationData:
    """load the solver_data_dict and simulation into SimulationData"""

    # constuct monitor_data dictionary
    monitor_data = {}
    for name, monitor in simulation.monitors.items():

        # get the type of the MonitorData
        monitor_data_type = monitor_data_map[type(monitor)]

        # separate coordinates and data from the SolverDataDict
        monitor_data_dict = solver_data_dict[name].copy()
        data = monitor_data_dict.pop("data")
        coords = monitor_data_dict

        # construct MonitorData and add to dictionary
        data = xr.DataArray(data=data, coords=coords, name=name)
        mon_data = monitor_data_type(data=data)
        monitor_data[name] = mon_data

    return SimulationData(simulation=simulation, monitor_data=monitor_data)


def save_solver_results(
    path: str, simulation: Simulation, solver_data_dict: SolverDataDict
) -> None:
    """save the solver_data_dict and simulation json to file"""

    # create SimulationData object
    sim_data = load_solver_results(simulation, solver_data_dict)

    # export the solver data to path
    sim_data.export(path)
    # potentially: export HTML for viz or other files here?

    """ or, you can do it manually if you dont want to use export(), see below """
    # if os.path.exists(path):
    # 	os.remove(path)
    # json_string = simulation.json()
    # with h5py.File(path, 'a') as f:
    # 	mon_data_grp = f.create_group('monitor_data')
    # 	f.attrs['json_string'] = json_string
    # 	for mon_name, mon_data in solver_data_dict.items():
    # 		mon_grp = mon_data_grp.create_group(mon_name)
    # 		for data_name, data_value in mon_data.items():
    # 			mon_grp.create_dataset(data_name, data=data_value)
