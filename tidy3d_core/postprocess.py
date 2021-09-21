import numpy as np

from typing import Dict, Tuple
import os
import h5py

import sys
sys.path.append('../')

from .solver import solve

from tidy3d import Simulation
from tidy3d.components.data import SimulationData, monitor_data_map
from tidy3d.components.data import FieldData, FluxData, ModeData

""" Loads solver raw data dictionary into tidy3d data objects for export """

def save_solver_results(path, simulation, solver_data_dict):
	# write to file
	if os.path.exists(path):
		os.remove(path)
	json_string = simulation.json()
	with h5py.File(path, 'a') as f:
	    mon_data_grp = f.create_group('monitor_data')
	    f.attrs['json_string'] = json_string
	    for mon_name, mon_data in solver_data_dict.items():
	        mon_grp = mon_data_grp.create_group(mon_name)
	        for data_name, data_value in mon_data.items():
	            mon_grp.create_dataset(data_name, data=data_value)

def load_solver_results(simulation: Simulation, solver_data_dict: dict[str, dict[str, np.ndarray]]) -> SimulationData:

	monitor_data = {}
	for name, monitor in simulation.monitors.items():
		monitor_data_type = monitor_data_map[type(monitor)]
		monitor_data_dict = solver_data_dict[name]
		data = monitor_data_dict.pop("data")
		coords = monitor_data_dict
		mon_data = monitor_data_type(data=data, coords=coords, name=name)
		monitor_data[name] = mon_data

	return SimulationData(
		simulation=simulation,
		monitor_data=monitor_data)
