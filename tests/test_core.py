import pytest
import numpy as np
import pydantic
import os

import sys
sys.path.append('./')

from tidy3d import *
import tidy3d_core as tdcore

SIM = Simulation(
	size=(2.0, 2.0, 2.0),
	grid_size=.1,
	monitors={
		"field_freq": FieldMonitor(size=(1,1,1), center=(0,1,0), sampler=FreqSampler(freqs=[1,2, 5, 7, 8])),
		"field_time": FieldMonitor(size=(1,1,0), center=(1,0,0), sampler=TimeSampler(times=[1])),
		"flux_freq": FluxMonitor(size=(1,1,0), center=(0,0,0), sampler=FreqSampler(freqs=[1,2, 5, 9])),
		"flux_time": FluxMonitor(size=(1,1,0), center=(0,0,0), sampler=TimeSampler(times=[1,2, 3])),
		"mode": ModeMonitor(size=(1,1,0), center=(0,0,0), sampler=FreqSampler(freqs=[1.90,2.01, 2.2]), modes=[Mode(mode_index=1)])
	},
)

TMP_DIR = 'tests/tmp/'

def _clear_dir(path=TMP_DIR):
	for f in os.listdir(path):
		os.remove(os.path.join(path, f))	

# decorator that clears the tmp/ diretory before test
def clear_tmp(fn):
	def new_fn(*args, **kwargs):
		_clear_dir(TMP_DIR)
		return fn(*args, **kwargs)
	return new_fn
 
def prepend_tmp(path):
	return os.path.join(TMP_DIR, path)

PATH_JSON = prepend_tmp('simulation.json')
PATH_DATA = prepend_tmp('sim.hdf5')

""" each stage of the flow """

def solve_sim(sim):
	solver_data = tdcore.solve(SIM)
	sim_data = tdcore.load_solver_results(SIM, solver_data)
	return sim_data


@clear_tmp
def test_flow_object_only():
	sim_data = solve_sim(SIM)

@clear_tmp
def test_flow_file_input():
	SIM.export(PATH_JSON)
	sim_core = tdcore.load_simulation_json(PATH_JSON)
	sim_data = solve_sim(sim_core)

@clear_tmp
def _test_flow_file_output():
	sim_data = solve_sim(SIM)  
	for mon_name, mon_data in sim_data.monitor_data.items():
		path = prepend_tmp(f'monitor_data_{mon_name}')
		mon_data.export(path=path)
		_mon_data = type(mon_data).load(path=path)

		assert mon_data == _mon_data

@clear_tmp
def _test_flow_file_output():
	sim_data = solve_sim(SIM)   
	# sim_data.export(path=PATH_DATA)
	# _sim_data = SimulationData.load(PATH_DATA)

	assert sim_data == _sim_data

@clear_tmp
def test_flow_file_all():

	SIM.export(PATH_JSON)
	sim_core = tdcore.load_simulation_json(PATH_JSON)
	solver_data_dict = tdcore.solve(sim_core)
	tdcore.save_solver_results(PATH_DATA, sim_core, solver_data_dict)
	_sim_data = SimulationData.load(PATH_DATA)

	import pdb; pdb.set_trace()

	assert _sim_data.simulation == SIM
