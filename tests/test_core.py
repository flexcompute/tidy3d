import pytest
import numpy as np
import pydantic
import os

import sys

sys.path.append("./")

from tidy3d import *
import tidy3d_core as tdcore
from .utils import clear_dir
from .utils import SIM_MONITORS as SIM

TMP_DIR = "tests/tmp/"


# decorator that clears the tmp/ diretory before test
def clear_tmp(fn):
    def new_fn(*args, **kwargs):
        clear_dir(TMP_DIR)
        return fn(*args, **kwargs)

    return new_fn


def prepend_tmp(path):
    """prepents "TMP_DIR" to the path"""
    return os.path.join(TMP_DIR, path)


# where we store json files
PATH_JSON = prepend_tmp("simulation.json")

# where we store simulation data files
PATH_DATA = prepend_tmp("sim.hdf5")


def _solve_sim(simulation: Simulation) -> SimulationData:
    """solves simulation and loads results into SimulationData"""
    solver_data_dict = tdcore.solve(simulation)
    return tdcore.load_solver_results(simulation, solver_data_dict)


def _assert_same_sim_data(
    sim_data1: SimulationData, sim_data2: SimulationData, sim_reference: Simulation = None
):
    """ensure two SimulationDatas are the same"""

    # assert the Simulations are the same as each other and original
    assert sim_data1.simulation == sim_data2.simulation
    if sim_reference is not None:
        assert sim_data1.simulation == sim_reference

    # check all monitors are the same
    monitor_data1 = sim_data1.monitor_data
    monitor_data2 = sim_data2.monitor_data
    assert monitor_data1.keys() == monitor_data2.keys()
    for mon_name in monitor_data1.keys():
        mon_data1 = monitor_data1[mon_name]
        mon_data2 = monitor_data2[mon_name]
        assert mon_data1 == mon_data2

    # assert the two SimulationData are equal generally
    assert sim_data1 == sim_data2


""" tests"""


@clear_tmp
def test_flow():
    """Test entire step in pipeline"""

    """ CLIENT SIDE """

    # export simulation to json
    SIM.export(PATH_JSON)

    """ SERVER SIDE """

    # load json file
    sim_core = Simulation.load(PATH_JSON)

    # get raw results in dict of dict of np.ndarray
    solver_data_dict = tdcore.solve(sim_core)

    # load these results as SimulationData server side if you want
    sim_data_core = tdcore.load_solver_results(sim_core, solver_data_dict)

    # or, download these results to hdf5 file
    tdcore.save_solver_results(PATH_DATA, sim_core, solver_data_dict)

    """ CLIENT SIDE """

    # load the file into SimulationData
    sim_data_client = SimulationData.load(PATH_DATA)

    # make sure the SimulationData is identical
    _assert_same_sim_data(sim_data_client, sim_data_core, sim_reference=SIM)


@clear_tmp
def test_mon_data():
    """Test that exporting and loading a MonitorData gives same results"""

    # make data
    sim_data = _solve_sim(SIM)

    # make sure all monitor data are same when exported and loaded
    for mon_name, mon in SIM.monitors.items():
        mon_data = sim_data.monitor_data[mon_name]

        # export MonitorData
        mon_path = prepend_tmp(f"monitor_{mon_name}.hdf5")
        mon_data.export_as_file(mon_path)

        # load with the correct MonitorData
        mon_data_type = monitor_data_map[type(mon)]
        _mon_data = mon_data_type.load_from_file(mon_path)

        assert mon_data == _mon_data


@clear_tmp
def test_sim_data():
    """Test that exporting and loading a SimulationData gives same results"""

    # make data
    sim_data = _solve_sim(SIM)

    # save SimulationData to file
    sim_data.export(PATH_DATA)

    # load different SimulationData from file
    _sim_data = SimulationData.load(PATH_DATA)

    # make sure the SimulationData is identical
    _assert_same_sim_data(sim_data, _sim_data, sim_reference=SIM)


@clear_tmp
def test_sim_data_postprocess():
    """Tests that Simulation data exported from postprocess.py/save_solver_results is consistent same as original"""

    # make data, load into SimulationData
    solver_data = tdcore.solve(SIM)
    sim_data = tdcore.load_solver_results(SIM, solver_data)

    # save the SimulationData to file using tdcore convenience function
    tdcore.save_solver_results(PATH_DATA, SIM, solver_data)

    # load different SimulationData from file
    _sim_data = SimulationData.load(PATH_DATA)

    # make sure the SimulationData is identical
    _assert_same_sim_data(sim_data, _sim_data, sim_reference=SIM)
