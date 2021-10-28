import pytest
import numpy as np
import pydantic
import os
import json

from tidy3d import *
from tidy3d.convert import export_old_json, load_old_monitor_data, load_solver_results
from .utils import clear_dir, clear_tmp, prepend_tmp
from .utils import SIM_FULL as SIM


@clear_tmp
def test_sim_to_old_json():
    """Test export to old-style json. Just tests that no errors are raised."""

    sim_dict = export_old_json(SIM)
    sim_js = json.dumps(sim_dict, indent=4)
    # print(sim_js)

    # Write to tmp folder in case we want to run the solver manually.
    with open(prepend_tmp("simulation.json"), "w") as fjson:
        json.dump(sim_dict, fjson, indent=4)


@clear_tmp
def test_load_old_monitor_data():
    """Test loading of an old ``monitor_data.hdf5`` file. Just tests that no errors are raised."""

    data_dict = load_old_monitor_data(SIM, "tests/data/monitor_data.hdf5")
    sim_data = load_solver_results(SIM, data_dict)
