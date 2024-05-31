"""Generates sample simulation json and h5 files in the tests/sims folder"""

import sys
from os.path import join

sys.path.append("tests")

from utils import SIM_FULL

FPREFIX_SIMULATION_SAMPLE = join("tests", "sims", "simulation_sample")

# Store a simulation sample as json and hdf5
SIM_FULL.to_json(FPREFIX_SIMULATION_SAMPLE + ".json")
SIM_FULL.to_hdf5(FPREFIX_SIMULATION_SAMPLE + ".h5")
