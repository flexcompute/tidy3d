"""Generates sample simulation json and h5 files in the tests/sims folder"""

from __future__ import annotations

import sys
from os.path import join

sys.path.append("tests")

from utils import SAMPLE_SIMULATIONS

FPREFIX_SIMULATION_SAMPLE = join("tests", "sims")

for key, sim in SAMPLE_SIMULATIONS.items():
    # Store a simulation sample as json and hdf5
    sim.to_json(join(FPREFIX_SIMULATION_SAMPLE, key) + ".json")
    sim.to_hdf5(join(FPREFIX_SIMULATION_SAMPLE, key) + ".h5")
