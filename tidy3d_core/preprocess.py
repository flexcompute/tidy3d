import numpy as np

from typing import Dict, Tuple

import sys
sys.path.append('../')

from .solver import solve

from tidy3d import Simulation

""" Loads the JSON file into Simulation and prepares data for solver """

def load_simulation_json(json_fname: str) -> Simulation:
	sim = Simulation.load(json_fname)
	return sim