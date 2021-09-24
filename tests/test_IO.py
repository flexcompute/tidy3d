import pydantic
import numpy as np
import os
from time import time

import sys

sys.path.append("./")

from tidy3d import *
from .utils import SIM_FULL as SIM
from .utils import clear_tmp


def test_simulation_load_export():
    path = "tests/tmp/simulation.json"
    SIM.export(path)
    SIM2 = Simulation.load(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


def test_validation_speed():

    sizes_bytes = []
    times_sec = []
    path = "tests/tmp/simulation.json"

    sim_base = SIM
    N_tests = 10
    num_structures = np.logspace(0, 4, N_tests).astype(int)

    for n in num_structures:
        S = SIM.copy()
        S.structures = n * [SIM.structures[0]]

        S.export(path)
        time_start = time()
        _S = Simulation.load(path)
        time_validate = time() - time_start
        times_sec.append(time_validate)
        assert S == _S

        size = os.path.getsize(path)
        sizes_bytes.append(size)

        print(f"{n} structures \t {size:.1e} bytes \t {time_validate:.1f} seconds to validate")


@clear_tmp
def test_yaml():
    path = "tests/tmp/simulation.json"
    SIM.export(path)
    sim = Simulation.load(path)
    path1 = "tests/tmp/simulation.yaml"
    sim.export_yaml(path1)
    sim1 = Simulation.load_yaml(path1)
    assert sim1 == sim
