from __future__ import annotations

from tests.utils import SAMPLE_SIMULATIONS
from tidy3d import SimulationMap

# Reusable test constants
SIM_MAP_DATA = {
    "sim_A": SAMPLE_SIMULATIONS["full_fdtd"],
    "sim_B": SAMPLE_SIMULATIONS["full_fdtd"].updated_copy(run_time=2e-12),
}
print(*SIM_MAP_DATA.keys())
print(SimulationMap(keys=tuple(SIM_MAP_DATA.keys()), values=tuple(SIM_MAP_DATA.values())))
