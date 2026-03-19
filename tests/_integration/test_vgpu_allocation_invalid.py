"""Integration test: verify that vgpu_allocation=9 raises ValueError."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import tidy3d as td

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import configure_integration_environment

configure_integration_environment()

freq0 = td.C_0 / 0.75

sim = td.Simulation(
    size=(4, 3, 3),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25),
    structures=[
        td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=(1.5, 1.5, 1.5)),
            medium=td.Medium(permittivity=2.0),
        )
    ],
    sources=[
        td.PointDipole(
            center=(-1.5, 0, 0),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
            polarization="Ey",
        )
    ],
    monitors=[
        td.FieldMonitor(
            size=(td.inf, td.inf, 0),
            freqs=[freq0],
            name="fields",
            colocate=True,
        )
    ],
    run_time=120 / freq0,
)

print(
    f"simulation grid is shaped {sim.grid.num_cells} "
    f"for {int(np.prod(sim.grid.num_cells) / 1e6)} million cells."
)

print("\n--- Testing vgpu_allocation=9 (should raise ValueError) ---")
try:
    data = td.web.run(
        sim,
        task_name="integration_vgpu_invalid",
        path="data/data_invalid.hdf5",
        verbose=True,
        vgpu_allocation=9,
    )
    raise AssertionError("Expected ValueError was not raised")
except ValueError as e:
    print(f"ValueError caught as expected: {e}")

print("PASSED")
