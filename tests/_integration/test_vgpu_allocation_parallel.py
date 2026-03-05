"""Integration test: submit two simulations in parallel with vgpu_allocation=4."""

from __future__ import annotations

import os

import numpy as np

import tidy3d as td
import tidy3d.web as web

td.config.web.enable_caching = False
td.config.local_cache.enabled = False

web.configure(os.environ["TIDY3D_VGPU_API_KEY"])

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

print("\n--- Submitting two 4-GPU tasks concurrently ---")
batch_data = td.web.run_async(
    {
        "integration_vgpu_4gpu_a": sim,
        "integration_vgpu_4gpu_b": sim,
    },
    path_dir="data",
    verbose=True,
    vgpu_allocation=4,
)

print("\n--- Both tasks completed ---")
for task_name, sim_data in batch_data.items():
    print(f"  {task_name}: status=success, monitors={list(sim_data.monitor_data.keys())}")

print("PASSED")
