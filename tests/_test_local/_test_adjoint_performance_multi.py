import pytest
import numpy as np
import os
import sys
from memory_profiler import profile
import matplotlib.pylab as plt
import time
import jax

import jax.numpy as jnp
from ..utils import run_emulated

import tidy3d as td
import tidy3d.plugins.adjoint as tda
from tidy3d.plugins.adjoint.web import run_local as run

import cProfile

# name of the output monitor used in tests
MNT_NAME = "field"

WAVELENGTH = 1.0
BOX_LENGTH = 1.0
SPACE = 1.0
FREQ0 = td.C_0 / WAVELENGTH
N_SIDE = 15

surface_area = 6 * (N_SIDE * BOX_LENGTH**2)


def make_sim(box_length) -> tda.JaxSimulation:
    """Construt a simulation out of some input parameters."""

    num_structures = int(N_SIDE**2)
    sim_length = N_SIDE * (BOX_LENGTH + SPACE)
    med = tda.JaxMedium(permittivity=2.0)

    xmin = (-sim_length + SPACE + BOX_LENGTH) / 2
    ymin = (-sim_length + SPACE + BOX_LENGTH) / 2

    structures = []
    for i in range(N_SIDE):
        x0 = xmin + i * (SPACE + BOX_LENGTH)
        for j in range(N_SIDE):
            y0 = ymin + j * (SPACE + BOX_LENGTH)
            box = tda.JaxBox(size=(box_length, box_length, box_length), center=(x0, y0, 0))
            struct = tda.JaxStructure(geometry=box, medium=med)
            structures.append(struct)

    # ModeMonitors
    mnt = td.FieldMonitor(
        size=(0, 0, 0),
        center=(0, 0, (BOX_LENGTH + SPACE / 2) / 2),
        freqs=[FREQ0],
        name=MNT_NAME,
    )

    sim = tda.JaxSimulation(
        size=(sim_length, sim_length, 2 * BOX_LENGTH + 2 * SPACE),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        input_structures=structures,
        output_monitors=(mnt,),
        normalize_index=None,
    )

    return sim


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.plugins.adjoint.web as adjoint_web

    monkeypatch.setattr(adjoint_web, "tidy3d_run_fn", run_emulated)


@profile
def test_large_custom_medium(use_emulated_run):
    def f(box_length):
        sim = make_sim(box_length)
        new_size = list(sim.size)
        new_size[0] += 1.0
        sim2 = sim.updated_copy(size=new_size)
        sim_data = run(sim2, task_name="test")
        fld = sim_data[MNT_NAME].Ez
        fld2 = fld.copy()
        fld2 = fld2.values
        return jnp.sum(jnp.abs(jnp.array(fld2)))

    with cProfile.Profile() as pr:
        grad_f = jax.grad(f)
        df_eps_values = grad_f(BOX_LENGTH)
        pr.print_stats(sort="cumtime")
        pr.dump_stats("results.prof")

    # with cProfile.Profile() as pr:
    #     res = f(BOX_LENGTH)
    #     res2 = f(BOX_LENGTH)
    #     pr.print_stats(sort="cumtime")
    #     pr.dump_stats("results.prof")
