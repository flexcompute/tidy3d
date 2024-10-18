#!/usr/bin/env -S poetry run python
# ruff: noqa: F401

import autograd as ag
import autograd.numpy as anp
import matplotlib.pylab as plt

import tidy3d as td
from tidy3d.web import run

wavelength = 1.55
freq0 = td.C_0 / wavelength
eps_box = 2
L = 10 * wavelength
buffer = 1.0 * wavelength

size_min = 0
size_max = L - 4 * buffer


source = td.PointDipole(
    center=(-L / 2 + buffer, 0, 0),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
    polarization="Ez",
)

monitor = td.FieldMonitor(
    center=(+L / 2 - buffer, 0, 0),
    size=(0.0, 1.0, 1.0),
    freqs=[freq0],
    name="field_monitor",
    colocate=False,
)

sim = td.Simulation(
    size=(L, L, L),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25),
    structures=[],
    sources=[source],
    monitors=[monitor],
    run_time=120 / freq0,
)


def make_sim(param: float) -> float:
    param_01 = 0.5 * (anp.tanh(param) + 1)
    size_box = (size_max * param_01) + (size_min * (1 - param_01))

    box = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(size_box, size_box, size_box)),
        medium=td.Medium(permittivity=eps_box),
    )

    return sim.updated_copy(structures=[box])


@ag.value_and_grad
def obj_fun(param: float) -> float:
    sim = make_sim(param)

    sim_data = run(sim, task_name="dataarray_dbg", verbose=False)
    flux = sim_data[monitor.name].flux
    return flux.item()


v, g = obj_fun(-0.5)

print(v)
print(g)
