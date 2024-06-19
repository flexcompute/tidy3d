#!/usr/bin/env -S poetry run python
# ruff: noqa: F401
import autograd.numpy as anp
import matplotlib.pylab as plt
from autograd import value_and_grad

import tidy3d as td
from tidy3d.web import run


def main():
    wavelength = 1.0
    freq0 = td.C_0 / wavelength

    eps_box = 2.0

    L = 10 * wavelength
    buffer = 1.0 * wavelength

    source = td.PointDipole(
        center=(-L / 2 + buffer, 0, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
        polarization="Ez",
    )

    monitor = td.FieldMonitor(
        center=(L / 2 - buffer, 0, 0),
        size=(0.0, 0.0, 0.0),
        freqs=[freq0],
        name="point",
        colocate=False,
    )

    sim = td.Simulation(
        size=(L, L, L),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10),
        structures=[],
        sources=[source],
        monitors=[monitor],
        run_time=1e-12,
    )

    size_min = 0
    size_max = L - 4 * buffer

    def get_size(param: float):
        param_01 = 0.5 * (anp.tanh(param) + 1)
        return (size_max * param_01) + (size_min * (1 - param_01))

    def make_sim(param: float) -> float:
        if param is None:
            return sim.copy()

        size_box = get_size(param)

        box = td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=(size_box, size_box, size_box)),
            medium=td.Medium(permittivity=eps_box),
        )

        return sim.updated_copy(structures=[box])

    def measure_intensity(data: td.SimulationData) -> float:
        return anp.sum(anp.array(data.get_intensity(monitor.name).values))

    @value_and_grad
    def intensity(param: float) -> float:
        sim_with_square = make_sim(param)
        data = run(sim_with_square, task_name="adjoint_quickstart", verbose=False)
        return measure_intensity(data)

    v, g = intensity(-0.5)

    print(v)
    print(g)


if __name__ == "__main__":
    main()
