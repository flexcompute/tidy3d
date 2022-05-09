"""Makes sure static context manager works as expected."""

import pytest
import time

import numpy as np

import tidy3d as td
from tidy3d.static import MakeStatic, StaticException, make_static


def make_sim(num_structures=100):
    """Returns a fresh simulation."""
    return td.Simulation(
        size=(5, 5, 5),
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1)),
                medium=td.Medium(permittivity=1 + np.random.random()),
            )
            for i in range(num_structures)
        ],
        sources=[
            td.PointDipole(
                center=(0, 0.5, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(
                    freq0=2e14,
                    fwidth=4e13,
                ),
            )
        ],
    )


def test_context_manager_works_static():
    """Makes sure the context manager works as expected when sim not mutated."""

    sim_original = make_sim()

    assert sim_original._hash is None, "sim hash is not None to start."

    with MakeStatic(sim_original) as sim_static:

        assert sim_static._hash is not None, "static sim._hash should not be None"

        run_time_x2 = sim_original.run_time * 2

    assert sim_original._hash is None


def test_context_manager_fails_on_mutation():
    """Makes sure the context manager raises StaticException when sim_static changed."""

    sim_original = make_sim()

    assert sim_original._hash is None, "sim hash is not None to start."

    with pytest.raises(StaticException):
        with MakeStatic(sim_original) as sim_static:
            sim_static.run_time *= 2

    assert sim_original._hash is None


def test_context_manager_fails_on_list():
    """Makes sure the context manager raises StaticException when sim_static's attr mutates."""

    sim_original = make_sim()

    assert sim_original._hash is None, "sim hash is not None to start."

    with pytest.raises(StaticException):
        with MakeStatic(sim_original) as sim_static:
            sim_static.sources[0].source_time.freq0 *= 2

    assert sim_original._hash is None


def test_decorator_works_static():
    """Make sure decorator works in static case, with many args."""

    @make_static
    def mult_and_shift_run_time(simulation, mult_factor, shift_factor=1e-12) -> float:
        """Multiply run time by factor and shift by another factor."""
        return simulation.run_time * mult_factor + shift_factor

    sim_original = make_sim()

    ret = mult_and_shift_run_time(sim_original, 2)


def test_decorator_fails_mutate():
    """Make sure decorator fails in mutated case, with many args."""

    @make_static
    def mult_and_shift_run_time_inplace(simulation, mult_factor, shift_factor=1e-12) -> None:
        """Augment simulation run time by multiplicative factor and shift by another factor."""
        simulation.run_time *= mult_factor
        simulation.run_time += shift_factor

    sim_original = make_sim()

    with pytest.raises(StaticException):
        ret = mult_and_shift_run_time_inplace(sim_original, 2)


def test_one_arg_static():
    """Make sure decorator works in static case when only one arg supplied."""

    @make_static
    def run_time_squared(simulation) -> float:
        """Return 2x simulation run_time."""
        return simulation.run_time * 2

    sim_original = make_sim()

    ret = run_time_squared(sim_original)


def test_one_arg_static():
    """Make sure decorator works in mutated case when only one arg supplied."""

    @make_static
    def run_time_squared_inplace(simulation) -> float:
        """double simulation run time."""
        simulation.run_time *= 2

    sim_original = make_sim()

    with pytest.raises(StaticException):
        ret = run_time_squared_inplace(sim_original)


def sum_of_N_timesteps(sim, N=50_000):
    """sum of len(dt) computed N times."""
    ret = 0
    for _ in range(N):
        ret += len(sim.tmesh)
    return ret


@pytest.mark.timeout(3.0)
def test_grid_run_fast():
    """Make sure repeated .grid computation is fast."""
    sim = make_sim()
    sum_of_N_timesteps_static = make_static(sum_of_N_timesteps)
    sum = sum_of_N_timesteps_static(sim, N=10_000)
