import pytest
import numpy as np
import pydantic

import sys

sys.path.append("./")

import tidy3d as td

from tidy3d.plugins import DispersionFitter
from tidy3d.plugins.dispersion.fit import _poles_to_coeffs, _coeffs_to_poles
from tidy3d.plugins.dispersion.fit import _pack_coeffs, _unpack_coeffs

from tidy3d.plugins import ModeSolver


def test_mode_solver():
    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_size=0.1,
    )
    plane = td.Box(center=(0, 0, 0), size=(1, 1, 0))
    ms = ModeSolver(simulation=simulation, plane=plane, freq=1.0)
    # modes = ms.solve(mode=td.Mode(mode_index=1))


def test_coeffs():
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    a, c = _unpack_coeffs(coeffs)
    coeffs_ = _pack_coeffs(a, c)
    a_, c_ = _unpack_coeffs(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(a, a_)
    assert np.allclose(c, c_)


def test_pole_coeffs():
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    poles = _coeffs_to_poles(coeffs)
    coeffs_ = _poles_to_coeffs(poles)
    poles_ = _coeffs_to_poles(coeffs_)
    assert np.allclose(coeffs, coeffs_)


def test_dispersion():
    num_data = 10
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = DispersionFitter(wvls, n_data)
    medium, rms = fitter.fit_single()
    medium, rms = fitter.fit(num_tries=2, verbose=False)
    medium.export("tests/tmp/medium_fit.json")


def test_dispersion_load():
    fitter = DispersionFitter.load("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20, verbose=False)


def test_dispersion_plot():
    fitter = DispersionFitter.load("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20, verbose=False)
    fitter.plot(medium)
