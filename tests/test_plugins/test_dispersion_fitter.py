import pytest
import numpy as np
import matplotlib.pyplot as plt
import pydantic

import tidy3d as td

from tidy3d.plugins.dispersion import DispersionFitter
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.solver import compute_modes
from tidy3d import FieldData, ScalarFieldDataArray, FieldMonitor
from tidy3d.plugins.smatrix.smatrix import Port, ComponentModeler
from tidy3d.plugins.smatrix.smatrix import ComponentModeler
from ..utils import clear_tmp, run_emulated


def _test_coeffs():
    """make sure pack_coeffs and unpack_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    a, c = _unpack_coeffs(coeffs)
    coeffs_ = _pack_coeffs(a, c)
    a_, c_ = _unpack_coeffs(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(a, a_)
    assert np.allclose(c, c_)


def _test_pole_coeffs():
    """make sure coeffs_to_poles and poles_to_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    poles = _coeffs_to_poles(coeffs)
    coeffs_ = _poles_to_coeffs(poles)
    poles_ = _coeffs_to_poles(coeffs_)
    assert np.allclose(coeffs, coeffs_)


@clear_tmp
def test_dispersion():
    """performs a fit on some random data"""
    num_data = 10
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = DispersionFitter(wvl_um=wvls, n_data=n_data)
    medium, rms = fitter._fit_single()
    medium, rms = fitter.fit(num_tries=2)
    medium.to_file("tests/tmp/medium_fit.json")

    k_data = np.random.random(num_data)
    fitter = DispersionFitter(wvl_um=wvls, n_data=n_data, k_data=k_data)


def test_dispersion_load():
    """loads dispersion model from nk data file"""
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)


def test_dispersion_plot():
    """plots a medium fit from file"""
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)
    fitter.plot(medium)


def test_dispersion_set_wvg_range():
    """set wavelength range function"""
    num_data = 50
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = DispersionFitter(wvl_um=wvls, n_data=n_data)

    wvl_min = np.random.random(1)[0] * 0.5 + 1
    wvl_max = wvl_min + 0.5
    fitter = fitter.copy(update=dict(wvl_range=[wvl_min, wvl_max]))
    assert len(fitter.freqs) < num_data
    medium, rms = fitter.fit(num_tries=2)
