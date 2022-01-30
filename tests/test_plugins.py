import pytest
import numpy as np
import pydantic

import tidy3d as td

from tidy3d.plugins import DispersionFitter
from tidy3d.plugins.dispersion.fit import _poles_to_coeffs, _coeffs_to_poles
from tidy3d.plugins.dispersion.fit import _pack_coeffs, _unpack_coeffs

from tidy3d.plugins import ModeSolver
from tidy3d.plugins import Near2Far
from tidy3d import FieldData, ScalarFieldData, FieldMonitor
from .utils import clear_tmp


def _test_near2far():
    """make sure mode solver runs"""

    def rand_data():
        return ScalarFieldData(
            x=np.linspace(-1, 1, 10),
            y=np.linspace(-1, 1, 10),
            z=np.array([0.0]),
            f=np.array([1.0]),
            values=np.random.random((10, 10, 1, 1)),
        )

    fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    data_dict = {field: rand_data() for field in fields}
    field_data = FieldData(data_dict=data_dict)

    n2f = Near2Far(field_data)
    n2f.radar_cross_section(1, 1)
    n2f.power_spherical(1, 1, 1)
    n2f.power_cartesian(1, 1, 1)
    n2f.fields_spherical(1, 1, 1)
    n2f.fields_cartesian(1, 1, 1)


def test_mode_solver():
    """make sure mode solver runs"""
    waveguide = td.Structure(
        geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0)
    )
    simulation = td.Simulation(size=(2, 2, 2), grid_size=(0.1, 0.1, 0.1), structures=[waveguide])
    plane = td.Box(center=(0, 0, 0), size=(0, 1, 1))
    ms = ModeSolver(simulation=simulation, plane=plane, freq=td.constants.C_0 / 1.5)
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        bend_radius=3.0,
        bend_axis=0,
        num_pml=(10, 10),
    )
    modes = ms.solve(mode_spec=mode_spec)


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
    fitter = DispersionFitter(wvls, n_data)
    medium, rms = fitter.fit_single()
    medium, rms = fitter.fit(num_tries=2)
    medium.to_file("tests/tmp/medium_fit.json")


def test_dispersion_load():
    """loads dispersion model from nk data file"""
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)


def test_dispersion_plot():
    """plots a medium fit from file"""
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)
    fitter.plot(medium)
