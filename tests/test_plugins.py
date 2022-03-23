import pytest
import numpy as np
import pydantic

import tidy3d as td

from tidy3d.plugins import DispersionFitter

from tidy3d.plugins import ModeSolver
from tidy3d.plugins import Near2Far
from tidy3d import FieldData, ScalarFieldData, FieldMonitor
from .utils import clear_tmp


def test_near2far():
    """make sure Near2Far runs"""

    center = (0, 0, 0)
    size = (2, 2, 2)
    f0 = 1
    monitors = FieldMonitor(size=size, center=center, freqs=[f0], name="near_field").surfaces()

    sim_size = (5, 5, 5)
    dl = 0.1
    sim = td.Simulation(size=sim_size, grid_size=[dl, dl, dl], monitors=monitors, run_time=1e-12)

    def rand_data():
        return ScalarFieldData(
            x=np.linspace(-1, 1, 10),
            y=np.linspace(-1, 1, 10),
            z=np.linspace(-1, 1, 10),
            f=[f0],
            values=np.random.random((10, 10, 10, 1)),
        )

    fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    data_dict = {field: rand_data() for field in fields}
    field_data = FieldData(data_dict=data_dict)

    data_dict_mon = {mon.name: field_data for mon in monitors}
    sim_data = td.SimulationData(simulation=sim, monitor_data=data_dict_mon)

    n2f = Near2Far.from_surface_monitors(
        sim_data=sim_data,
        monitors=monitors,
        normal_dirs=["-", "+", "-", "+", "-", "+"],
        frequency=f0,
    )

    # single inputs
    n2f.radar_cross_section(1, 1)
    n2f.power_spherical(1, 1, 1)
    n2f.power_cartesian(1, 1, 1)
    n2f.fields_spherical(1, 1, 1)
    n2f.fields_cartesian(1, 1, 1)

    # vectorized inputs
    pts1 = [0, 1]
    pts2 = [0, 1, 2]
    pts3 = [3, 4, 5]
    n2f.radar_cross_section(pts1, pts2)
    n2f.power_spherical(1, pts2, pts3)
    n2f.power_cartesian(pts1, pts2, pts3)
    n2f.fields_spherical(1, pts2, pts3)
    n2f.fields_cartesian(pts1, pts2, pts3)


def test_mode_solver():
    """make sure mode solver runs"""
    waveguide = td.Structure(
        geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0)
    )
    simulation = td.Simulation(
        size=(2, 2, 2), grid_size=(0.1, 0.1, 0.1), structures=[waveguide], run_time=1e-12
    )
    plane = td.Box(center=(0, 0, 0), size=(0, 1, 1))
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        bend_radius=3.0,
        bend_axis=0,
        num_pml=(10, 10),
    )
    ms = ModeSolver(
        simulation=simulation, plane=plane, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()


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
    fitter.wvl_range = [wvl_min, wvl_max]
    assert len(fitter.freqs) < num_data
    medium, rms = fitter.fit(num_tries=2)
