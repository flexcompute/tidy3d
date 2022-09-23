import pytest
import numpy as np
import matplotlib.pyplot as plt
import pydantic

import gdspy
import tidy3d as td

from tidy3d.plugins import DispersionFitter
from tidy3d.plugins import ModeSolver
from tidy3d.plugins.mode.solver import compute_modes
from tidy3d import FieldData, ScalarFieldDataArray, FieldMonitor
from tidy3d.plugins.smatrix.smatrix import Port, ComponentModeler
from tidy3d.plugins.smatrix.smatrix import ComponentModeler
from ..utils import clear_tmp, run_emulated

WAVEGUIDE = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0))
PLANE = td.Box(center=(0, 0, 0), size=(1, 0, 1))


def test_mode_solver_simple():
    """Simple mode solver run (with symmetry)"""

    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(1, 0, -1),
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        filter_pol="tm",
        precision="double",
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()


def test_mode_solver_angle_bend():
    """Run mode solver with angle and bend and symmetry"""
    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(-1, 0, 1),
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        bend_radius=3,
        bend_axis=0,
        angle_theta=np.pi / 3,
        angle_phi=np.pi,
    )
    # put plane entirely in the symmetry quadrant rather than sitting on its center
    plane = td.Box(center=(0, 0.5, 0), size=(1, 0, 1))
    ms = ModeSolver(
        simulation=simulation, plane=plane, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()
    # Plot field
    _, ax = plt.subplots(1)
    ms.plot_field("Ex", ax=ax, mode_index=1)

    # Create source and monitor
    st = td.GaussianPulse(freq0=1.0, fwidth=1.0)
    msource = ms.to_source(source_time=st, direction="-")
    mmonitor = ms.to_monitor(freqs=[1.0, 2.0], name="mode_mnt")


def test_mode_solver_2D():
    """Run mode solver in 2D simulations."""
    mode_spec = td.ModeSpec(num_modes=3, filter_pol="te", precision="double", num_pml=(0, 10))
    simulation = td.Simulation(
        size=(0, 2, 2),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()

    mode_spec = td.ModeSpec(num_modes=3, filter_pol="te", precision="double", num_pml=(10, 0))
    simulation = td.Simulation(
        size=(2, 2, 0),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()


def test_compute_modes():
    """Test direct call to ``compute_modes`` (used in e.g. gdsfactory)."""
    eps_cross = np.random.rand(10, 10)
    coords = np.arange(11)
    mode_spec = td.ModeSpec(num_modes=3, target_neff=2.0)
    modes = compute_modes(
        eps_cross=[eps_cross] * 3,
        coords=[coords, coords],
        freq=td.constants.C_0 / 1.0,
        mode_spec=mode_spec,
    )


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
