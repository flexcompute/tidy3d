import numpy as np
import pytest
import responses

import matplotlib.pylab as plt

import tidy3d as td
from tidy3d.plugins.dispersion import DispersionFitter, FastDispersionFitter
from tidy3d.plugins.dispersion import AdvancedFastFitterParam
from tidy3d.plugins.dispersion.web import run as run_fitter

advanced_param = AdvancedFastFitterParam(num_iters=1, passivity_num_iters=1)


_, AX = plt.subplots()


@pytest.fixture
def random_data():
    data_points = 11
    wvl_um = np.linspace(1, 2, data_points)
    n_data = np.random.random(data_points)
    k_data = np.random.random(data_points)
    return wvl_um, n_data, k_data


@pytest.fixture
def mock_remote_api(monkeypatch):
    def mock_url(*args, **kwargs):
        return "http://monkeypatched.com"

    monkeypatch.setattr("tidy3d.plugins.dispersion.web.FitterData._set_url", mock_url)
    responses.add(responses.GET, f"{mock_url()}/health", status=200)
    responses.add(
        responses.POST,
        f"{mock_url()}/dispersion/fit",
        json={"message": td.PoleResidue().json(), "rms": 1e-16},
        status=200,
    )


def test_coeffs():
    """make sure pack_coeffs and unpack_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    a, c = DispersionFitter._unpack_coeffs(coeffs)
    coeffs_ = DispersionFitter._pack_coeffs(a, c)
    a_, c_ = DispersionFitter._unpack_coeffs(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(a, a_)
    assert np.allclose(c, c_)


def test_pole_coeffs():
    """make sure coeffs_to_poles and poles_to_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    poles = DispersionFitter._coeffs_to_poles(coeffs)
    coeffs_ = DispersionFitter._poles_to_coeffs(poles)
    poles_ = DispersionFitter._coeffs_to_poles(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(poles, poles_)


@responses.activate
def test_lossless_dispersion(random_data, mock_remote_api):
    """perform fitting on random data"""
    wvl_um, n_data, _ = random_data
    fitter = DispersionFitter(wvl_um=wvl_um.tolist(), n_data=tuple(n_data))
    medium, rms = fitter._fit_single()
    medium, rms = fitter.fit(num_tries=2)
    medium, rms = run_fitter(fitter)

    fitter = FastDispersionFitter(wvl_um=wvl_um.tolist(), n_data=tuple(n_data))
    medium, rms = fitter.fit(advanced_param=advanced_param)


@responses.activate
def test_lossy_dispersion(random_data, mock_remote_api):
    """perform fitting on random lossy data"""
    wvl_um, n_data, k_data = random_data
    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
    medium, rms = fitter._fit_single()
    medium, rms = fitter.fit(num_tries=2)
    medium, rms = run_fitter(fitter)

    fitter = FastDispersionFitter(wvl_um=wvl_um.tolist(), n_data=n_data, k_data=k_data)
    medium, rms = fitter.fit(advanced_param=advanced_param)


def test_dispersion_load():
    """loads dispersion model from nk data file"""
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)

    fitter = FastDispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(advanced_param=advanced_param)


def test_dispersion_plot(random_data):
    """plots a medium fit from file"""
    wvl_um, n_data, k_data = random_data

    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data)
    fitter.plot(ax=AX)
    medium, rms = fitter.fit(num_tries=2)
    fitter.plot(medium, ax=AX)

    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
    fitter.plot()
    medium, rms = fitter.fit(num_tries=2)
    fitter.plot(medium, ax=AX)


def test_dispersion_set_wvg_range(random_data):
    """set wavelength range function"""
    wvl_um, n_data, k_data = random_data
    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data)
    fastfitter = FastDispersionFitter(wvl_um=wvl_um, n_data=n_data)

    wvl_range = [1.2, 1.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range})
    assert len(fitter.freqs) == 7
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 7
    medium, rms = fastfitter.fit(advanced_param=advanced_param)

    wvl_range = [1.2, 2.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range, "k_data": k_data})
    assert len(fitter.freqs) == 9
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 9
    medium, rms = fastfitter.fit(advanced_param=advanced_param)

    wvl_range = [0.2, 1.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range})
    assert len(fitter.freqs) == 9
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 9
    medium, rms = fastfitter.fit(advanced_param=advanced_param)

    wvl_range = [0.2, 2.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range, "k_data": k_data})
    assert len(fitter.freqs) == 11
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 11
    medium, rms = fastfitter.fit(advanced_param=advanced_param)
