import numpy as np
import pytest
import responses
import tidy3d as td
from tidy3d import DispersionFitter
from tidy3d.plugins.dispersion.web import run as run_fitter


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


@responses.activate
def test_run_fit(mock_remote_api):
    """perform fitting on random data"""

    data_points = 11
    wvl_um = np.linspace(1, 2, data_points)
    n_data = np.random.random(data_points)
    k_data = np.random.random(data_points)

    # lossless
    fitter = DispersionFitter(wvl_um=wvl_um.tolist(), n_data=tuple(n_data))
    medium, rms = run_fitter(fitter)

    # lossy
    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
    medium, rms = run_fitter(fitter)
