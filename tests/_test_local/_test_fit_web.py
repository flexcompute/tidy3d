import numpy as np
from math import isclose

from tidy3d.plugins.fitter import StableDispersionFitter, AdvancedFitterParam

np.random.seed(4)
ATOL = 1e-50


def test_dispersion_load_list():
    """performs a fit on some random data"""
    num_data = 10
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = StableDispersionFitter(wvl_um=wvls, n_data=n_data)

    num_poles = 3
    num_tries = 10
    tolerance_rms = 1e-3
    best_medium, best_rms = fitter.fit(
        num_tries=num_tries, num_poles=num_poles, tolerance_rms=tolerance_rms
    )
    print(best_rms)
    print(best_medium.eps_model(1e12))


def test_dispersion_lossless():
    """lossless fitting when k_data is not supplied"""
    num_data = 10
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = StableDispersionFitter(wvl_um=wvls, n_data=n_data)

    num_poles = 3
    num_tries = 10
    tolerance_rms = 1e-3
    best_medium, best_rms = fitter.fit(
        num_tries=num_tries, num_poles=num_poles, tolerance_rms=tolerance_rms
    )

    # a and c for each pole should be purely imaginary
    for (a, c) in best_medium.poles:
        assert isclose(np.real(a), 0)
        assert isclose(np.real(c), 0)

    # Im[ep] = 0 at any frequency
    freq = np.random.random(100) * 1e14
    ep = best_medium.eps_model(freq)
    assert np.allclose(ep.imag, 0, atol=ATOL)


def test_dispersion_load_file():
    """loads dispersion model from nk data file"""
    fitter = StableDispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")

    num_poles = 3
    num_tries = 10
    tolerance_rms = 1e-3
    best_medium, best_rms = fitter.fit(
        num_tries=num_tries, num_poles=num_poles, tolerance_rms=tolerance_rms
    )
    print(best_rms)
    print(best_medium.eps_model(1e12))


def test_dispersion_load_url():

    url_csv = "https://refractiveindex.info/data_csv.php?datafile=data/main/Ag/Johnson.yml"
    fitter = StableDispersionFitter.from_url(url_csv)

    num_poles = 2
    num_tries = 10
    tolerance_rms = 1e-3
    best_medium, best_rms = fitter.fit(
        num_tries=num_tries,
        num_poles=num_poles,
        tolerance_rms=tolerance_rms,
        advanced_param=AdvancedFitterParam(constraint="hard", bound_eps_inf=10),
    )
    print(best_rms)
    print(best_medium.eps_inf)

    fitter.wvl_range = (1.0, 1.3)
    print(len(fitter.freqs))
    best_medium, best_rms = fitter.fit(
        num_tries=num_tries,
        num_poles=num_poles,
        tolerance_rms=tolerance_rms,
        advanced_param=AdvancedFitterParam(constraint="hard", bound_eps_inf=10),
    )
    print(best_rms)
    print(best_medium.eps_inf)
