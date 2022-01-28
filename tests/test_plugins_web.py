import numpy as np

from tidy3d.plugins import StableDispersionFitter


def test_dispersion_load_list():
    """performs a fit on some random data"""
    num_data = 10
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = StableDispersionFitter(wvls, n_data)

    # num_poles = 3
    # num_tries = 50
    # tolerance_rms = 1e-3
    # local_run = True
    # best_medium, best_rms = fitter.fit(
    #     num_tries=num_tries, num_poles=num_poles, tolerance_rms=tolerance_rms, local_run=True
    # )
    # print(best_rms)
    # print(best_medium.eps_model(1e12))


def test_dispersion_load_file():
    """loads dispersion model from nk data file"""
    fitter = StableDispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")

    # num_poles = 3
    # num_tries = 50
    # tolerance_rms = 1e-3
    # local_run = True
    # best_medium, best_rms = fitter.fit(
    #     num_tries=num_tries, num_poles=num_poles, tolerance_rms=tolerance_rms, local_run=True
    # )
    # print(best_rms)
    # print(best_medium.eps_model(1e12))
