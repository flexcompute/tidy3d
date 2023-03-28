import numpy as np

from tidy3d.plugins.fitter import StableDispersionFitter, DispersionFitter


def test_dispersion_load_list():
    """performs a fit on some random data"""
    num_data = 10
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = StableDispersionFitter(wvl_um=wvls, n_data=n_data)


def test_dispersion_load_file():
    """loads dispersion model from nk data file"""
    fitter = StableDispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")


def test_dispersion_load_url():
    """performs a fit on some random data"""

    # both n and k
    url_csv = "https://refractiveindex.info/data_csv.php?datafile=data/main/Ag/Johnson.yml"
    url_txt = "https://refractiveindex.info/data_txt.php?datafile=data/main/Ag/Johnson.yml"
    fitter = DispersionFitter.from_url(url_csv, delimiter=",")
    fitter = StableDispersionFitter.from_url(url_csv, delimiter=",")
    fitter_txt = DispersionFitter.from_url(url_txt, delimiter="\t")
    fitter_txt = StableDispersionFitter.from_url(url_txt, delimiter="\t")
    fitter_txt.wvl_range = [0.3, 0.8]
    assert len(fitter_txt.freqs) < len(fitter.freqs)

    # only k
    url_csv = "https://refractiveindex.info/data_csv.php?datafile=data/main/N2/Peck-0C.yml"
    url_txt = "https://refractiveindex.info/data_txt.php?datafile=data/main/N2/Peck-0C.yml"
    fitter = DispersionFitter.from_url(url_csv, delimiter=",")
    fitter_txt = DispersionFitter.from_url(url_txt, delimiter="\t")
