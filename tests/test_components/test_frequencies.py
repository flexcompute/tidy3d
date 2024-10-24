import numpy as np
import pytest
import tidy3d as td


def test_classification():
    assert td.frequencies.classification(1) == ("near static",)
    assert td.wavelengths.classification(td.C_0) == ("near static",)
    assert td.frequencies.classification(td.C_0 / 1.55) == ("infrared", "NIR")
    assert td.wavelengths.classification(1.55) == ("infrared", "NIR")


@pytest.mark.parametrize("band", ["O", "E", "S", "C", "L", "U"])
def test_bands(band):
    freqs = getattr(td.frequencies, band.lower() + "_band")()
    ldas = getattr(td.wavelengths, band.lower() + "_band")()
    assert np.allclose(freqs, td.C_0 / np.array(ldas))
