"""Test frequenies and wavelength utilities and convenience class ``FreqRange`` for frequency and wavelength handling."""

from __future__ import annotations

import numpy as np
import pytest

import tidy3d as td
from tidy3d.components.source.time import GaussianPulse


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


def test_FreqRange_constructor():
    """
    test construction of ``FreqRange`` from central frequency
    ``freq0`` and bandwidth ``fwidth``.
    """
    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e9  # set one-side bandwidth

    # construct instance of FreqRange class
    freq_range = td.FreqRange(freq0=freq0, fwidth=fwidth)

    fmin = freq0 - fwidth / 2.0
    fmax = freq0 + fwidth / 2.0

    lmin = td.C_0 / fmax
    lmax = td.C_0 / fmin

    # validated if class atributes were initialized correctly
    assert freq_range.fmin == fmin
    assert freq_range.fmax == fmax
    assert freq_range.wvl0 == (lmin + lmax) / 2
    assert freq_range.freq0 == freq0
    assert freq_range.fwidth == fwidth


def test_invalid_freq_range():
    with pytest.raises(ValueError, match=r"fwidth.*less than.*freq0"):
        td.FreqRange(freq0=1e9, fwidth=2e9)


def test_from_freq_interval():
    """
    test construction of ``FreqRange`` from frequency interval.
    """

    fmin = 1e10  # new min frequency
    fmax = 1e11  # new max frequency

    lmin = td.C_0 / fmax
    lmax = td.C_0 / fmin

    # update object given new frequencies
    freq_range = td.FreqRange.from_freq_interval(fmin=fmin, fmax=fmax)

    # validate if frequencies were updated correctly
    assert freq_range.fmin == fmin
    assert freq_range.fmax == fmax
    assert np.isclose(freq_range.freq0, (fmin + fmax) / 2, rtol=1e-14)
    assert np.isclose(freq_range.fwidth, (fmax - fmin), rtol=1e-14)
    assert np.isclose(freq_range.wvl0, (lmin + lmax) / 2, rtol=1e-14)


def test_from_wvl():
    """
    test construction of ``FreqRange`` from a central wavelength and a wavelength.
    """
    # set initial params
    wvl0 = 1
    wvl_width = 0.1

    # get the shortest and the longest wavelengths
    wvl_min = wvl0 - wvl_width / 2.0
    wvl_max = wvl0 + wvl_width / 2.0

    # define frequency range
    freq_range = td.FreqRange.from_wvl(wvl0=wvl0, wvl_width=wvl_width)

    #
    assert freq_range.wvl0 == wvl0

    fmin = td.C_0 / wvl_max
    fmax = td.C_0 / wvl_min

    assert np.isclose(freq_range.fmin, fmin, rtol=1e-14)
    assert np.isclose(freq_range.fmax, fmax, rtol=1e-14)
    assert np.isclose(freq_range.freq0, 0.5 * (fmin + fmax), rtol=1e-14)
    assert np.isclose(freq_range.fwidth, fmax - fmin, rtol=1e-14)


def test_from_wvl_interval():
    """
    test construction of ``FreqRange`` from an interval of wavelengths.
    """
    # set initial params
    wvl_min = 0.1
    wvl_max = 1
    fmin = td.C_0 / wvl_max
    fmax = td.C_0 / wvl_min

    # freq_range = FreqRange(freq0, fwidth)

    # update frequencies based on wavelength
    freq_range = td.FreqRange.from_wvl_interval(wvl_min=wvl_min, wvl_max=wvl_max)

    # ensure that parameters are updated correctly
    assert np.isclose(freq_range.wvl0, (wvl_min + wvl_max) / 2, rtol=1e-14)
    assert np.isclose(freq_range.freq0, 0.5 * (fmin + fmax), rtol=1e-14)
    assert np.isclose(freq_range.fmax, fmax, rtol=1e-14)
    assert np.isclose(freq_range.fmin, fmin, rtol=1e-14)
    assert np.isclose(freq_range.fwidth, fmax - fmin, rtol=1e-14)


def test_freqs():
    """
    test generation of uniformly distributed frequency samples
    from a given frequency interval.
    """
    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e8  # set one-side bandwidth

    fmin = freq0 - fwidth / 2.0
    fmax = freq0 + fwidth / 2.0
    lmin = td.C_0 / fmax
    lmax = td.C_0 / fmin
    num_points = 11

    # construct instance of FreqRange class
    freq_range = td.FreqRange(freq0=freq0, fwidth=fwidth)

    # form sampling frequency points
    freqs = freq_range.freqs(num_points=num_points)

    # make sure
    assert np.allclose(freqs, np.linspace(fmin, fmax, num_points))

    # test ``uniform_wvl`` mode
    wvls = np.linspace(lmin, lmax, num_points)
    freqs = freq_range.freqs(num_points=num_points, spacing="uniform_wvl")
    assert np.allclose(freqs, np.sort(td.C_0 / wvls))

    # reset number of sampling points to 1
    num_points = 1
    freqs = freq_range.freqs(num_points=num_points, spacing="uniform_freq")

    # check if freqs == freq0
    assert np.allclose(freqs, np.array([freq0]))

    wvls = np.array((lmin + lmax) / 2)

    freqs = freq_range.freqs(num_points=num_points, spacing="uniform_wvl")
    assert np.allclose(freqs, td.C_0 / wvls)

    with pytest.raises(ValueError) as excinfo:
        freq_range.freqs(num_points=num_points, spacing="unifomr_wvl")
    assert "invalid `spacing` value" in str(excinfo.value).lower()


def test_wvls():
    """
    test generation of uniformly distributed wavelength samples
    from a given wavelength interval.
    """
    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e8  # set one-side bandwidth
    num_points = 11

    fmin = freq0 - fwidth / 2.0
    fmax = freq0 + fwidth / 2.0

    lmin = td.C_0 / fmax
    lmax = td.C_0 / fmin
    wvl0 = (lmin + lmax) / 2

    # construct instance of FreqRange class
    freq_range = td.FreqRange(freq0=freq0, fwidth=fwidth)

    # form sampling frequency points
    wvls = freq_range.wvls(num_points=num_points)

    # make sure
    assert np.allclose(wvls, np.linspace(lmin, lmax, num_points))

    # test ``uniform_freq``
    wvls = freq_range.wvls(num_points=num_points, spacing="uniform_freq")
    assert np.allclose(wvls, td.C_0 / np.linspace(fmax, fmin, num_points))

    # reset number of sampling points to 1
    num_points = 1
    wvls = freq_range.wvls(num_points=num_points, spacing="uniform_wvl")
    # check if freqs == freq0
    assert np.allclose(wvls, np.array([wvl0]))

    # test ``uniform_freq`` mode
    wvls = freq_range.wvls(num_points=num_points, spacing="uniform_freq")
    assert np.allclose([td.C_0 / freq0], wvls)

    with pytest.raises(ValueError) as excinfo:
        freq_range.wvls(num_points=num_points, spacing="unifomr_freq")
    assert "invalid `spacing` value" in str(excinfo.value).lower()


def test_gaussian_pulse():
    """
    test generation of a ``GaussianPulse`` with frequency parameters
    defined in ``FreqRange``.
    """

    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e8  # set one-side bandwidth
    fmin = freq0 - fwidth / 2.0
    fmax = freq0 + fwidth / 2.0

    # construct instance of FreqRange class
    freq_range = td.FreqRange(freq0=freq0, fwidth=fwidth)

    pulse_exp = GaussianPulse.from_frequency_range(
        fmin=fmin, fmax=fmax
    )  # instantiate GaussianPulse explicitly
    pulse_imp = freq_range.to_gaussian_pulse()  # get instance by calling a method gaussian_pulse()

    assert pulse_exp == pulse_imp  # compare two pulses

    with pytest.raises(ValueError) as excinfo:
        pulse_new = freq_range.to_gaussian_pulse(fmin=1234)
    assert (
        "keyword argument 'fmin'" in str(excinfo.value).lower()
        or "keys 'fmin'" in str(excinfo.value).lower()
    )

    # Case 2: multiple duplicate keys 'fmin' and 'fmax'
    with pytest.raises(ValueError) as excinfo:
        freq_range.to_gaussian_pulse(fmin=1234, fmax=5678)
    # Check plural message with both keys mentioned
    msg = str(excinfo.value)
    assert "'fmax'" in msg and "'fmin'" in msg
    assert "conflict" in msg.lower()
    assert "exclude" in msg.lower()


def test_sweep_decade():
    """
    Test the sweep_decade method for generating logarithmically spaced frequencies.
    """
    # Test basic functionality
    freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)  # 1 kHz to 1 MHz (3 decades)
    freqs = freq_range.sweep_decade(10)  # 10 points per decade

    # Check that the frequencies have the correct bounds and are logarithmically spaced
    assert len(freqs) > 0
    ratios = freqs[1:] / freqs[:-1]
    assert np.allclose(ratios, ratios[0])
    assert np.isclose(freqs[0], freq_range.fmin)
    assert np.isclose(freqs[-1], freq_range.fmax)

    # Test with different number of points per decade
    freqs_9 = freq_range.sweep_decade(9)
    freqs_11 = freq_range.sweep_decade(11)

    # More points per decade should result in more total points
    assert len(freqs_9) < len(freqs) < len(freqs_11)

    # All should span the same range
    assert np.isclose(freqs_9[0], freqs_11[0])
    assert np.isclose(freqs_9[-1], freqs_11[-1])
    assert np.isclose(freqs[0], freqs_11[0])
    assert np.isclose(freqs[-1], freqs_11[-1])


def test_sweep_decade_edge_cases():
    """
    Test edge cases and error conditions for sweep_decade.
    """
    # Test with single decade
    freq_range = td.FreqRange.from_freq_interval(1e3, 1e4)  # 1 decade
    freqs = freq_range.sweep_decade(5)

    # Check that the frequencies have the correct bounds and are logarithmically spaced
    assert len(freqs) >= 5  # Should have at least 5 points
    ratios = freqs[1:] / freqs[:-1]
    assert np.allclose(ratios, ratios[0])
    assert np.isclose(freqs[0], freq_range.fmin)
    assert np.isclose(freqs[-1], freq_range.fmax)

    # Test error conditions
    freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)

    # Negative points per decade
    with pytest.raises(
        ValueError, match="'num_points_per_decade' must be strictly positive, got -1."
    ):
        freq_range.sweep_decade(-1)

    # Zero points per decade
    with pytest.raises(
        ValueError, match="'num_points_per_decade' must be strictly positive, got 0."
    ):
        freq_range.sweep_decade(0)
