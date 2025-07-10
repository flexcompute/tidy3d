"""Test class ``FreqRange`` for frequency and wavelength handling."""

from __future__ import annotations

import numpy as np

import tidy3d.constants as td_const
from tidy3d.components.source.freq_range import FreqRange
from tidy3d.components.source.time import GaussianPulse


def test_constructor():
    """
    test construction of ``FreqRange`` from central frequency
    ``freq0`` and bandwidth ``fwidth``.
    """
    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e8  # set one-side bandwidth

    # construct instance of FreqRange class
    freq_range = FreqRange(freq0=freq0, fwidth=fwidth)

    fmin = freq0 - fwidth
    fmax = freq0 + fwidth

    lmin = td_const.C_0 / fmax
    lmax = td_const.C_0 / fmin

    # validated if class atributes were initialized correctly
    assert freq_range.fmin == fmin
    assert freq_range.fmax == fmax
    assert freq_range.lda0 == (lmin + lmax) / 2
    assert freq_range.freq0 == freq0
    assert freq_range.fwidth == fwidth


def test_from_freq_interval():
    """
    test construction of ``FreqRange`` from frequency interval.
    """

    fmin = 1e10  # new min frequency
    fmax = 1e11  # new max frequency

    lmin = td_const.C_0 / fmax
    lmax = td_const.C_0 / fmin

    # update object given new frequencies
    freq_range = FreqRange.from_freq_interval(fmin=fmin, fmax=fmax)

    # validate if frequencies were updated correctly
    assert freq_range.fmin == fmin
    assert freq_range.fmax == fmax
    assert np.isclose(freq_range.freq0, (fmin + fmax) / 2, rtol=1e-14)
    assert np.isclose(freq_range.fwidth, (fmax - fmin) / 2, rtol=1e-14)
    assert np.isclose(freq_range.lda0, (lmin + lmax) / 2, rtol=1e-14)


def test_from_wavelength():
    """
    test construction of ``FreqRange`` from a central wavelength and a wavelength.
    """
    # set initial params
    wvl0 = 1
    wvl_width = 0.1

    # get the shortest and the longest wavelengths
    wvl_min = wvl0 - wvl_width
    wvl_max = wvl0 + wvl_width

    # define frequency range
    freq_range = FreqRange.from_wavelength(wvl0=wvl0, wvl_width=wvl_width)

    #
    assert freq_range.lda0 == wvl0

    fmin = td_const.C_0 / wvl_max
    fmax = td_const.C_0 / wvl_min

    assert np.isclose(freq_range.fmin, fmin, rtol=1e-14)
    assert np.isclose(freq_range.fmax, fmax, rtol=1e-14)
    assert np.isclose(freq_range.freq0, 0.5 * (fmin + fmax), rtol=1e-14)
    assert np.isclose(freq_range.fwidth, 0.5 * (fmax - fmin), rtol=1e-14)


def test_from_wvl_interval():
    """
    test construction of ``FreqRange`` from an interval of wavelengths.
    """
    # set initial params
    wvl_min = 0.1
    wvl_max = 1
    fmin = td_const.C_0 / wvl_max
    fmax = td_const.C_0 / wvl_min

    # freq_range = FreqRange(freq0, fwidth)

    # update frequencies based on wavelength
    freq_range = FreqRange.from_wvl_interval(wvl_min=wvl_min, wvl_max=wvl_max)

    # ensure that parameters are updated correctly
    assert np.isclose(freq_range.lda0, (wvl_min + wvl_max) / 2, rtol=1e-14)
    assert np.isclose(freq_range.freq0, 0.5 * (fmin + fmax), rtol=1e-14)
    assert np.isclose(freq_range.fmax, fmax, rtol=1e-14)
    assert np.isclose(freq_range.fmin, fmin, rtol=1e-14)
    assert np.isclose(freq_range.fwidth, 0.5 * (fmax - fmin), rtol=1e-14)


def test_freqs():
    """
    test generation of uniformly distributed frequency samples
    from a given frequency interval.
    """
    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e8  # set one-side bandwidth
    num_points = 11

    # construct instance of FreqRange class
    freq_range = FreqRange(freq0=freq0, fwidth=fwidth)

    # form sampling frequency points
    freqs = freq_range.freqs(num_points=num_points)

    # make sure
    assert np.array_equal(freqs, np.linspace(freq0 - fwidth, freq0 + fwidth, num_points))

    # reset number of sampling points to 1
    num_points = 1
    freqs = freq_range.freqs(num_points=num_points)

    # check if freqs == freq0
    assert np.array_equal(freqs, np.array([freq0]))


def test_ldas():
    """
    test generation of uniformly distributed wavelength samples
    from a given wavelength interval.
    """
    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e8  # set one-side bandwidth
    num_points = 11

    lmin = td_const.C_0 / (freq0 + fwidth)
    lmax = td_const.C_0 / (freq0 - fwidth)
    lda0 = (lmin + lmax) / 2

    # construct instance of FreqRange class
    freq_range = FreqRange(freq0=freq0, fwidth=fwidth)

    # form sampling frequency points
    ldas = freq_range.ldas(num_points=num_points)

    # make sure
    assert np.array_equal(ldas, np.linspace(lmin, lmax, num_points))

    # reset number of sampling points to 1
    num_points = 1
    ldas = freq_range.ldas(num_points=num_points)

    # check if freqs == freq0
    assert np.array_equal(ldas, np.array([lda0]))


def test_gaussian_pulse():
    """
    test generation of a ``GaussianPulse`` with frequency parameters
    defined in ``FreqRange``.
    """

    # set initial params
    freq0 = 1e9  # set central frequency
    fwidth = 1e8  # set one-side bandwidth
    fmin = freq0 - fwidth
    fmax = freq0 + fwidth

    # construct instance of FreqRange class
    freq_range = FreqRange(freq0=freq0, fwidth=fwidth)

    pulse_exp = GaussianPulse.from_frequency_range(
        fmin=fmin, fmax=fmax
    )  # instantiate GaussianPulse explicitly
    pulse_imp = freq_range.to_gaussian_pulse()  # get instance by calling a method gaussian_pulse()

    assert pulse_exp == pulse_imp  # compare two pulses
