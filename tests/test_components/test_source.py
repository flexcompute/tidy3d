"""Tests sources."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.source.field import CHEB_GRID_WIDTH, DirectionalSource
from tidy3d.exceptions import SetupError

from ..utils import AssertLogLevel, AssertLogStr, assert_single_value_error_loc

ST = td.GaussianPulse(freq0=2e14, fwidth=1e14)
S = td.PointDipole(source_time=ST, polarization="Ex")
FWIDTH_FRAC = 0.1


ATOL = 1e-8


def test_plot_source_time():
    for val in ("real", "imag", "abs"):
        ST.plot(times=[1e-15, 2e-15, 3e-15], val=val)
        ST.plot_spectrum(times=[1e-15, 2e-15, 3e-15], num_freqs=4, val=val)

    ST_DC = ST.updated_copy(remove_dc_component=False)
    for val in ("real", "imag", "abs"):
        ST_DC.plot(times=[1e-15, 2e-15, 3e-15], val=val)
        ST_DC.plot_spectrum(times=[1e-15, 2e-15, 3e-15], num_freqs=4, val=val)

    with pytest.raises(ValueError):
        ST.plot(times=[1e-15, 2e-15, 3e-15], val="blah")

    with pytest.raises(ValueError):
        ST.plot_spectrum(times=[1e-15, 2e-15, 3e-15], num_freqs=4, val="blah")

    # uneven spacing in times
    with pytest.raises(SetupError):
        ST.plot_spectrum(times=[1e-15, 3e-15, 4e-15], num_freqs=4)

    plt.close("all")


def test_dir_vector():
    MS = td.ModeSource(size=(1, 0, 1), mode_spec=td.ModeSpec(), source_time=ST, direction="+")
    DirectionalSource._dir_vector.fget(MS)
    assert DirectionalSource._dir_vector.fget(S) is None


def test_mode_bend_radius():
    """Test that small bend radius fails."""

    with pytest.raises(ValueError):
        src = td.ModeSource(
            size=(1, 0, 5),
            source_time=ST,
            mode_spec=td.ModeSpec(num_modes=1, bend_radius=1, bend_axis=0),
        )
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            sources=[src],
        )


def test_UniformCurrentSource():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    # test we can make generic UniformCurrentSource
    _ = td.UniformCurrentSource(
        size=(1, 1, 1),
        source_time=g,
        polarization="Ez",
        interpolate=False,
        current_amplitude_definition="total",
    )
    _ = td.UniformCurrentSource(
        size=(1, 1, 1),
        source_time=g,
        polarization="Ez",
        interpolate=True,
        current_amplitude_definition="total",
    )


def test_UniformCurrentSource_amplitude_definition_warning():
    """Test deprecation warning when current_amplitude_definition is not explicitly set."""
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    # Not set: should warn
    with AssertLogLevel("WARNING", contains_str="current_amplitude_definition"):
        _ = td.UniformCurrentSource(size=(1, 1, 1), source_time=g, polarization="Ez")

    # Explicitly "density": no warning
    with AssertLogLevel(None):
        _ = td.UniformCurrentSource(
            size=(1, 1, 1),
            source_time=g,
            polarization="Ez",
            current_amplitude_definition="density",
        )

    # Explicitly "total": no warning
    with AssertLogLevel(None):
        _ = td.UniformCurrentSource(
            size=(1, 1, 1),
            source_time=g,
            polarization="Ez",
            current_amplitude_definition="total",
        )


def test_source_times():
    # test we can make gaussian pulse
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)
    ts = np.linspace(0, 30, 1001) * 1e-12
    g.amp_time(ts)
    # g.plot(ts)
    # plt.close()

    # test we can make cw pulse
    from tidy3d.components.source.time import ContinuousWave

    c = ContinuousWave(freq0=1e12, fwidth=0.1e12)
    c.amp_time(ts)

    # test gaussian pulse with and without DC component
    g = td.GaussianPulse(freq0=0.1e12, fwidth=1e12)
    dc_comp = g.spectrum(ts, [0], ts[1] - ts[0])
    assert abs(dc_comp) ** 2 < 1e-32
    g = td.GaussianPulse(freq0=0.1e12, fwidth=1e12, remove_dc_component=False)
    dc_comp = g.spectrum(ts, [0], ts[1] - ts[0])
    assert abs(dc_comp) ** 2 > 1e-32


def test_gaussian_from_frequency_range():
    # error with negative fmin
    with pytest.raises(ValueError):
        _ = td.GaussianPulse.from_frequency_range(fmin=-1e10, fmax=1e10)
    # error with fmin = 0
    with pytest.raises(ValueError):
        _ = td.GaussianPulse.from_frequency_range(fmin=0, fmax=1e10)
    # error with fmin >= fmax
    with pytest.raises(ValueError):
        _ = td.GaussianPulse.from_frequency_range(fmin=1e10, fmax=0.9e10)
    # error with minimum_source_bandwidth negative or 0
    with pytest.raises(ValueError):
        _ = td.GaussianPulse.from_frequency_range(
            fmin=9e10, fmax=11e10, minimum_source_bandwidth=0.0
        )
        _ = td.GaussianPulse.from_frequency_range(
            fmin=9e10, fmax=11e10, minimum_source_bandwidth=-1.0
        )
        # error with minimum_source_bandwidth greater than 1
        _ = td.GaussianPulse.from_frequency_range(
            fmin=9e10, fmax=11e10, minimum_source_bandwidth=1.0
        )

    fmin = 1e9
    fmax = 20e9
    # dc component on
    g = td.GaussianPulse.from_frequency_range(fmin=fmin, fmax=fmax, remove_dc_component=False)
    assert g.freq0 == 0.5 * (fmin + fmax)
    assert g.fwidth == 0.5 * (fmax - fmin)

    # dc component removed
    g1 = td.GaussianPulse.from_frequency_range(fmin=fmin, fmax=fmax, remove_dc_component=True)
    # default to dc removed
    g2 = td.GaussianPulse.from_frequency_range(fmin=fmin, fmax=fmax)
    assert g2.remove_dc_component

    with AssertLogLevel("WARNING", contains_str="broadband"):
        g_small = td.GaussianPulse.from_frequency_range(
            fmin=fmin, fmax=60e9, remove_dc_component=True
        )

    # 1) broadband: assert enough amplitude at fmin and fmax
    time = np.linspace(0, 5 / fmin, 10001)
    freqs = np.linspace(fmin, fmax, 101)
    spectrum = np.abs(g2.spectrum(time, freqs, time[1] - time[0]))
    max_amp = np.max(spectrum)
    assert spectrum[0] / max_amp > 0.1
    assert spectrum[-1] / max_amp > 0.1

    # 2) narrow band: close to regular Gaussian result
    fmin = 10e9
    bandwidth = 1e6
    fmax = fmin + 2 * bandwidth
    g = td.GaussianPulse.from_frequency_range(fmin=fmin, fmax=fmax)
    assert abs(g.fwidth - bandwidth) / bandwidth < 1e-4
    assert abs(g.freq0 - fmin) / fmin < 1e-4


def test_gaussian_frequency_sigma_range():
    sigma = 4
    # derivative Gaussian
    g = td.GaussianPulse.from_frequency_range(fmin=1e9, fmax=10e9, remove_dc_component=True)
    f_range = g.frequency_range_sigma(sigma=sigma)
    peak_amp = np.abs(g.amp_freq(g.peak_frequency))
    amp = np.array([np.abs(g.amp_freq(f)) for f in f_range])
    assert f_range[1] > f_range[0]
    assert np.allclose(amp / peak_amp, np.exp(-(sigma**2) / 2))
    # freq0 from frequency_range_sigma is larger than freq0
    assert g._freq0_sigma_centroid > g._freq0

    # pure Gaussian
    g = td.GaussianPulse.from_frequency_range(fmin=1e9, fmax=10e9, remove_dc_component=False)
    assert np.allclose(g.frequency_range(num_fwidth=sigma), g.frequency_range_sigma(sigma=sigma))


def test_frequency_source_width():
    """Ensure the source bandwidth has a lower bound regardless of the input frequencies."""

    middle_freq = 1e10
    fmin_large_bw = middle_freq * (1 - 2 * FWIDTH_FRAC)
    fmax_large_bw = middle_freq * (1 + 2 * FWIDTH_FRAC)
    fmin, fmax = td.GaussianPulse._minimum_source_bandwidth(
        fmin=fmin_large_bw, fmax=fmax_large_bw, minimum_source_bandwidth=FWIDTH_FRAC
    )

    assert np.isclose(fmin, fmin_large_bw), (
        "Expected no change in bandwidth (fmin unexpectedly changed)."
    )

    assert np.isclose(fmax, fmax_large_bw), (
        "Expected no change in bandwidth (fmax unexpectedly changed)."
    )

    fmin_small_bw = middle_freq * (1 - 0.1 * FWIDTH_FRAC)
    fmax_small_bw = middle_freq * (1 + 0.1 * FWIDTH_FRAC)
    fmin, fmax = td.GaussianPulse._minimum_source_bandwidth(
        fmin=fmin_small_bw, fmax=fmax_small_bw, minimum_source_bandwidth=FWIDTH_FRAC
    )
    fmin_expected = middle_freq * (1 - 0.5 * FWIDTH_FRAC)
    fmax_expected = middle_freq * (1 + 0.5 * FWIDTH_FRAC)

    assert np.isclose(fmin, fmin_expected), (
        "Expected increase in bandwidth (fmin unexpected value)."
    )

    assert np.isclose(fmax, fmax_expected), (
        "Expected increase in bandwidth (fmax unexpected value)."
    )


def test_dipole():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)
    _ = td.PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex", interpolate=True)
    _ = td.PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex", interpolate=False)
    # p.plot(y=2)
    # plt.close()

    with pytest.raises(ValidationError):
        _ = td.PointDipole(size=(1, 1, 1), source_time=g, center=(1, 2, 3), polarization="Ex")


def test_dipole_sources_from_angles():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    with pytest.raises(ValidationError):
        _ = td.PointDipole.sources_from_angles(
            size=(1, 1, 1),
            source_time=g,
            center=(1, 2, 3),
            angle_theta=np.pi / 4,
            angle_phi=np.pi / 4,
        )

    with pytest.raises(ValueError):
        _ = td.PointDipole.sources_from_angles(
            source_time=g,
            angle_theta=np.pi / 4,
            angle_phi=np.pi / 4,
            component="invalid",
            center=(1, 2, 3),
        )

    assert (
        len(
            td.PointDipole.sources_from_angles(
                source_time=g,
                angle_theta=np.pi / 4,
                angle_phi=np.pi / 4,
                component="electric",
                center=(1, 2, 3),
            )
        )
        == 3
    )

    assert (
        len(
            td.PointDipole.sources_from_angles(
                source_time=g,
                angle_theta=np.pi / 4,
                angle_phi=np.pi / 2,
                component="electric",
                center=(1, 2, 3),
            )
        )
        == 2
    )

    assert (
        len(
            td.PointDipole.sources_from_angles(
                source_time=g,
                angle_theta=np.pi / 2,
                angle_phi=np.pi / 2,
                component="electric",
                center=(1, 2, 3),
            )
        )
        == 1
    )


def test_FieldSource():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)
    mode_spec = td.ModeSpec(num_modes=2)

    # test we can make planewave
    _ = td.PlaneWave(size=(0, td.inf, td.inf), source_time=g, pol_angle=np.pi / 2, direction="+")
    # s.plot(y=0)
    # plt.close()

    # test we can make gaussian beam
    _ = td.GaussianBeam(size=(0, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    # s.plot(y=0)
    # plt.close()

    # test we can make an astigmatic gaussian beam
    _ = td.AstigmaticGaussianBeam(
        size=(0, 1, 1),
        source_time=g,
        pol_angle=np.pi / 2,
        direction="+",
        waist_sizes=(0.2, 0.4),
        waist_distances=(0.1, 0.3),
    )

    # test we can make mode source
    _ = td.ModeSource(
        size=(0, 1, 1), direction="+", source_time=g, mode_spec=mode_spec, mode_index=0
    )
    # s.plot(y=0)
    # plt.close()

    # test that non-planar geometry crashes plane wave and gaussian beams
    with pytest.raises(ValidationError):
        _ = td.PlaneWave(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(ValidationError):
        _ = td.GaussianBeam(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(ValidationError):
        _ = td.AstigmaticGaussianBeam(
            size=(1, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
        )
    with pytest.raises(ValidationError):
        _ = td.ModeSource(size=(1, 1, 1), source_time=g, mode_spec=mode_spec)

    tfsf = td.TFSF(size=(1, 1, 1), direction="+", source_time=g, injection_axis=2)
    _ = tfsf.injection_plane_center

    # assert that TFSF must be volumetric
    with pytest.raises(ValidationError):
        _ = td.TFSF(size=(1, 1, 0), direction="+", source_time=g, injection_axis=2)

    # s.plot(z=0)
    # plt.close()


def test_gaussian_beam_backward_waist_distance_warning():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    with AssertLogLevel(
        "WARNING",
        contains_str="GaussianBeam with direction '-' and non-zero 'waist_distance'",
    ):
        td.GaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="-",
            waist_distance=1.0,
        )

    with AssertLogLevel(None):
        td.GaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="-",
            waist_distance=0.0,
        )


def test_astigmatic_gaussian_beam_backward_waist_distance_warning():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    with AssertLogLevel(
        "WARNING",
        contains_str="AstigmaticGaussianBeam with direction '-' and non-zero 'waist_distances'",
    ):
        td.AstigmaticGaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="-",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.0),
        )

    with AssertLogLevel(None):
        td.AstigmaticGaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="-",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.0, 0.0),
        )


def test_wrapped_glancing_incidence_warning():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    with AssertLogLevel("WARNING", contains_str="close to glancing angle"):
        td.PlaneWave(
            size=(0, td.inf, td.inf),
            source_time=g,
            direction="+",
            angle_theta=5 * np.pi / 2,
        )


def test_pol_arrow():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    def get_pol_dir(axis, pol_angle=0, angle_theta=0, angle_phi=0):
        size = [td.inf, td.inf, td.inf]
        size[axis] = 0

        pw = td.PlaneWave(
            size=size,
            source_time=g,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            direction="+",
        )

        return pw._pol_vector

    assert np.allclose(get_pol_dir(axis=0), (0, 1, 0))
    assert np.allclose(get_pol_dir(axis=1), (1, 0, 0))
    assert np.allclose(get_pol_dir(axis=2), (1, 0, 0))
    assert np.allclose(get_pol_dir(axis=0, angle_phi=np.pi / 2), (0, 0, 1))
    assert np.allclose(get_pol_dir(axis=1, angle_phi=np.pi / 2), (0, 0, 1))
    assert np.allclose(get_pol_dir(axis=2, angle_phi=np.pi / 2), (0, 1, 0))
    assert np.allclose(get_pol_dir(axis=0, pol_angle=np.pi / 2), (0, 0, 1))
    assert np.allclose(get_pol_dir(axis=1, pol_angle=np.pi / 2), (0, 0, 1))
    assert np.allclose(get_pol_dir(axis=2, pol_angle=np.pi / 2), (0, 1, 0))
    assert np.allclose(
        get_pol_dir(axis=0, angle_theta=np.pi / 4), (-1 / np.sqrt(2), +1 / np.sqrt(2), 0)
    )
    assert np.allclose(
        get_pol_dir(axis=1, angle_theta=np.pi / 4), (+1 / np.sqrt(2), -1 / np.sqrt(2), 0)
    )
    assert np.allclose(
        get_pol_dir(axis=2, angle_theta=np.pi / 4), (+1 / np.sqrt(2), 0, -1 / np.sqrt(2))
    )


def test_broadband_source():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)
    mode_spec = td.ModeSpec(num_modes=2)
    fmin, fmax = g.frequency_range_sigma(sigma=CHEB_GRID_WIDTH)
    fdiff = (fmax - fmin) / 2
    fmean = (fmax + fmin) / 2

    def check_freq_grid(freq_grid, num_freqs):
        """Test that chebyshev polynomials are orthogonal on provided grid."""
        cheb_grid = (freq_grid - fmean) / fdiff
        poly = np.polynomial.chebyshev.chebval(cheb_grid, np.ones(num_freqs))
        dot_prod_theory = num_freqs + num_freqs * (num_freqs - 1) / 2
        # print(len(freq_grid), num_freqs)
        # print(abs(dot_prod_theory - np.dot(poly, poly)))
        assert len(freq_grid) == num_freqs
        assert abs(dot_prod_theory - np.dot(poly, poly)) < 1e-10

    # test we can make a broadband gaussian beam
    num_freqs = 3
    s = td.GaussianBeam(
        size=(0, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+", num_freqs=num_freqs
    )
    freq_grid = s.frequency_grid
    check_freq_grid(freq_grid, num_freqs)

    # test we can make a broadband astigmatic gaussian beam
    num_freqs = 10
    s = td.AstigmaticGaussianBeam(
        size=(0, 1, 1),
        source_time=g,
        pol_angle=np.pi / 2,
        direction="+",
        waist_sizes=(0.2, 0.4),
        waist_distances=(0.1, 0.3),
        num_freqs=num_freqs,
    )
    freq_grid = s.frequency_grid
    check_freq_grid(freq_grid, num_freqs)

    # test we can make a broadband mode source
    num_freqs = 20
    s = td.ModeSource(
        size=(0, 1, 1),
        direction="+",
        source_time=g,
        mode_spec=mode_spec,
        mode_index=0,
        num_freqs=num_freqs,
    )
    freq_grid = s.frequency_grid
    check_freq_grid(freq_grid, num_freqs)

    # check validators for num_freqs
    with pytest.raises(ValidationError):
        s = td.GaussianBeam(
            size=(0, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+", num_freqs=200
        )
    with pytest.raises(ValidationError):
        s = td.AstigmaticGaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
            num_freqs=100,
        )
    with pytest.raises(ValidationError):
        s = td.ModeSource(
            size=(0, 1, 1),
            direction="+",
            source_time=g,
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=-10,
        )


def test_pole_residue_broadband_source():
    """Test pole_residue broadband method on various source types."""
    g = td.GaussianPulse(freq0=2e14, fwidth=1e14)
    mode_spec = td.ModeSpec(num_modes=2)

    # ModeSource with pole_residue returns num_freqs uniformly-spaced points
    num_freqs = 6
    s_pr = td.ModeSource(
        size=(0, 1, 1),
        direction="+",
        source_time=g,
        mode_spec=mode_spec,
        mode_index=0,
        num_freqs=num_freqs,
        broadband_method="pole_residue",
    )
    freq_grid_pr = s_pr.frequency_grid
    assert len(freq_grid_pr) == num_freqs
    diffs = np.diff(freq_grid_pr)
    assert np.allclose(diffs, diffs[0], rtol=1e-12), "pole_residue grid must be uniform"

    # GaussianBeam with pole_residue
    s_gb = td.GaussianBeam(
        size=(0, 1, 1),
        source_time=g,
        pol_angle=np.pi / 2,
        direction="+",
        num_freqs=5,
        broadband_method="pole_residue",
    )
    assert len(s_gb.frequency_grid) == 5

    # AstigmaticGaussianBeam with pole_residue
    s_agb = td.AstigmaticGaussianBeam(
        size=(0, 1, 1),
        source_time=g,
        pol_angle=np.pi / 2,
        direction="+",
        waist_sizes=(0.2, 0.4),
        waist_distances=(0.1, 0.3),
        num_freqs=8,
        broadband_method="pole_residue",
    )
    assert len(s_agb.frequency_grid) == 8

    # pole_residue allows num_freqs up to 50
    s_pr_high = td.ModeSource(
        size=(0, 1, 1),
        direction="+",
        source_time=g,
        mode_spec=mode_spec,
        mode_index=0,
        num_freqs=50,
        broadband_method="pole_residue",
    )
    assert len(s_pr_high.frequency_grid) == 50

    # Chebyshev rejects num_freqs > 20
    with pytest.raises(ValidationError):
        td.ModeSource(
            size=(0, 1, 1),
            direction="+",
            source_time=g,
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=25,
            broadband_method="chebyshev",
        )

    # pole_residue rejects num_freqs > 50 (pydantic le=50)
    with pytest.raises(ValidationError):
        td.ModeSource(
            size=(0, 1, 1),
            direction="+",
            source_time=g,
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=51,
            broadband_method="pole_residue",
        )

    # pole_residue rejects num_freqs < 3 (underdetermined system)
    with pytest.raises(ValidationError):
        td.ModeSource(
            size=(0, 1, 1),
            direction="+",
            source_time=g,
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=2,
            broadband_method="pole_residue",
        )

    # Literal type rejects invalid broadband_method
    with pytest.raises(ValidationError):
        td.ModeSource(
            size=(0, 1, 1),
            direction="+",
            source_time=g,
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=5,
            broadband_method="invalid",
        )

    # PlaneWave rejects pole_residue (only chebyshev is supported)
    with pytest.raises(ValidationError):
        td.PlaneWave(
            size=(0, 1, 1),
            source_time=g,
            direction="+",
            num_freqs=5,
            broadband_method="pole_residue",
        )

    # TFSF rejects pole_residue (only chebyshev is supported)
    with pytest.raises(ValidationError):
        td.TFSF(
            size=(1, 1, 1),
            source_time=g,
            direction="+",
            injection_axis=0,
            num_freqs=5,
            broadband_method="pole_residue",
        )


def test_custom_source_time():
    ts = np.linspace(0, 30e-12, 1001)
    amp_time = ts / max(ts)
    freq0 = 1e12

    # basic test
    cst = td.CustomSourceTime.from_values(
        freq0=freq0, fwidth=0.1e12, values=amp_time, dt=ts[1] - ts[0]
    )
    assert np.allclose(
        cst.amp_time(ts), amp_time * np.exp(-1j * 2 * np.pi * ts * freq0), rtol=0, atol=ATOL
    )

    # test interpolation
    cst = td.CustomSourceTime.from_values(
        freq0=freq0, fwidth=0.1e12, values=np.linspace(0, 9, 10), dt=0.1e-12
    )
    assert np.allclose(
        cst.amp_time(0.09e-12),
        [0.9 * np.exp(-1j * 2 * np.pi * 0.09e-12 * freq0)],
        rtol=0,
        atol=ATOL,
    )

    # test out of range handling
    source = td.PointDipole(center=(0, 0, 0), source_time=cst, polarization="Ex")
    sim = td.Simulation(
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[source],
        normalize_index=None,
    )
    cst = td.CustomSourceTime.from_values(freq0=freq0, fwidth=0.1e12, values=[0, 1], dt=sim.dt)
    source = td.PointDipole(center=(0, 0, 0), source_time=cst, polarization="Ex")
    sim = sim.updated_copy(sources=(source,))
    assert np.allclose(cst.amp_time(sim.tmesh[0]), [0], rtol=0, atol=ATOL)
    assert np.allclose(
        cst.amp_time(sim.tmesh[1:]),
        np.exp(-1j * 2 * np.pi * sim.tmesh[1:] * freq0),
        rtol=0,
        atol=ATOL,
    )

    # all times out of range
    _ = cst.amp_time([-1])
    _ = cst.amp_time(-1)
    assert np.allclose(cst.amp_time([2]), np.exp(-1j * 2 * np.pi * 2 * freq0), rtol=0, atol=ATOL)

    # unsorted time coordinates must raise
    unsorted_vals = td.components.data.data_array.TimeDataArray([1, 2, 3], coords={"t": [0, 2, 1]})
    unsorted_dataset = td.components.data.dataset.TimeDataset(values=unsorted_vals)
    with pytest.raises(ValidationError):
        td.CustomSourceTime(source_time_dataset=unsorted_dataset, freq0=freq0, fwidth=0.1e12)

    vals = td.components.data.data_array.TimeDataArray([1, 2], coords={"t": [-1, -0.5]})
    dataset = td.components.data.dataset.TimeDataset(values=vals)
    cst = td.CustomSourceTime(source_time_dataset=dataset, freq0=freq0, fwidth=0.1e12)
    source = td.PointDipole(center=(0, 0, 0), source_time=cst, polarization="Ex")
    with AssertLogLevel("WARNING", contains_str="defined over a time range"):
        sim = sim.updated_copy(sources=(source,))

    # test normalization warning
    with AssertLogLevel("WARNING"):
        sim = sim.updated_copy(normalize_index=0)

    with AssertLogLevel("WARNING"):
        source = source.updated_copy(source_time=td.ContinuousWave(freq0=freq0, fwidth=0.1e12))
        sim = sim.updated_copy(sources=(source,))

    # test single value validation error
    with pytest.raises(ValidationError):
        vals = td.components.data.data_array.TimeDataArray([1], coords={"t": [0]})
        dataset = td.components.data.dataset.TimeDataset(values=vals)
        cst = td.CustomSourceTime(source_time_dataset=dataset, freq0=freq0, fwidth=0.1e12)
        assert np.allclose(cst.amp_time([0]), [1], rtol=0, atol=ATOL)

    # dt <= 0 must raise
    with pytest.raises(td.exceptions.ValidationError):
        td.CustomSourceTime.from_values(freq0=1e12, fwidth=1e11, values=np.array([0, 1]), dt=0)
    with pytest.raises(td.exceptions.ValidationError):
        td.CustomSourceTime.from_values(freq0=1e12, fwidth=1e11, values=np.array([0, 1]), dt=-1)

    # all-zero dataset: end_time returns None instead of crashing
    cst_zeros = td.CustomSourceTime.from_values(
        freq0=1e12, fwidth=1e11, values=np.zeros(5), dt=1e-12
    )
    assert cst_zeros.end_time() is None


def test_custom_field_source():
    Nx, Ny, Nz, Nf = 4, 3, 1, 1
    X = np.linspace(-1, 1, Nx)
    Y = np.linspace(-1, 1, Ny)
    Z = [0]
    freqs = [2e14]
    n_data = np.ones((Nx, Ny, Nz, Nf))
    n_dataset = td.ScalarFieldDataArray(n_data, coords={"x": X, "y": Y, "z": Z, "f": freqs})

    def make_custom_field_source(field_ds):
        custom_source = td.CustomFieldSource(
            center=(1, 1, 1), size=(2, 2, 0), source_time=ST, field_dataset=field_ds
        )
        return custom_source

    field_dataset = td.FieldDataset(Ex=n_dataset, Hy=n_dataset)
    with AssertLogLevel(None):
        make_custom_field_source(field_dataset)

    with pytest.raises(ValidationError):
        # repeat some entries so data cannot be interpolated
        X2 = [X[0], *list(X)]
        n_data2 = np.vstack((n_data[0, :, :, :].reshape(1, Ny, Nz, Nf), n_data))
        n_dataset2 = td.ScalarFieldDataArray(n_data2, coords={"x": X2, "y": Y, "z": Z, "f": freqs})
        field_dataset = td.FieldDataset(Ex=n_dataset, Hy=n_dataset2)
        make_custom_field_source(field_dataset)


def test_fixed_angle_source():
    plane_wave = td.PlaneWave(
        size=(0, td.inf, td.inf),
        direction="+",
        angle_theta=np.pi / 6,
        angle_phi=np.pi / 4,
        pol_angle=np.pi / 5,
        source_time=td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0),
        angular_spec=td.FixedInPlaneKSpec(),
    )

    assert not plane_wave._is_periodic_fixed_angle

    plane_wave = td.PlaneWave(
        size=(0, td.inf, td.inf),
        direction="+",
        angle_theta=np.pi / 6,
        angle_phi=np.pi / 4,
        pol_angle=np.pi / 5,
        source_time=td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0),
        angular_spec=td.FixedAngleSpec(),
    )

    assert plane_wave._is_periodic_fixed_angle

    plane_wave = td.PlaneWave(
        size=(0, td.inf, td.inf),
        direction="+",
        angle_theta=0,
        angle_phi=np.pi / 4,
        pol_angle=np.pi / 5,
        source_time=td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0),
        angular_spec=td.FixedAngleSpec(),
    )

    assert not plane_wave._is_periodic_fixed_angle


def test_fixed_angle_tfsf_source():
    """TFSF accepts ``angular_spec`` like PlaneWave. ``_is_periodic_fixed_angle``
    is always False on TFSF — the predicate is only true for a fixed-angle
    :class:`PlaneWave` with non-zero theta."""
    g = td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0)

    tfsf = td.TFSF(
        size=(1, 1, 1),
        direction="+",
        injection_axis=2,
        angle_theta=np.pi / 6,
        angle_phi=np.pi / 4,
        pol_angle=np.pi / 5,
        source_time=g,
        angular_spec=td.FixedInPlaneKSpec(),
    )
    assert not tfsf._is_periodic_fixed_angle
    assert isinstance(tfsf.angular_spec, td.FixedInPlaneKSpec)

    tfsf_fixed_angle = tfsf.updated_copy(angular_spec=td.FixedAngleSpec())
    assert not tfsf_fixed_angle._is_periodic_fixed_angle
    assert isinstance(tfsf_fixed_angle.angular_spec, td.FixedAngleSpec)


def test_fixed_angle_tfsf_requires_absorbing_transverse():
    """A fixed-angle TFSF must use absorbing (PML/StablePML/Absorber/ABC)
    transverse boundaries, not Bloch or Periodic — fixed-angle TFSF
    models an isolated scatterer, so periodic/Bloch transverse BCs are
    physically inconsistent with the assumed background."""
    g = td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0)
    tfsf = td.TFSF(
        size=(1, 1, 1),
        direction="+",
        injection_axis=2,
        angle_theta=np.pi / 6,
        source_time=g,
        angular_spec=td.FixedAngleSpec(),
    )

    # Bloch on tangential axes is rejected.
    with pytest.raises(ValidationError, match="Fixed-angle TFSF"):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[tfsf],
            normalize_index=None,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.bloch_from_source(source=tfsf, domain_size=2, axis=0),
                y=td.Boundary.bloch_from_source(source=tfsf, domain_size=2, axis=1),
                z=td.Boundary.pml(),
            ),
        )

    # Periodic on tangential axes is also rejected.
    with pytest.raises(ValidationError, match="Fixed-angle TFSF"):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[tfsf],
            normalize_index=None,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.periodic(),
                y=td.Boundary.periodic(),
                z=td.Boundary.pml(),
            ),
        )

    # PML on tangential axes is allowed.
    _ = td.Simulation(
        size=(2, 2, 2),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[tfsf],
        normalize_index=None,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
    )


def test_fixed_angle_tfsf_rejects_glancing_angle_theta():
    """Fixed-angle TFSF rejects ``angle_theta`` close to ±π/2 (where the
    ``1/sqrt(cos θ)`` source-amplitude normalization is singular)."""
    g = td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0)

    def _sim(angle_theta: float) -> td.Simulation:
        tfsf = td.TFSF(
            size=(1, 1, 1),
            direction="+",
            injection_axis=2,
            angle_theta=angle_theta,
            source_time=g,
            angular_spec=td.FixedAngleSpec(),
        )
        return td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[tfsf],
            normalize_index=None,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml()
            ),
        )

    # Glancing positive theta is rejected.
    with pytest.raises(ValidationError, match="glancing"):
        _ = _sim(np.pi / 2)
    with pytest.raises(ValidationError, match="glancing"):
        _ = _sim(np.pi / 2 - 0.05)  # inside the GLANCING_CUTOFF
    # Glancing negative theta is rejected (symmetric).
    with pytest.raises(ValidationError, match="glancing"):
        _ = _sim(-np.pi / 2 + 0.05)
    # Sufficiently below glancing is allowed (covers both signs).
    _ = _sim(np.pi / 3)
    _ = _sim(-np.pi / 3)
    _ = _sim(0.0)


def test_tfsf_rejects_periodic_or_bloch_on_injection_axis():
    """Periodic / Bloch boundaries on a TFSF source's injection axis are
    physically inconsistent (the wave would re-enter the simulation
    through the opposite face) — rejected for any TFSF, both fixed-angle
    and constant-in-plane-k."""
    g = td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0)
    for angular_spec in (td.FixedInPlaneKSpec(), td.FixedAngleSpec()):
        tfsf = td.TFSF(
            size=(1, 1, 1),
            direction="+",
            injection_axis=2,
            angle_theta=0.0,
            source_time=g,
            angular_spec=angular_spec,
        )
        # Periodic on injection axis: rejected.
        with pytest.raises(ValidationError, match="injection axis"):
            _ = td.Simulation(
                size=(2, 2, 2),
                run_time=1e-12,
                grid_spec=td.GridSpec.uniform(dl=0.1),
                sources=[tfsf],
                normalize_index=None,
                boundary_spec=td.BoundarySpec(
                    x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.periodic()
                ),
            )
        # Bloch on injection axis: also rejected. (`bloch_from_source`
        # itself errors with "Bloch axis must be orthogonal to injection
        # axis"; construct the Bloch BC directly to test the validator.)
        with pytest.raises(ValidationError, match="injection axis"):
            _ = td.Simulation(
                size=(2, 2, 2),
                run_time=1e-12,
                grid_spec=td.GridSpec.uniform(dl=0.1),
                sources=[tfsf],
                normalize_index=None,
                boundary_spec=td.BoundarySpec(
                    x=td.Boundary.pml(),
                    y=td.Boundary.pml(),
                    z=td.Boundary(
                        plus=td.BlochBoundary(bloch_vec=0.5),
                        minus=td.BlochBoundary(bloch_vec=0.5),
                    ),
                ),
            )


def test_fixed_angle_tfsf_rejects_fixed_angle_planewave():
    """A fixed-angle PlaneWave must be the only source in the
    simulation, so combining it with a fixed-angle TFSF must be
    rejected. The two sources also have incompatible transverse BC
    requirements (fixed-angle TFSF wants non-Bloch; fixed-angle
    PlaneWave wants Bloch matching the angle), so any BC choice trips
    at least one of the two validators; this test only asserts that
    rejection happens regardless of which fires first."""
    g = td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0)
    tfsf = td.TFSF(
        size=(1, 1, 1),
        direction="+",
        injection_axis=2,
        angle_theta=np.pi / 6,
        source_time=g,
        angular_spec=td.FixedAngleSpec(),
    )
    pw = td.PlaneWave(
        size=(td.inf, td.inf, 0),
        center=(0, 0, -0.8),
        direction="+",
        angle_theta=np.pi / 6,
        source_time=g,
        angular_spec=td.FixedAngleSpec(),
    )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[tfsf, pw],
            normalize_index=None,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            ),
        )


def _make_fixed_angle_tfsf(**kwargs):
    """Helper: a 1×1×1 fixed-angle TFSF source with default oblique
    incidence and a default GaussianPulse, overridable via kwargs."""
    defaults = {
        "size": (1.0, 1.0, 1.0),
        "center": (0, 0, 0),
        "direction": "+",
        "injection_axis": 2,
        "angle_theta": np.pi / 6,
        "angle_phi": 0.0,
        "pol_angle": np.pi / 2,
        "source_time": td.GaussianPulse(freq0=2e14, fwidth=0.4e14),
        "angular_spec": td.FixedAngleSpec(),
    }
    defaults.update(kwargs)
    return td.TFSF(**defaults)


def _make_fixed_angle_sim(sources, structures=(), **kwargs):
    """Helper: a 1.5×1.5×3 simulation with PML on all sides, suitable
    for a fixed-angle TFSF 1×1×1 box centered at the origin."""
    defaults = {
        "size": (1.5, 1.5, 3),
        "run_time": 1e-12,
        "grid_spec": td.GridSpec.uniform(dl=0.05),
        "sources": list(sources),
        "structures": list(structures),
        "normalize_index": None,
        "boundary_spec": td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
    }
    defaults.update(kwargs)
    return td.Simulation(**defaults)


def test_fixed_angle_tfsf_requires_two_cells_on_injection_axis():
    """Fixed-angle TFSF needs at least 2 simulation-domain cells along
    the injection axis. The validator errors loc-aware at construction
    time."""
    # Pick a sim z extent so the injection-axis (z) physical domain
    # holds exactly 1 cell at dl=0.05 (so the box+source fit within
    # the resulting 1-cell column).
    tfsf = _make_fixed_angle_tfsf(size=(1.0, 1.0, 0.04))
    with pytest.raises(ValidationError, match="at least 2 grid cells"):
        td.Simulation(
            size=(1.5, 1.5, 0.05),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.05),
            sources=[tfsf],
            normalize_index=None,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        )


@pytest.mark.parametrize("side", ["-", "+"])
def test_fixed_angle_tfsf_rejects_non_semi_infinite_injection_axis(side):
    """Fixed-angle TFSF requires the region between each box face and
    the simulation edge along the injection axis to be a single medium
    (semi-infinite space). A slab whose edge lies between the box face
    and the simulation edge breaks that and is rejected. The
    constant-in-plane-k ``FixedInPlaneKSpec`` TFSF accepts the same
    setup."""
    # Box at z ∈ [-0.5, 0.5]. Slab on the chosen side lies between
    # the box face and the sim edge: introduces a vacuum→slab
    # interface in what's supposed to be semi-infinite space.
    if side == "-":
        slab_center, slab_extent = -1.0, 0.4  # slab at z ∈ [-1.2, -0.8]
    else:
        slab_center, slab_extent = +1.0, 0.4  # slab at z ∈ [+0.8, +1.2]
    slab = td.Structure(
        geometry=td.Box(center=(0, 0, slab_center), size=(td.inf, td.inf, slab_extent)),
        medium=td.Medium(permittivity=4.0),
    )
    fixed_tfsf = _make_fixed_angle_tfsf()
    with pytest.raises(ValidationError, match="semi-infinite"):
        _make_fixed_angle_sim([fixed_tfsf], structures=[slab])
    # Legacy spec accepts the same layered setup.
    legacy_tfsf = _make_fixed_angle_tfsf(angular_spec=td.FixedInPlaneKSpec())
    sim = _make_fixed_angle_sim([legacy_tfsf], structures=[slab])
    assert sim is not None


def test_fixed_angle_tfsf_accepts_layered_extending_to_pml():
    """A slab that fills the full sim z extent (``size_z=td.inf``) is
    one medium from each box face out to the PML — accepted."""
    slab = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, td.inf)),
        medium=td.Medium(permittivity=2.0),
    )
    tfsf = _make_fixed_angle_tfsf()
    sim = _make_fixed_angle_sim([tfsf], structures=[slab])
    assert sim is not None


def test_fixed_angle_tfsf_accepts_2d_periodic_oop():
    """2D fixed-angle TFSF: a transverse axis with ``sim.size[axis] == 0``
    is the out-of-plane axis and conventionally uses Periodic. The
    validator exempts such axes from the absorbing-BC rule because the
    0-width direction is a projection, not a periodic-structure
    assumption."""
    # Tidy3D requires TFSF.size to be volumetric (no 0 components);
    # the 2D-ness comes from the simulation's `size[1]=0`.
    tfsf = _make_fixed_angle_tfsf(angle_phi=0.0)
    sim = td.Simulation(
        size=(1.5, 0, 3),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.05),
        sources=[tfsf],
        normalize_index=None,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
    )
    assert sim is not None


def test_fixed_angle_tfsf_2d_rejects_angle_along_oop_axis():
    """2D fixed-angle TFSF: the wave must have no k-component along the
    0-size axis. φ=π/4 with y as the 2D axis gives ``sin(θ)·sin(φ)``
    projected along y — nonzero, so the validator should reject."""
    tfsf = _make_fixed_angle_tfsf(angle_phi=np.pi / 4)
    with pytest.raises(ValidationError, match="0-size axis"):
        td.Simulation(
            size=(1.5, 0, 3),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.05),
            sources=[tfsf],
            normalize_index=None,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.periodic(),
                z=td.Boundary.pml(),
            ),
        )


def test_fixed_angle_tfsf_2d_accepts_in_plane_angle():
    """Same 2D setup as the rejection test, but with φ=0 so the
    k-vector lies entirely in the in-plane (xz) plane — accepted."""
    tfsf = _make_fixed_angle_tfsf(angle_phi=0.0)
    sim = td.Simulation(
        size=(1.5, 0, 3),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.05),
        sources=[tfsf],
        normalize_index=None,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
    )
    assert sim is not None


@pytest.mark.parametrize(
    "angle_phi, expect_accepted",
    [
        # Regression for the pop_axis-vs-cyclic plane_dirs convention. For
        # ``injection_axis=1`` they differ — pop_axis gives ``(0, 2)``,
        # cyclic ``((1+1)%3, (1+2)%3) = (2, 0)``. With ``sim.size[0] = 0``
        # (2D in x), φ=0 puts the entire transverse k along x — the 0-size
        # axis — and must be REJECTED. The cyclic convention would mis-map
        # the φ=0 projection onto z and accept this case.
        (0.0, False),
        # φ=π/2 puts the transverse k along z (the volumetric axis), so x
        # has zero projection and the case is ACCEPTED. Cyclic would
        # mis-map this projection onto x and reject.
        (np.pi / 2, True),
    ],
)
def test_fixed_angle_tfsf_2d_axis1_uses_pop_axis_plane_dirs(angle_phi, expect_accepted):
    """The 2D validator's ``(angle_phi → tan_dir)`` mapping must use
    ``pop_axis`` to match ``Source._dir_vector``; for ``injection_axis=1``
    this differs from the cyclic ``((axis+1)%3, (axis+2)%3)`` permutation."""
    tfsf = _make_fixed_angle_tfsf(
        injection_axis=1,
        angle_theta=np.pi / 4,
        angle_phi=angle_phi,
    )
    kwargs = {
        "size": (0, 1.5, 3),
        "run_time": 1e-12,
        "grid_spec": td.GridSpec.uniform(dl=0.05),
        "sources": [tfsf],
        "normalize_index": None,
        "boundary_spec": td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
    }
    if expect_accepted:
        sim = td.Simulation(**kwargs)
        assert sim is not None
    else:
        with pytest.raises(ValidationError, match="0-size axis"):
            td.Simulation(**kwargs)


def test_fixed_angle_tfsf_rejects_continuous_wave():
    """``ContinuousWave`` has unbounded time support — never decays by
    ``run_time`` — so the fixed-angle Fourier-series synthesis would
    alias its steady-state back to t=0. Validator rejects.

    The rejection must be the same actionable, source-localized error
    whether ``run_time`` is a scalar or a ``RunTimeSpec``: the localization
    validator short-circuits non-decaying source times before evaluating
    ``_run_time`` (which is itself undefined for a ``RunTimeSpec`` when no
    source decays)."""
    cw_source = _make_fixed_angle_tfsf(
        source_time=td.ContinuousWave(freq0=2e14, fwidth=0.4e14),
    )
    # The actionable error is source-localized at ``sources[0]``. Asserting
    # the loc (not just a "decay" substring) distinguishes it from the
    # generic ``RunTimeSpec`` "could not compute source contributions"
    # setup error, whose message also happens to contain "decaying".
    with pytest.raises(ValidationError) as excinfo:
        _make_fixed_angle_sim([cw_source])
    assert_single_value_error_loc(excinfo, ("sources", 0), "unbounded")
    # Same source-localized error with a RunTimeSpec — the localization
    # validator short-circuits the non-decaying source before evaluating
    # ``_run_time`` (which would otherwise raise the generic, non-localized
    # RunTimeSpec error first).
    with pytest.raises(ValidationError) as excinfo_spec:
        _make_fixed_angle_sim([cw_source], run_time=td.RunTimeSpec(quality_factor=2.0))
    assert_single_value_error_loc(excinfo_spec, ("sources", 0), "unbounded")


def test_fixed_angle_tfsf_accepts_run_time_spec():
    """Regression: the fixed-angle TFSF validators/warnings must use the
    evaluated ``Simulation._run_time`` rather than ``float(self.run_time)``,
    which raises ``TypeError`` when ``run_time`` is a :class:`RunTimeSpec`.
    Construction with a ``RunTimeSpec`` must succeed (and the evaluated
    run time must be a finite positive float)."""
    tfsf = _make_fixed_angle_tfsf()
    sim = _make_fixed_angle_sim([tfsf], run_time=td.RunTimeSpec(quality_factor=3.0))
    assert isinstance(sim.run_time, td.RunTimeSpec)
    assert np.isfinite(sim._run_time) and sim._run_time > 0.0


@pytest.mark.parametrize(
    "angular_spec",
    [td.FixedInPlaneKSpec(), td.FixedAngleSpec()],
)
def test_tfsf_rejects_anisotropic_source_plane(angular_spec):
    """Both TFSF angular specs reject ``AnisotropicMedium`` /
    ``FullyAnisotropicMedium`` on the injection plane. Validator fires
    at simulation construction."""
    aniso = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2.0),
        yy=td.Medium(permittivity=2.5),
        zz=td.Medium(permittivity=3.0),
    )
    aniso_layer = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, td.inf)),
        medium=aniso,
    )
    tfsf = _make_fixed_angle_tfsf(angular_spec=angular_spec)
    with pytest.raises(ValidationError, match="anisotropic"):
        _make_fixed_angle_sim([tfsf], structures=[aniso_layer])


def test_fixed_angle_tfsf_rejects_sidewall_surface_bc():
    """Fixed-angle TFSF rejects ``LossyMetalMedium`` / ``PECMedium`` /
    ``PMCMedium`` structures that intersect its sidewalls. The
    constant-in-plane-k ``FixedInPlaneKSpec`` TFSF accepts the same
    setup. Check fires at ``validate_pre_upload``."""
    metal = td.LossyMetalMedium(conductivity=10, frequency_range=(1e14, 4e14))
    layer = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, 0.2)),
        medium=metal,
    )
    tfsf = _make_fixed_angle_tfsf()
    sim = _make_fixed_angle_sim([tfsf], structures=[layer])
    with pytest.raises(ValidationError, match="LossyMetalMedium"):
        sim.validate_pre_upload()
    # The constant-in-plane-k spec passes the same sidewall check.
    legacy_tfsf = _make_fixed_angle_tfsf(angular_spec=td.FixedInPlaneKSpec())
    legacy_sim = _make_fixed_angle_sim([legacy_tfsf], structures=[layer])
    legacy_sim._validate_tfsf_structure_intersections()
    # A penetrable lossy metal is a regular medium, not a surface BC, so a
    # fixed-angle TFSF sidewall may intersect it.
    penetrable_layer = layer.updated_copy(medium=metal.updated_copy(penetrable=True))
    penetrable_sim = _make_fixed_angle_sim([tfsf], structures=[penetrable_layer])
    penetrable_sim._validate_tfsf_structure_intersections()


def test_fixed_angle_tfsf_normal_incidence_warns():
    """θ=0 with ``FixedAngleSpec`` emits a warning suggesting the
    legacy ``FixedInPlaneKSpec`` (Bloch TFSF) is equivalent and faster."""
    tfsf = _make_fixed_angle_tfsf(angle_theta=0.0)
    with AssertLogStr("WARNING", contains_str="angle_theta=0"):
        sim = _make_fixed_angle_sim([tfsf])
    assert sim is not None


def test_fixed_angle_tfsf_long_run_time_warns():
    """Long ``run_time`` + wide ``fwidth`` with ``FixedAngleSpec`` emits
    a warning that the O(run_time²) fixed-angle cost may dominate the
    FDTD update cost. Threshold is ``run_time·fwidth > 500``."""
    tfsf = _make_fixed_angle_tfsf()
    # With fwidth=0.4e14, run_time > ~12.5 ps triggers the threshold.
    # Pick 100 ps to be well above.
    with AssertLogStr("WARNING", contains_str="run_time ** 2"):
        sim = td.Simulation(
            size=(1.5, 1.5, 3),
            run_time=1e-10,
            grid_spec=td.GridSpec.uniform(dl=0.05),
            sources=[tfsf],
            normalize_index=None,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            ),
        )
    assert sim is not None


def test_fixed_angle_tfsf_rejects_nonuniform_transverse_grid():
    """Non-uniform transverse grid inside the TFSF box is rejected —
    fixed-angle TFSF requires uniform transverse spacing."""
    tfsf = _make_fixed_angle_tfsf()
    fine_x = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, 0, 0), size=(0.4, td.inf, td.inf)),
        dl=(0.025, None, None),
    )
    with pytest.raises(ValidationError, match="uniform transverse grid"):
        td.Simulation(
            size=(1.5, 1.5, 3),
            run_time=1e-12,
            grid_spec=td.GridSpec.auto(
                wavelength=0.5,
                min_steps_per_wvl=10,
                override_structures=[fine_x],
            ),
            sources=[tfsf],
            normalize_index=None,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            ),
        )


def test_plane_wave_critical_frequency_error_loc():
    with pytest.raises(ValidationError) as excinfo:
        _ = td.PlaneWave(
            size=(0, td.inf, td.inf),
            direction="+",
            angle_theta=np.pi / 2,
            angle_phi=0.0,
            pol_angle=0.0,
            source_time=td.GaussianPulse(freq0=1e14, fwidth=1e8),
            num_freqs=3,
        )

    assert_single_value_error_loc(
        excinfo,
        ("num_freqs",),
        "critical frequency of oblique incidence",
    )


def test_broadband_angled_gaussian_warning():
    g = td.GaussianPulse(freq0=1e14, fwidth=0.8e14)
    # Case 1: num_freqs = 3, angle_theta = np.pi / 3, should warn
    with AssertLogLevel("WARNING", contains_str="number of frequencies"):
        s = td.GaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            angle_theta=np.pi / 3,
            num_freqs=3,
        )
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[s],
            normalize_index=None,
        )

    # Case 2: Increasing to num_freqs = 10 should NOT warn
    with AssertLogLevel(None):
        s = td.GaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            angle_theta=np.pi / 3,
            num_freqs=10,
        )
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[s],
            normalize_index=None,
        )

    # Case 3: Case 2 but changed to astigmatic gaussian beam with one larger waist size should warn
    with AssertLogLevel("WARNING", contains_str="number of frequencies"):
        s = td.AstigmaticGaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            angle_theta=np.pi / 3,
            num_freqs=10,
            waist_sizes=(1, 5),
        )
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[s],
            normalize_index=None,
        )

    # Case 4: Case 3 but with num_freqs = 1 should NOT warn (broadband treatment is off)
    with AssertLogLevel(None):
        s = td.AstigmaticGaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            angle_theta=np.pi / 3,
            num_freqs=1,
            waist_sizes=(1, 5),
        )
        _ = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[s],
            normalize_index=None,
        )


def test_source_frame():
    _ = td.PECFrame()
    _ = td.PECFrame(length=4)
    with pytest.raises(ValidationError):
        _ = td.PECFrame(length=0)

    _ = td.ModeSource(
        source_time=td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0),
        direction="+",
        size=(1, 1, 0),
        frame=td.PECFrame(),
    )


def test_rf_frequency_range_miswarning():
    """GaussianPulse with DC removed is asymmetric. More accurate frequency range
    computation for validating if it's a photonics simulation.
    """
    source_time = td.GaussianPulse(freq0=400e12, fwidth=99.999e12)
    source = td.PointDipole(center=(0, 0, 0), polarization="Ex", source_time=source_time)
    with AssertLogStr("WARNING", excludes_str="outside of the simulation domain"):
        sim = td.Simulation(
            size=(1, 1, 1),
            run_time=td.RunTimeSpec(quality_factor=1),
            sources=[source],
        )
