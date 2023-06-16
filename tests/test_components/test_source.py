"""Tests sources."""
import pytest
import pydantic
import matplotlib.pylab as plt
import numpy as np
import tidy3d as td
from tidy3d.exceptions import SetupError, DataError, ValidationError
from tidy3d.components.source import DirectionalSource, CHEB_GRID_WIDTH

_, AX = plt.subplots()

ST = td.GaussianPulse(freq0=2e14, fwidth=1e14)
S = td.PointDipole(source_time=ST, polarization="Ex")


def test_plot_source_time():

    for val in ("real", "imag", "abs"):
        ST.plot(times=[1e-15, 2e-15, 3e-15], val=val, ax=AX)
        ST.plot_spectrum(times=[1e-15, 2e-15, 3e-15], num_freqs=4, val=val, ax=AX)

    with pytest.raises(ValueError):
        ST.plot(times=[1e-15, 2e-15, 3e-15], val="blah", ax=AX)

    with pytest.raises(ValueError):
        ST.plot_spectrum(times=[1e-15, 2e-15, 3e-15], num_freqs=4, val="blah", ax=AX)

    # uneven spacing in times
    with pytest.raises(SetupError):
        ST.plot_spectrum(times=[1e-15, 3e-15, 4e-15], num_freqs=4, ax=AX)


def test_dir_vector():
    MS = td.ModeSource(size=(1, 0, 1), mode_spec=td.ModeSpec(), source_time=ST, direction="+")
    DirectionalSource._dir_vector.fget(MS)
    assert DirectionalSource._dir_vector.fget(S) is None


def test_UniformCurrentSource():

    g = td.GaussianPulse(freq0=1, fwidth=0.1)

    # test we can make generic UniformCurrentSource
    s1 = td.UniformCurrentSource(
        size=(1, 1, 1), source_time=g, polarization="Ez", interpolate=False
    )
    s2 = td.UniformCurrentSource(size=(1, 1, 1), source_time=g, polarization="Ez", interpolate=True)


def test_source_times():

    # test we can make gaussian pulse
    g = td.GaussianPulse(freq0=1, fwidth=0.1)
    ts = np.linspace(0, 30, 1001)
    g.amp_time(ts)
    # g.plot(ts)

    # test we can make cw pulse
    from tidy3d.components.source import ContinuousWave

    c = ContinuousWave(freq0=1, fwidth=0.1)
    ts = np.linspace(0, 30, 1001)
    c.amp_time(ts)


def test_dipole():

    g = td.GaussianPulse(freq0=1, fwidth=0.1)
    p1 = td.PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex", interpolate=True)
    p2 = td.PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex", interpolate=False)
    # p.plot(y=2)

    with pytest.raises(pydantic.ValidationError) as e_info:
        p = td.PointDipole(size=(1, 1, 1), source_time=g, center=(1, 2, 3), polarization="Ex")


def test_FieldSource():
    g = td.GaussianPulse(freq0=1, fwidth=0.1)
    mode_spec = td.ModeSpec(num_modes=2)

    # test we can make planewave
    s = td.PlaneWave(size=(0, td.inf, td.inf), source_time=g, pol_angle=np.pi / 2, direction="+")
    # s.plot(y=0)

    # test we can make gaussian beam
    s = td.GaussianBeam(size=(0, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    # s.plot(y=0)

    # test we can make an astigmatic gaussian beam
    s = td.AstigmaticGaussianBeam(
        size=(0, 1, 1),
        source_time=g,
        pol_angle=np.pi / 2,
        direction="+",
        waist_sizes=(0.2, 0.4),
        waist_distances=(0.1, 0.3),
    )

    # test we can make mode source
    s = td.ModeSource(
        size=(0, 1, 1), direction="+", source_time=g, mode_spec=mode_spec, mode_index=0
    )
    # s.plot(y=0)

    # test that non-planar geometry crashes plane wave and gaussian beams
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.PlaneWave(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.GaussianBeam(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.AstigmaticGaussianBeam(
            size=(1, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
        )
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.ModeSource(size=(1, 1, 1), source_time=g, mode_spec=mode_spec)

    from tidy3d.components.source import TFSF

    tfsf = td.TFSF(size=(1, 1, 1), direction="+", source_time=g, injection_axis=2)
    _ = tfsf.injection_plane_center

    # assert that TFSF must be volumetric
    with pytest.raises(pydantic.ValidationError) as e_info:
        _ = td.TFSF(size=(1, 1, 0), direction="+", source_time=g, injection_axis=2)

    # s.plot(z=0)


def test_pol_arrow():

    g = td.GaussianPulse(freq0=1, fwidth=0.1)

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
    assert np.allclose(get_pol_dir(axis=0, angle_phi=np.pi / 2), (0, 0, +1))
    assert np.allclose(get_pol_dir(axis=1, angle_phi=np.pi / 2), (0, 0, -1))
    assert np.allclose(get_pol_dir(axis=2, angle_phi=np.pi / 2), (0, +1, 0))
    assert np.allclose(get_pol_dir(axis=0, pol_angle=np.pi / 2), (0, 0, +1))
    assert np.allclose(get_pol_dir(axis=1, pol_angle=np.pi / 2), (0, 0, -1))
    assert np.allclose(get_pol_dir(axis=2, pol_angle=np.pi / 2), (0, +1, 0))
    assert np.allclose(
        get_pol_dir(axis=0, angle_theta=np.pi / 4), (+1 / np.sqrt(2), -1 / np.sqrt(2), 0)
    )
    assert np.allclose(
        get_pol_dir(axis=1, angle_theta=np.pi / 4), (-1 / np.sqrt(2), +1 / np.sqrt(2), 0)
    )
    assert np.allclose(
        get_pol_dir(axis=2, angle_theta=np.pi / 4), (-1 / np.sqrt(2), 0, +1 / np.sqrt(2))
    )


def test_broadband_source():
    g = td.GaussianPulse(freq0=1, fwidth=0.1)
    mode_spec = td.ModeSpec(num_modes=2)
    fmin, fmax = g.frequency_range(num_fwidth=CHEB_GRID_WIDTH)
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
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.GaussianBeam(
            size=(0, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+", num_freqs=200
        )
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.AstigmaticGaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
            num_freqs=100,
        )
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.ModeSource(
            size=(0, 1, 1),
            direction="+",
            source_time=g,
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=-10,
        )
