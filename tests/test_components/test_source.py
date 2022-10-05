"""Tests sources."""
import pytest
import pydantic
import matplotlib.pylab as plt
import numpy as np
import tidy3d as td
from tidy3d.log import SetupError, DataError, ValidationError
from tidy3d.components.source import DirectionalSource

_, AX = plt.subplots()

ST = td.GaussianPulse(freq0=2e14, fwidth=1e14)
S = td.PointDipole(source_time=ST, polarization="Ex")


def test_plot_source_time():
    ST.plot(times=[1e-15, 2e-15, 3e-15], ax=AX)
    ST.plot_spectrum(times=[1e-15, 2e-15, 3e-15], num_freqs=4, ax=AX)

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
    s = td.UniformCurrentSource(size=(1, 1, 1), source_time=g, polarization="Ez")


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
    p = td.PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex")
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
    with pytest.raises(ValidationError) as e_info:
        s = td.PlaneWave(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(ValidationError) as e_info:
        s = td.GaussianBeam(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(ValidationError) as e_info:
        s = td.AstigmaticGaussianBeam(
            size=(1, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
        )
    with pytest.raises(ValidationError) as e_info:
        s = td.ModeSource(size=(1, 1, 1), source_time=g, mode_spec=mode_spec)

    from tidy3d.components.source import TFSF

    s = TFSF(size=(1, 1, 1), direction="+", source_time=g, injection_axis=2)
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
