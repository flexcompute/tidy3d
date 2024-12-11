"""Tests sources."""

import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.components.source import CHEB_GRID_WIDTH, DirectionalSource
from tidy3d.exceptions import SetupError

from ..utils import AssertLogLevel, assert_log_level

ST = td.GaussianPulse(freq0=2e14, fwidth=1e14)
S = td.PointDipole(source_time=ST, polarization="Ex")


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
    _ = td.UniformCurrentSource(size=(1, 1, 1), source_time=g, polarization="Ez", interpolate=False)
    _ = td.UniformCurrentSource(size=(1, 1, 1), source_time=g, polarization="Ez", interpolate=True)


def test_source_times():
    # test we can make gaussian pulse
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)
    ts = np.linspace(0, 30, 1001) * 1e-12
    g.amp_time(ts)
    # g.plot(ts)
    # plt.close()

    # test we can make cw pulse
    from tidy3d.components.source import ContinuousWave

    c = ContinuousWave(freq0=1e12, fwidth=0.1e12)
    c.amp_time(ts)

    # test gaussian pulse with and without DC component
    g = td.GaussianPulse(freq0=0.1e12, fwidth=1e12)
    dc_comp = g.spectrum(ts, [0], ts[1] - ts[0])
    assert abs(dc_comp) ** 2 < 1e-32
    g = td.GaussianPulse(freq0=0.1e12, fwidth=1e12, remove_dc_component=False)
    dc_comp = g.spectrum(ts, [0], ts[1] - ts[0])
    assert abs(dc_comp) ** 2 > 1e-32


def test_dipole():
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)
    _ = td.PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex", interpolate=True)
    _ = td.PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex", interpolate=False)
    # p.plot(y=2)
    # plt.close()

    with pytest.raises(pydantic.ValidationError):
        _ = td.PointDipole(size=(1, 1, 1), source_time=g, center=(1, 2, 3), polarization="Ex")


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
    with pytest.raises(pydantic.ValidationError):
        _ = td.PlaneWave(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(pydantic.ValidationError):
        _ = td.GaussianBeam(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(pydantic.ValidationError):
        _ = td.AstigmaticGaussianBeam(
            size=(1, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
        )
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSource(size=(1, 1, 1), source_time=g, mode_spec=mode_spec)

    tfsf = td.TFSF(size=(1, 1, 1), direction="+", source_time=g, injection_axis=2)
    _ = tfsf.injection_plane_center

    # assert that TFSF must be volumetric
    with pytest.raises(pydantic.ValidationError):
        _ = td.TFSF(size=(1, 1, 0), direction="+", source_time=g, injection_axis=2)

    # s.plot(z=0)
    # plt.close()


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
    with pytest.raises(pydantic.ValidationError):
        s = td.GaussianBeam(
            size=(0, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+", num_freqs=200
        )
    with pytest.raises(pydantic.ValidationError):
        s = td.AstigmaticGaussianBeam(
            size=(0, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
            num_freqs=100,
        )
    with pytest.raises(pydantic.ValidationError):
        s = td.ModeSource(
            size=(0, 1, 1),
            direction="+",
            source_time=g,
            mode_spec=mode_spec,
            mode_index=0,
            num_freqs=-10,
        )


def test_custom_source_time(log_capture):
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
    sim = sim.updated_copy(sources=[source])
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

    assert_log_level(log_capture, None)

    vals = td.components.data.data_array.TimeDataArray([1, 2], coords=dict(t=[-1, -0.5]))
    dataset = td.components.data.dataset.TimeDataset(values=vals)
    cst = td.CustomSourceTime(source_time_dataset=dataset, freq0=freq0, fwidth=0.1e12)
    source = td.PointDipole(center=(0, 0, 0), source_time=cst, polarization="Ex")
    with AssertLogLevel(log_capture, "WARNING", contains_str="defined over a time range"):
        sim = sim.updated_copy(sources=[source])

    # test normalization warning
    with AssertLogLevel(log_capture, "WARNING"):
        sim = sim.updated_copy(normalize_index=0)

    with AssertLogLevel(log_capture, "WARNING"):
        source = source.updated_copy(source_time=td.ContinuousWave(freq0=freq0, fwidth=0.1e12))
        sim = sim.updated_copy(sources=[source])

    # test single value validation error
    with pytest.raises(pydantic.ValidationError):
        vals = td.components.data.data_array.TimeDataArray([1], coords=dict(t=[0]))
        dataset = td.components.data.dataset.TimeDataset(values=vals)
        cst = td.CustomSourceTime(source_time_dataset=dataset, freq0=freq0, fwidth=0.1e12)
        assert np.allclose(cst.amp_time([0]), [1], rtol=0, atol=ATOL)


def test_custom_field_source(log_capture):
    Nx, Ny, Nz, Nf = 4, 3, 1, 1
    X = np.linspace(-1, 1, Nx)
    Y = np.linspace(-1, 1, Ny)
    Z = [0]
    freqs = [2e14]
    n_data = np.ones((Nx, Ny, Nz, Nf))
    n_dataset = td.ScalarFieldDataArray(n_data, coords=dict(x=X, y=Y, z=Z, f=freqs))

    def make_custom_field_source(field_ds):
        custom_source = td.CustomFieldSource(
            center=(1, 1, 1), size=(2, 2, 0), source_time=ST, field_dataset=field_ds
        )
        return custom_source

    field_dataset = td.FieldDataset(Ex=n_dataset, Hy=n_dataset)
    make_custom_field_source(field_dataset)
    assert_log_level(log_capture, None)

    with pytest.raises(pydantic.ValidationError):
        # repeat some entries so data cannot be interpolated
        X2 = [X[0]] + list(X)
        n_data2 = np.vstack((n_data[0, :, :, :].reshape(1, Ny, Nz, Nf), n_data))
        n_dataset2 = td.ScalarFieldDataArray(n_data2, coords=dict(x=X2, y=Y, z=Z, f=freqs))
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

    assert not plane_wave._is_fixed_angle

    plane_wave = td.PlaneWave(
        size=(0, td.inf, td.inf),
        direction="+",
        angle_theta=np.pi / 6,
        angle_phi=np.pi / 4,
        pol_angle=np.pi / 5,
        source_time=td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0),
        angular_spec=td.FixedAngleSpec(),
    )

    assert plane_wave._is_fixed_angle

    plane_wave = td.PlaneWave(
        size=(0, td.inf, td.inf),
        direction="+",
        angle_theta=0,
        angle_phi=np.pi / 4,
        pol_angle=np.pi / 5,
        source_time=td.GaussianPulse(freq0=td.C_0, fwidth=0.2 * td.C_0),
        angular_spec=td.FixedAngleSpec(),
    )

    assert not plane_wave._is_fixed_angle
