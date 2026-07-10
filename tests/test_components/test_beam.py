"""Tests for the various BeamProfile components."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import j1, jv

import tidy3d as td
from tidy3d.components import simulation as simulation_module
from tidy3d.components.beam import (
    AstigmaticGaussianBeamProfile,
    BeamGrid,
    BeamPose,
    GaussianBeamProfile,
    PlaneWaveBeamProfile,
    ThinLensProfile,
)
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.grid.grid import _compute_1d_cell_sizes

from ..utils import AssertLogLevel, assert_single_value_error_loc

FREQS = np.linspace(1e14, 2e14, 10).tolist()


def _thin_lens_uniform_pupil_power_fractions(na: float, background_n: float = 1.0) -> np.ndarray:
    """Analytic component power fractions for a filled x-polarized aplanatic pupil."""
    cos_theta_max = np.sqrt(1 - (na / background_n) ** 2)
    power_x = np.pi / 4 * (5 - 3 * cos_theta_max - cos_theta_max**2 - cos_theta_max**3)
    power_y = np.pi / 12 * (1 - cos_theta_max) ** 3
    power_z = np.pi * (2 / 3 - cos_theta_max + cos_theta_max**3 / 3)
    powers = np.array([power_x, power_y, power_z])
    return powers / np.sum(powers)


def _airy_disk_intensity(radius: np.ndarray, na: float, wavelength: float) -> np.ndarray:
    """Normalized scalar Airy intensity for a uniformly filled circular pupil."""
    arg = 2 * np.pi * na * radius / wavelength
    amplitude = np.ones_like(arg)
    nonzero = arg != 0
    amplitude[nonzero] = 2 * j1(arg[nonzero]) / arg[nonzero]
    return np.abs(amplitude) ** 2


def _beam_plane_flux(
    fields: dict[str, np.ndarray],
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normal_axis: int,
) -> np.ndarray:
    """Compute discrete flux through a beam sampling plane."""
    coords = (x, y, z)
    tangential_axes = [idx for idx in range(3) if idx != normal_axis]
    ax1, ax2 = tangential_axes

    def _normal_plane(values: np.ndarray) -> np.ndarray:
        if normal_axis == 0:
            return values[0, :, :, :]
        if normal_axis == 1:
            return values[:, 0, :, :]
        return values[:, :, 0, :]

    e1 = _normal_plane(fields[f"E{'xyz'[ax1]}"])
    e2 = _normal_plane(fields[f"E{'xyz'[ax2]}"])
    h1 = _normal_plane(fields[f"H{'xyz'[ax1]}"])
    h2 = _normal_plane(fields[f"H{'xyz'[ax2]}"])
    complex_poynting = 0.5 * (e1 * np.conj(h2) - e2 * np.conj(h1))
    d_area = (
        _compute_1d_cell_sizes(coords[ax1])[:, np.newaxis]
        * _compute_1d_cell_sizes(coords[ax2])[np.newaxis, :]
    )
    return np.real(np.sum(complex_poynting * d_area[:, :, np.newaxis], axis=(0, 1)))


def _richards_wolf_x_polarized_focal_field(
    points: np.ndarray,
    na: float,
    wavelength: float,
    background_n: float = 1.0,
    num_quad: int = 600,
) -> np.ndarray:
    """Independent Richards-Wolf focal-plane field for a filled x-polarized pupil."""
    alpha = np.arcsin(na / background_n)
    nodes, weights = np.polynomial.legendre.leggauss(num_quad)
    theta = 0.5 * alpha * (nodes + 1)
    theta_weights = 0.5 * alpha * weights
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sqrt_cos_theta = np.sqrt(cos_theta)

    x_points, y_points, z_points = points
    np.testing.assert_allclose(z_points, 0.0, rtol=0, atol=1e-15)

    radius = np.hypot(x_points, y_points)
    azimuth = np.arctan2(y_points, x_points)
    k_medium = 2 * np.pi * background_n / wavelength
    bessel_arg = np.outer(k_medium * radius, sin_theta)

    weight_i0 = sqrt_cos_theta * sin_theta * (1 + cos_theta) * theta_weights
    weight_i1 = sqrt_cos_theta * sin_theta**2 * theta_weights
    weight_i2 = sqrt_cos_theta * sin_theta * (1 - cos_theta) * theta_weights
    integral_0 = jv(0, bessel_arg) @ weight_i0
    integral_1 = jv(1, bessel_arg) @ weight_i1
    integral_2 = jv(2, bessel_arg) @ weight_i2

    return np.stack(
        (
            np.pi * (integral_0 + integral_2 * np.cos(2 * azimuth)),
            np.pi * integral_2 * np.sin(2 * azimuth),
            -2j * np.pi * integral_1 * np.cos(azimuth),
        )
    )


def test_gaussian_beam():
    """
    Test Gaussian beam creation and field data validation.
    """
    center = (0, 0, 0)
    size = (10, 10, 0)
    resolution = 150
    waist_radius = 1.0
    waist_distance = 3.0
    beam = GaussianBeamProfile(
        center=center,
        size=size,
        resolution=resolution,
        freqs=FREQS,
        waist_radius=waist_radius,
        waist_distance=waist_distance,
    )
    field_data = beam.field_data
    assert isinstance(field_data, FieldData)
    assert np.allclose(field_data.flux, 1)


def test_plane_wave():
    """
    Test plane wave creation and field data validation.
    """
    center = (0, 0, 0)
    size = (10, 0, 10)
    resolution = 100
    beam = PlaneWaveBeamProfile(center=center, size=size, resolution=resolution, freqs=FREQS)
    field_data = beam.field_data
    assert isinstance(field_data, FieldData)
    assert np.allclose(field_data.flux, 1)


def test_astigmatic_gaussian_beam():
    """
    Test astigmatic Gaussian beam creation, field data validatio variation along y and z axes.
    """
    center = (0, 0, 0)
    size = (0, 20, 20)
    waist_sizes = (4.0, 2.0)
    waist_distances = (1.0, 2.0)
    beam = AstigmaticGaussianBeamProfile(
        center=center,
        size=size,
        freqs=FREQS,
        waist_sizes=waist_sizes,
        waist_distances=waist_distances,
    )
    field_data = beam.field_data
    assert isinstance(field_data, FieldData)
    assert np.allclose(field_data.flux, 1)

    # Test that the spread is larger in the dimension in which the waist size is larger
    ey_variation_y = np.std(field_data.Ey.isel(f=0).sel(z=0, method="nearest").values)
    ey_variation_z = np.std(field_data.Ey.isel(f=0).sel(y=0, method="nearest").values)
    assert ey_variation_y > ey_variation_z

    # Test that the opposite direction field data has opposite flux
    field_data_bwd = beam.updated_copy(direction="-").field_data

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3, 2)
    # field_data.Ex.abs.isel(f=0).plot(ax=ax[0, 0])
    # field_data_bwd.Ex.abs.isel(f=0).plot(ax=ax[0, 1])
    # field_data.Ey.abs.isel(f=0).plot(ax=ax[1, 0])
    # field_data_bwd.Ey.abs.isel(f=0).plot(ax=ax[1, 1])
    # field_data.Ez.abs.isel(f=0).plot(ax=ax[2, 0])
    # field_data_bwd.Ez.abs.isel(f=0).plot(ax=ax[2, 1])
    # plt.show()
    assert field_data.flux == -field_data_bwd.flux


def test_gaussian_beam_profile_backward_waist_distance_warning():
    center = (0, 0, 0)
    size = (10, 10, 0)
    resolution = 100

    with AssertLogLevel(
        "WARNING",
        contains_str="GaussianBeamProfile with direction '-' and non-zero 'waist_distance'",
    ):
        GaussianBeamProfile(
            center=center,
            size=size,
            resolution=resolution,
            freqs=FREQS,
            direction="-",
            waist_distance=1.0,
        )

    with AssertLogLevel(None):
        GaussianBeamProfile(
            center=center,
            size=size,
            resolution=resolution,
            freqs=FREQS,
            direction="-",
            waist_distance=0.0,
        )


def test_astigmatic_gaussian_beam_profile_backward_waist_distance_warning():
    center = (0, 0, 0)
    size = (0, 20, 20)

    with AssertLogLevel(
        "WARNING",
        contains_str="AstigmaticGaussianBeamProfile with direction '-' and non-zero 'waist_distances'",
    ):
        AstigmaticGaussianBeamProfile(
            center=center,
            size=size,
            freqs=FREQS,
            direction="-",
            waist_sizes=(4.0, 2.0),
            waist_distances=(1.0, 0.0),
        )

    with AssertLogLevel(None):
        AstigmaticGaussianBeamProfile(
            center=center,
            size=size,
            freqs=FREQS,
            direction="-",
            waist_sizes=(4.0, 2.0),
            waist_distances=(0.0, 0.0),
        )


def test_invalid_beam_size():
    """
    Test that a beam with three nonzero size values raises a validation error.
    """
    center = (0, 0, 0)
    size = (10, 10, 10)
    resolution = 100
    with pytest.raises(ValidationError):
        GaussianBeamProfile(center=center, size=size, resolution=resolution, freqs=FREQS)


def test_gaussian_beam_ex_spread_waist_distance():
    """
    Test that the Gaussian beam has a larger spread when the waist distance is decreased.
    """
    center = (0, 0, 0)
    size = (10, 10, 0)
    resolution = 150
    waist_radius = 1.0
    waist_distance_1 = 3.0
    waist_distance_2 = 1.0
    beam_data_1 = GaussianBeamProfile(
        center=center,
        size=size,
        resolution=resolution,
        freqs=FREQS,
        waist_radius=waist_radius,
        waist_distance=waist_distance_1,
    )
    beam_data_2 = GaussianBeamProfile(
        center=center,
        size=size,
        resolution=resolution,
        freqs=FREQS,
        waist_radius=waist_radius,
        waist_distance=waist_distance_2,
    )
    ex_spread_1 = np.std(beam_data_1.field_data.Ex.isel(f=0).values)
    ex_spread_2 = np.std(beam_data_2.field_data.Ex.isel(f=0).values)
    assert ex_spread_2 > ex_spread_1


def test_gaussian_beam_center():
    """
    Test that a Gaussian beam with a nonzero
    angle_theta and a nonzero waist_distance is centered away from x=0, y=0.
    """
    center = (0, 0, 0)
    size = (10, 10, 0)
    resolution = 151
    waist_radius = 1.0

    # Normal beam data should be centered at x=0, y=0
    beam = GaussianBeamProfile(
        center=center,
        size=size,
        resolution=resolution,
        freqs=FREQS,
        waist_radius=waist_radius,
    )

    # Angled beam data without waist_distance should also be centered at x=0, y=0
    beam_data_angled = beam.updated_copy(angle_theta=np.pi / 4, angle_phi=np.pi / 4)

    # Angled beam data with waist_distance should be centered away from the center plane
    beam_data_angled_distance = beam_data_angled.updated_copy(waist_distance=5)

    field_data = beam.field_data
    field_data_angled = beam_data_angled.field_data
    field_data_angled_distance = beam_data_angled_distance.field_data

    def get_center(data):
        ex_field = data.Ex.isel(f=0).abs
        x, y = ex_field.coords["x"].values, ex_field.coords["y"].values
        max_ex_index = np.unravel_index(np.argmax(ex_field.values), ex_field.shape)
        return x[max_ex_index[0]], y[max_ex_index[1]]

    print(get_center(field_data))
    print(get_center(field_data_angled))
    print(get_center(field_data_angled_distance))

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3, 2)
    # field_data.Ex.abs.isel(f=0).plot(ax=ax[0, 0])
    # field_data.Ey.abs.isel(f=0).plot(ax=ax[1, 0])
    # field_data.Ez.abs.isel(f=0).plot(ax=ax[2, 0])
    # field_data_angled_distance.Ex.abs.isel(f=0).plot(ax=ax[0, 1])
    # field_data_angled_distance.Ey.abs.isel(f=0).plot(ax=ax[1, 1])
    # field_data_angled_distance.Ez.abs.isel(f=0).plot(ax=ax[2, 1])
    # plt.show()

    assert np.allclose(get_center(field_data), (0, 0))
    assert np.allclose(get_center(field_data_angled), (0, 0))
    assert np.all(np.abs(get_center(field_data_angled_distance)) > 1)


def test_beam_overlap():
    """Test that an outer dot can be computed between two beams."""
    center = (0, 0, 0)
    size = (10, 10, 0)
    resolution1 = 150
    resolution2 = 200
    waist_radius = 1.0
    waist_distance = 3.0
    beam1 = GaussianBeamProfile(
        center=center,
        size=size,
        resolution=resolution1,
        freqs=FREQS,
        waist_radius=waist_radius,
        waist_distance=waist_distance,
    )
    beam2 = AstigmaticGaussianBeamProfile(
        center=center,
        size=size,
        resolution=resolution2,
        freqs=FREQS,
        waist_sizes=[waist_radius, waist_radius],
        waist_distances=[waist_distance, waist_distance],
    )
    outer_dot = beam1.field_data.outer_dot(beam2.field_data)
    # Equivalent beam definition apart from different discretization, leading to a small mismatch
    assert np.allclose(outer_dot, 1.0, rtol=1e-3)


def test_thin_lens_profile_field_data_and_power_fractions():
    """Test vectorial thin-lens field data against analytic component power fractions."""
    wvl0_um = 1.55
    freq0 = td.C_0 / wvl0_um
    size_val = 20 * wvl0_um
    na = 0.5

    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(size_val, size_val, 0),
        freqs=[freq0],
        numerical_aperture=na,
        resolution=41,
        num_plane_waves=41,
    )
    field_data = beam.field_data

    assert isinstance(field_data, FieldData)
    assert field_data.Ex.shape == (40, 40, 1, 1)
    assert np.allclose(field_data.flux, 1)

    ex = field_data.Ex.values[..., 0]
    ey = field_data.Ey.values[..., 0]
    ez = field_data.Ez.values[..., 0]
    powers = np.array([np.sum(np.abs(field) ** 2) for field in (ex, ey, ez)])
    power_fractions = powers / np.sum(powers)

    expected_fractions = _thin_lens_uniform_pupil_power_fractions(na)
    tolerances = np.array([3e-3, 3e-4, 3e-3])
    assert np.all(np.abs(power_fractions - expected_fractions) < tolerances)
    assert np.max(np.abs(field_data.Hy.values)) > 0


def test_thin_lens_profile_spectrum_obeys_vector_diffraction_identities():
    """Test the corrected Mansuripur/Richards-Wolf polarization and magnetic spectra."""
    wvl0_um = 1.55
    freq0 = td.C_0 / wvl0_um
    na = 0.7
    num_plane_waves = 13
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=[freq0],
        numerical_aperture=na,
        resolution=11,
        num_plane_waves=num_plane_waves,
    )

    sigma_x, sigma_y, k_hat_z, e_spectrum, h_spectrum = beam._angular_spectrum_components(
        freq=freq0, background_n=1.0
    )
    sigma_z = k_hat_z
    denom = 1 + sigma_z
    sample_area = (2 * na / num_plane_waves) ** 2
    expected_e_spectrum = (
        np.stack(
            (
                1 - sigma_x**2 / denom,
                -sigma_x * sigma_y / denom,
                -sigma_x,
            ),
            axis=0,
        )
        / np.sqrt(sigma_z)
        * sample_area
    )
    expected_h_spectrum = (
        np.stack(
            (
                sigma_y * expected_e_spectrum[2] - sigma_z * expected_e_spectrum[1],
                sigma_z * expected_e_spectrum[0] - sigma_x * expected_e_spectrum[2],
                sigma_x * expected_e_spectrum[1] - sigma_y * expected_e_spectrum[0],
            ),
            axis=0,
        )
        / td.ETA_0
    )

    np.testing.assert_allclose(e_spectrum, expected_e_spectrum, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h_spectrum, expected_h_spectrum, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        sigma_x * e_spectrum[0] + sigma_y * e_spectrum[1] + k_hat_z * e_spectrum[2],
        0,
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("na", [0.3, 0.5, 0.966])
def test_thin_lens_profile_spectrum_power_fractions_across_na(na):
    """Test angular-spectrum component fractions against closed-form pupil integrals."""
    freq0 = td.C_0 / 1.55
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=[freq0],
        numerical_aperture=na,
        resolution=11,
        num_plane_waves=151,
    )

    _, _, _, e_spectrum, _ = beam._angular_spectrum_components(freq=freq0, background_n=1.0)
    power_fractions = np.sum(np.abs(e_spectrum) ** 2, axis=1)
    power_fractions = power_fractions / np.sum(power_fractions)
    expected_fractions = _thin_lens_uniform_pupil_power_fractions(na)

    np.testing.assert_allclose(power_fractions, expected_fractions, rtol=0, atol=5e-4)


def test_thin_lens_profile_matches_richards_wolf_focal_line_cuts():
    """Test focal line cuts against independent Richards-Wolf Bessel-integral quadrature."""
    wvl0_um = 1.55
    freq0 = td.C_0 / wvl0_um
    na = 0.8
    k0 = 2 * np.pi / wvl0_um
    line_u = np.linspace(-4, 4, 33)
    diagonal_u = np.linspace(0, 4, 21)
    points = np.concatenate(
        (
            np.stack((line_u / k0, np.zeros_like(line_u), np.zeros_like(line_u))),
            np.stack((np.zeros_like(line_u), line_u / k0, np.zeros_like(line_u))),
            np.stack(
                (
                    diagonal_u / (k0 * np.sqrt(2)),
                    diagonal_u / (k0 * np.sqrt(2)),
                    np.zeros_like(diagonal_u),
                )
            ),
        ),
        axis=1,
    )

    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=[freq0],
        numerical_aperture=na,
        resolution=11,
        num_plane_waves=151,
    )

    actual = beam._thin_lens_fields_propagation_frame(points, background_n=np.array([1.0]))[0][
        :, :, 0
    ]
    expected = _richards_wolf_x_polarized_focal_field(points, na=na, wavelength=wvl0_um)
    focus_index = len(line_u) // 2
    actual = actual / actual[0, focus_index]
    expected = expected / expected[0, focus_index]

    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-4)


@pytest.mark.parametrize(
    ("na", "expected_peak_ratios", "tolerances"),
    [
        (0.5, np.array([100, 0.022, 3.47]), np.array([1e-12, 0.004, 0.15])),
        (0.966, np.array([100, 0.78, 19.1]), np.array([1e-12, 0.08, 0.7])),
    ],
)
def test_thin_lens_profile_mansuripur_1993_peak_intensity_ratios(
    na, expected_peak_ratios, tolerances
):
    """Test focal peak ratios from the corrected Mansuripur aplanatic-lens erratum."""
    wvl0_um = 1.55
    freq0 = td.C_0 / wvl0_um
    radius_max = 3 * wvl0_um / na
    coords = np.linspace(-radius_max, radius_max, 61)
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(2 * radius_max, 2 * radius_max, 0),
        freqs=[freq0],
        numerical_aperture=na,
        resolution=61,
        num_plane_waves=101,
    )

    e_field = beam.analytic_beam(x=coords, y=coords, z=np.array([0.0]), background_n=1.0, field="E")
    peak_intensities = np.array(
        [np.max(np.abs(e_field[axis, :, :, 0, 0]) ** 2) for axis in range(3)]
    )
    peak_ratios = peak_intensities / peak_intensities[0] * 100

    assert np.all(np.abs(peak_ratios - expected_peak_ratios) < tolerances)


def test_thin_lens_profile_low_na_matches_airy_pattern():
    """Test the low-NA filled-pupil focal profile against the scalar Airy limit."""
    wvl0_um = 1.55
    freq0 = td.C_0 / wvl0_um
    na = 0.05
    radius = np.linspace(0, 0.58 * wvl0_um / na, 41)
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=[freq0],
        numerical_aperture=na,
        resolution=11,
        num_plane_waves=151,
    )

    e_field = beam.analytic_beam(
        x=radius, y=np.array([0.0]), z=np.array([0.0]), background_n=1.0, field="E"
    )
    ex_profile = e_field[0, :, 0, 0, 0]
    intensity = np.abs(ex_profile / ex_profile[0]) ** 2
    expected_intensity = _airy_disk_intensity(radius, na, wvl0_um)

    np.testing.assert_allclose(intensity, expected_intensity, rtol=0, atol=1e-3)
    assert np.max(np.abs(e_field[2, :, 0, 0, 0] / ex_profile[0]) ** 2) < 5e-4


def test_thin_lens_profile_lens_offset_shifts_focus():
    """Test that lens_offset translates the focal field in the tangential plane."""
    wvl0_um = 1.55
    freq0 = td.C_0 / wvl0_um
    lens_offset = (0.25, -0.15)
    coords = np.linspace(-1.0, 1.0, 81)

    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=[freq0],
        numerical_aperture=0.5,
        resolution=11,
        num_plane_waves=151,
        lens_offset=lens_offset,
    )
    reference = beam.updated_copy(lens_offset=(0.0, 0.0))

    e_shifted = beam.analytic_beam(
        x=coords, y=coords, z=np.array([0.0]), background_n=1.0, field="E"
    )
    e_reference = reference.analytic_beam(
        x=coords, y=coords, z=np.array([0.0]), background_n=1.0, field="E"
    )
    shifted_intensity = np.sum(np.abs(e_shifted[:, :, :, 0, 0]) ** 2, axis=0)
    reference_intensity = np.sum(np.abs(e_reference[:, :, :, 0, 0]) ** 2, axis=0)

    peak_index = np.unravel_index(np.argmax(shifted_intensity), shifted_intensity.shape)
    assert coords[peak_index[0]] == pytest.approx(lens_offset[0], abs=coords[1] - coords[0])
    assert coords[peak_index[1]] == pytest.approx(lens_offset[1], abs=coords[1] - coords[0])

    offset_index = tuple(np.where(np.isclose(coords, offset))[0][0] for offset in lens_offset)
    zero_index = tuple(np.where(np.isclose(coords, 0.0))[0][0] for _ in lens_offset)
    assert shifted_intensity[offset_index] == pytest.approx(
        reference_intensity[zero_index], rel=5e-4
    )


@pytest.mark.parametrize(
    "normal_axis",
    [0, 1, 2],
)
def test_thin_lens_profile_angled_lens_offset_uses_component_frame(normal_axis):
    """Test lens_offset uses component-frame tangential axes for angled beams."""
    freq0 = td.C_0 / 1.55
    lens_offset = (0.18, -0.12)
    coords = [
        np.array([-0.21, 0.06, 0.32]),
        np.array([-0.17, 0.04, 0.23]),
        np.array([-0.19, 0.02, 0.27]),
    ]
    coords[normal_axis] = np.array([0.0])
    size = [10, 10, 10]
    size[normal_axis] = 0
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=tuple(size),
        freqs=[freq0],
        numerical_aperture=0.55,
        resolution=11,
        num_plane_waves=41,
        lens_offset=lens_offset,
        angle_theta=0.37,
        angle_phi=0.58,
    )
    reference = beam.updated_copy(lens_offset=(0.0, 0.0))

    expected_coords = [coord.copy() for coord in coords]
    tangent_axes = [axis for axis in range(3) if axis != normal_axis]
    for offset, axis in zip(lens_offset, tangent_axes):
        expected_coords[axis] = expected_coords[axis] - offset

    shifted = beam.analytic_beam(x=coords[0], y=coords[1], z=coords[2], background_n=1.0, field="E")
    expected = reference.analytic_beam(
        x=expected_coords[0],
        y=expected_coords[1],
        z=expected_coords[2],
        background_n=1.0,
        field="E",
    )

    np.testing.assert_allclose(shifted, expected, rtol=1e-12, atol=1e-12)


def test_thin_lens_profile_angled_waist_distance_uses_propagation_frame():
    """Test angled waist_distance is measured along the rotated propagation axis."""
    freq0 = td.C_0 / 1.55
    waist_distance = 0.37
    x = np.array([-0.18, 0.07])
    y = np.array([-0.13, 0.21])
    z = np.array([0.04, 0.16])
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=[freq0],
        numerical_aperture=0.5,
        resolution=11,
        num_plane_waves=31,
        angle_theta=0.41,
        angle_phi=0.62,
        waist_distance=waist_distance,
    )
    reference = beam.updated_copy(waist_distance=0.0)
    component_shift = beam._inverse_rotate_field_vals_z(
        np.array([[0.0], [0.0], [waist_distance]]),
        background_n=1.0,
    ).reshape(3)

    actual = beam.analytic_beam(x=x, y=y, z=z, background_n=1.0, field="E")
    expected = reference.analytic_beam(
        x=x + component_shift[0],
        y=y + component_shift[1],
        z=z + component_shift[2],
        background_n=1.0,
        field="E",
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_thin_lens_profile_singleton_background_n_broadcasts_over_freqs():
    """Test scalar-like background_n arrays are broadcast over all thin-lens frequencies."""
    freqs = [td.C_0 / 1.55, td.C_0 / 1.31]
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=freqs,
        numerical_aperture=0.4,
        resolution=11,
        num_plane_waves=21,
    )
    x = np.array([-0.2, 0.1])
    y = np.array([-0.15, 0.25])
    z = np.array([0.0])

    tensor_scalar = beam.analytic_beam(x=x, y=y, z=z, background_n=1.0, field="E")
    tensor_singleton = beam.analytic_beam(x=x, y=y, z=z, background_n=np.array([1.0]), field="E")

    np.testing.assert_allclose(tensor_singleton, tensor_scalar, rtol=1e-14, atol=1e-14)
    assert np.max(np.abs(tensor_singleton[..., 1])) > 0

    points = np.array(
        [
            [-0.1, 0.0, 0.2],
            [0.05, -0.15, 0.1],
            [0.0, 0.0, 0.0],
        ]
    )
    arbitrary_scalar = beam._thin_lens_fields_propagation_frame(
        points, background_n=1.0, field_kinds=("E",)
    )[0]
    arbitrary_singleton = beam._thin_lens_fields_propagation_frame(
        points, background_n=[1.0], field_kinds=("E",)
    )[0]

    np.testing.assert_allclose(arbitrary_singleton, arbitrary_scalar, rtol=1e-14, atol=1e-14)
    assert np.max(np.abs(arbitrary_singleton[..., 1])) > 0

    with pytest.raises(ValueError, match="background_n"):
        beam._thin_lens_fields_propagation_frame(points, background_n=[1.0, 1.1, 1.2])


def test_thin_lens_profile_pol_angle_rotates_components():
    """Test that polarization angle rotates the incident lens polarization."""
    wvl0_um = 1.55
    freq0 = td.C_0 / wvl0_um
    size_val = 20 * wvl0_um

    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(size_val, size_val, 0),
        freqs=[freq0],
        numerical_aperture=0.5,
        resolution=41,
        num_plane_waves=41,
        pol_angle=np.pi / 2,
    )
    field_data = beam.field_data

    ex = field_data.Ex.values[..., 0]
    ey = field_data.Ey.values[..., 0]
    ez = field_data.Ez.values[..., 0]
    powers = np.array([np.sum(np.abs(field) ** 2) for field in (ex, ey, ez)])
    frac_x, frac_y, frac_z = powers / np.sum(powers)

    assert frac_x < 0.003
    assert frac_y == pytest.approx(0.94, abs=0.02)
    assert frac_z == pytest.approx(0.06, abs=0.02)


@pytest.mark.parametrize(
    ("size", "expected_shape"),
    [
        ((0, 10, 10), (1, 20, 20, 1)),
        ((10, 0, 10), (20, 1, 20, 1)),
        ((10, 10, 0), (20, 20, 1, 1)),
    ],
)
def test_thin_lens_profile_normal_axes(size, expected_shape):
    """Test thin-lens coordinate remapping for all normal axes."""
    freq0 = td.C_0 / 1.55
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=size,
        freqs=[freq0],
        numerical_aperture=0.5,
        resolution=21,
        num_plane_waves=(21, 23),
    )

    field_data = beam.field_data
    assert field_data.Ex.shape == expected_shape
    assert np.allclose(field_data.flux, 1)


def test_thin_lens_profile_backward_direction_and_multifrequency():
    """Test backward propagation sign and stable multi-frequency output grid."""
    freqs = [td.C_0 / 1.55, td.C_0 / 1.3]
    beam = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=freqs,
        numerical_aperture=0.5,
        direction="-",
        resolution=21,
        num_plane_waves=21,
    )

    field_data = beam.field_data
    assert field_data.Ex.shape == (20, 20, 1, 2)
    assert np.allclose(field_data.flux, -1)
    assert field_data.Ex.coords["x"].size == 20
    assert field_data.Ex.coords["f"].size == 2


def test_thin_lens_profile_validation():
    """Test validation for thin-lens coupled settings."""
    freq0 = td.C_0 / 1.55
    kwargs = {"center": (0, 0, 0), "size": (10, 10, 0), "freqs": [freq0], "numerical_aperture": 0.5}

    with pytest.raises(ValidationError):
        ThinLensProfile(**kwargs, num_plane_waves=2)

    with pytest.raises(ValidationError, match="must not exceed"):
        ThinLensProfile(**kwargs, num_plane_waves=1001)

    with pytest.raises(ValidationError, match="must not exceed"):
        ThinLensProfile(**kwargs, num_plane_waves=(1000, 1001))

    with pytest.raises(ValidationError) as excinfo:
        ThinLensProfile(**kwargs, fill_lens=False)
    assert_single_value_error_loc(excinfo, ("lens_diameter",), "required")

    with pytest.raises(ValidationError) as excinfo:
        ThinLensProfile(**kwargs, fill_lens=False, lens_diameter=2, beam_diameter=3)
    assert_single_value_error_loc(excinfo, ("beam_diameter",), "must not exceed")

    with pytest.raises(ValidationError):
        ThinLensProfile(**{**kwargs, "numerical_aperture": 0})


def test_thin_lens_profile_immersion_numerical_aperture():
    """Test numerical aperture is bounded by the medium index, not by vacuum."""
    freq0 = td.C_0 / 1.55
    profile = ThinLensProfile(
        center=(0, 0, 0),
        size=(10, 10, 0),
        freqs=[freq0],
        numerical_aperture=1.2,
        num_plane_waves=11,
    )

    sigma_x, sigma_y, *_ = profile._angular_spectrum_grid(background_n=1.5)
    assert np.max(np.sqrt(sigma_x**2 + sigma_y**2)) < 1

    with pytest.raises(ValueError, match="numerical_aperture"):
        profile._angular_spectrum_grid(background_n=1.0)
    with pytest.raises(ValueError, match="less than"):
        profile._angular_spectrum_grid(background_n=1.2)

    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    td.ThinLensBeam(
        center=(0, 0, 0),
        size=(0, 4, 4),
        source_time=source_time,
        direction="+",
        numerical_aperture=1.2,
    )
    td.ThinLensOverlapMonitor(
        center=(0, 0, 0),
        size=(0, 4, 4),
        freqs=[freq0],
        name="thin_lens",
        numerical_aperture=1.2,
    )


def test_thin_lens_source_and_overlap_monitor_api():
    """Test source and overlap monitor APIs backed by thin-lens profiles."""
    freq0 = td.C_0 / 1.55
    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    source = td.ThinLensBeam(
        center=(-1, 0, 0),
        size=(0, 4, 4),
        source_time=source_time,
        direction="+",
        numerical_aperture=0.4,
        waist_distance=-1,
        fill_lens=False,
        lens_diameter=4,
        beam_diameter=2,
        num_plane_waves=11,
    )
    monitor = td.ThinLensOverlapMonitor(
        center=(0, 0, 0),
        size=(0, 4, 4),
        freqs=[freq0],
        name="thin_lens",
        numerical_aperture=source.numerical_aperture,
        waist_distance=0,
        fill_lens=source.fill_lens,
        lens_diameter=source.lens_diameter,
        beam_diameter=source.beam_diameter,
        num_plane_waves=source.num_plane_waves,
    )
    sim = td.Simulation(
        size=(3, 4, 4),
        grid_spec=td.GridSpec.uniform(dl=0.2),
        run_time=1e-12,
        sources=[source],
        monitors=[monitor],
    )

    assert sim.sources[0].type == "ThinLensBeam"
    assert sim.monitors[0].type == "ThinLensOverlapMonitor"

    kwargs = {
        "center": (0, 0, 0),
        "size": (0, 4, 4),
        "source_time": source_time,
        "direction": "+",
        "numerical_aperture": 0.4,
    }
    with pytest.raises(ValidationError):
        td.ThinLensBeam(**kwargs, num_plane_waves=2)
    with pytest.raises(ValidationError, match="must not exceed"):
        td.ThinLensBeam(**kwargs, num_plane_waves=1001)
    with pytest.raises(ValidationError) as excinfo:
        td.ThinLensBeam(**kwargs, fill_lens=False)
    assert_single_value_error_loc(excinfo, ("lens_diameter",), "required")

    monitor_kwargs = {
        "center": (0, 0, 0),
        "size": (0, 4, 4),
        "freqs": [freq0],
        "name": "thin_lens_validation",
        "numerical_aperture": 0.4,
    }
    with pytest.raises(ValidationError) as excinfo:
        td.ThinLensOverlapMonitor(**monitor_kwargs, fill_lens=False)
    assert_single_value_error_loc(excinfo, ("lens_diameter",), "required")

    with pytest.raises(ValidationError, match="must not exceed"):
        td.ThinLensOverlapMonitor(**monitor_kwargs, num_plane_waves=1001)

    with pytest.raises(ValidationError) as excinfo:
        td.ThinLensOverlapMonitor(
            **monitor_kwargs,
            fill_lens=False,
            lens_diameter=2,
            beam_diameter=3,
        )
    assert_single_value_error_loc(excinfo, ("beam_diameter",), "must not exceed")


def test_thin_lens_source_compute_beam_fields_honors_normalize():
    """Test that ThinLensBeam honors source-hook flux normalization."""
    freq0 = td.C_0 / 1.55
    source = td.ThinLensBeam(
        center=(0, 0, 0),
        size=(0, 4, 4),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
        direction="+",
        numerical_aperture=0.4,
        num_plane_waves=15,
    )
    x = np.array([0.0])
    y = np.linspace(-2, 2, 41)
    z = np.linspace(-2, 2, 43)
    grid = BeamGrid(
        x=x,
        y=y,
        z=z,
        freqs=np.array([freq0]),
        background_n=np.array([1.0]),
    )
    pose = BeamPose(
        injection_axis=0,
        direction=source.direction,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        center=source.center,
    )

    unnormalized = source.compute_beam_fields_on_grid(pose=pose, grid=grid, normalize=False)
    normalized = source.compute_beam_fields_on_grid(pose=pose, grid=grid, normalize=True)

    flux_raw = _beam_plane_flux(unnormalized, x=x, y=y, z=z, normal_axis=0)
    flux_normalized = _beam_plane_flux(normalized, x=x, y=y, z=z, normal_axis=0)

    assert np.abs(flux_raw[0] - 1) > 1e-2
    np.testing.assert_allclose(flux_normalized, 1.0, rtol=0, atol=1e-12)
    assert not np.allclose(unnormalized["Ey"], normalized["Ey"])


def test_thin_lens_setup_work_validation(monkeypatch):
    """Test simulation-level thin-lens setup work guard."""
    monkeypatch.setattr(simulation_module, "MAX_THIN_LENS_SETUP_WORK_UNITS", 1)

    freq0 = td.C_0 / 1.55
    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    source = td.ThinLensBeam(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        source_time=source_time,
        direction="+",
        numerical_aperture=0.4,
        num_plane_waves=3,
    )
    sim = td.Simulation(
        size=(1, 1, 0.2),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-14,
        sources=[source],
    )
    assert (
        sim._thin_lens_setup_work_units(
            plane_cells=2,
            num_plane_waves=3,
            num_freqs=5,
            num_evaluations=simulation_module.THIN_LENS_SOURCE_SETUP_EVALUATIONS,
        )
        == 2 * 9 * 5 * simulation_module.THIN_LENS_SOURCE_SETUP_EVALUATIONS
    )
    assert (
        sim._thin_lens_setup_work_limit(
            num_evaluations=simulation_module.THIN_LENS_SOURCE_SETUP_EVALUATIONS
        )
        == simulation_module.THIN_LENS_SOURCE_SETUP_EVALUATIONS
    )
    with pytest.raises(ValidationError) as excinfo:
        sim.validate_pre_upload()
    assert_single_value_error_loc(excinfo, ("sources", 0), "estimated setup work units")

    monitor = td.ThinLensOverlapMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[freq0],
        name="thin_lens_work_guard",
        numerical_aperture=0.4,
        num_plane_waves=3,
    )
    assert (
        sim._thin_lens_monitor_setup_evaluations(monitor)
        == simulation_module.THIN_LENS_MONITOR_SETUP_EVALUATIONS
    )
    assert sim._thin_lens_monitor_setup_evaluations(monitor.updated_copy(colocate=False)) == (
        simulation_module.THIN_LENS_MONITOR_SETUP_EVALUATIONS
        * simulation_module.THIN_LENS_FIELD_COMPONENTS
    )
    sim = sim.updated_copy(sources=[], monitors=[monitor])
    with pytest.raises(ValidationError) as excinfo:
        sim.validate_pre_upload(source_required=False)
    assert_single_value_error_loc(excinfo, ("monitors", 0), "estimated setup work units")


def test_gaussian_like_beam_background_validation():
    """Test loc-aware simulation validation for Gaussian-like beam background assumptions."""
    freq0 = td.C_0 / 1.55
    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    source = td.ThinLensBeam(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        source_time=source_time,
        direction="+",
        numerical_aperture=1.0,
        num_plane_waves=3,
    )
    sim = td.Simulation(
        size=(1, 1, 0.2),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-14,
        sources=[source],
    )
    with pytest.raises(ValidationError) as excinfo:
        sim.validate_pre_upload()
    assert_single_value_error_loc(excinfo, ("sources", 0, "numerical_aperture"), "must be less")

    thin_lens_monitor = td.ThinLensOverlapMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[freq0],
        name="thin_lens_na",
        numerical_aperture=1.0,
        num_plane_waves=3,
    )
    sim = sim.updated_copy(sources=[], monitors=[thin_lens_monitor])
    with pytest.raises(ValidationError) as excinfo:
        sim.validate_pre_upload(source_required=False)
    assert_single_value_error_loc(excinfo, ("monitors", 0, "numerical_aperture"), "must be less")

    monitor = td.GaussianOverlapMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[freq0],
        name="gaussian_interface",
    )
    structure = td.Structure(
        geometry=td.Box(center=(0.25, 0, 0), size=(0.5, 1, 0.2)),
        medium=td.Medium(permittivity=2.0),
    )
    sim = sim.updated_copy(
        sources=[],
        monitors=[monitor],
        structures=[structure],
    )
    with pytest.raises(ValidationError) as excinfo:
        sim.validate_pre_upload(source_required=False)
    assert_single_value_error_loc(excinfo, ("monitors", 0), "different mediums")


def test_thin_lens_source_and_monitor_serialization_roundtrip(tmp_path):
    """Test Simulation JSON/HDF5 roundtrip for thin-lens source and monitor fields."""
    freq0 = td.C_0 / 1.55
    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    source = td.ThinLensBeam(
        center=(-1, 0.1, -0.2),
        size=(0, 4, 5),
        source_time=source_time,
        direction="+",
        numerical_aperture=0.4,
        waist_distance=-0.8,
        fill_lens=False,
        lens_diameter=4.0,
        beam_diameter=2.0,
        num_plane_waves=(13, 15),
        lens_offset=(0.12, -0.08),
        angle_theta=0.1,
        angle_phi=0.3,
        pol_angle=0.4,
        num_freqs=5,
    )
    monitor = td.ThinLensOverlapMonitor(
        center=(-0.4, 0.15, -0.18),
        size=(0, 4, 5),
        freqs=[freq0, freq0 * 1.05],
        name="thin_lens",
        numerical_aperture=source.numerical_aperture,
        waist_distance=-0.2,
        fill_lens=source.fill_lens,
        lens_diameter=source.lens_diameter,
        beam_diameter=source.beam_diameter,
        num_plane_waves=source.num_plane_waves,
        lens_offset=source.lens_offset,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        store_fields_direction="+",
    )
    sim = td.Simulation(
        size=(3, 4, 5),
        grid_spec=td.GridSpec.uniform(dl=0.2),
        run_time=1e-12,
        sources=[source],
        monitors=[monitor],
    )

    json_path = tmp_path / "thin_lens_sim.json"
    hdf5_path = tmp_path / "thin_lens_sim.hdf5"
    sim.to_file(json_path)
    sim.to_hdf5(hdf5_path)

    for loaded in (td.Simulation.from_file(json_path), td.Simulation.from_hdf5(hdf5_path)):
        loaded_source = loaded.sources[0]
        loaded_monitor = loaded.monitors[0]
        assert isinstance(loaded_source, td.ThinLensBeam)
        assert isinstance(loaded_monitor, td.ThinLensOverlapMonitor)
        for component in (loaded_source, loaded_monitor):
            assert component.numerical_aperture == source.numerical_aperture
            assert component.fill_lens is False
            assert component.lens_diameter == source.lens_diameter
            assert component.beam_diameter == source.beam_diameter
            assert component.num_plane_waves == source.num_plane_waves
            assert component.lens_offset == source.lens_offset
            assert component.angle_theta == source.angle_theta
            assert component.angle_phi == source.angle_phi
            assert component.pol_angle == source.pol_angle
        assert loaded_source.waist_distance == source.waist_distance
        assert loaded_source.num_freqs == source.num_freqs
        assert loaded_monitor.waist_distance == monitor.waist_distance
        assert loaded_monitor.store_fields_direction == "+"


def test_thin_lens_overlap_monitor_adjoint_source_factory_is_deferred():
    """Test thin-lens overlap monitors reject adjoint source construction for now."""
    from tidy3d.components.autograd.source_factory import gaussian_source_from_monitor

    freq0 = td.C_0 / 1.55
    monitor = td.ThinLensOverlapMonitor(
        center=(0.1, -0.2, 0.3),
        size=(0, 4, 5),
        freqs=[freq0],
        name="thin_lens",
        numerical_aperture=0.42,
        waist_distance=-0.35,
        fill_lens=False,
        lens_diameter=4.0,
        beam_diameter=1.8,
        num_plane_waves=(13, 15),
        lens_offset=(0.12, -0.08),
        angle_theta=0.11,
        angle_phi=-0.25,
        pol_angle=0.4,
    )

    with pytest.raises(NotImplementedError, match="ThinLensOverlapMonitor"):
        gaussian_source_from_monitor(
            monitor=monitor,
            freq=freq0,
            direction="+",
            coefficient=2.0 - 0.5j,
            fwidth=freq0 / 10,
        )


if __name__ == "__main__":
    # test_gaussian_beam()
    # test_plane_wave()
    # test_astigmatic_gaussian_beam()
    # test_invalid_beam_size()
    test_gaussian_beam_ex_spread_waist_distance()
    # test_gaussian_beam_center()
