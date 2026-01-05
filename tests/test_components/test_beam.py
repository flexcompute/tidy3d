"""Tests for the various BeamProfile components."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tidy3d.components.beam import (
    AstigmaticGaussianBeamProfile,
    GaussianBeamProfile,
    PlaneWaveBeamProfile,
)
from tidy3d.components.data.monitor_data import FieldData

from ..utils import AssertLogLevel

FREQS = np.linspace(1e14, 2e14, 10).tolist()


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


if __name__ == "__main__":
    # test_gaussian_beam()
    # test_plane_wave()
    # test_astigmatic_gaussian_beam()
    # test_invalid_beam_size()
    test_gaussian_beam_ex_spread_waist_distance()
    # test_gaussian_beam_center()
