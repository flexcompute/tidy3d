from __future__ import annotations

import numpy as np
import pytest

from tidy3d.plugins.autograd.invdes.filters import (
    _get_kernel_size,
    make_circular_filter,
    make_conic_filter,
    make_filter,
    make_gaussian_filter,
)
from tidy3d.plugins.autograd.types import PaddingType


@pytest.mark.parametrize(
    "radius, dl, size_px, expected",
    [
        (1, 0.1, None, (21,)),
        (1, [0.1, 0.2], None, (21, 11)),
        ([1, 2], 0.1, None, (21, 41)),
        ([1, 1], [0.1, 0.2], None, (21, 11)),
        ([1, 2], [0.1, 0.1], None, (21, 41)),
        (None, None, 5, (5,)),
        (None, None, (5, 7), (5, 7)),
    ],
)
def test_get_kernel_size(radius, dl, size_px, expected):
    result = _get_kernel_size(radius, dl, size_px)
    assert result == expected


def test_get_kernel_size_invalid_arguments():
    with pytest.raises(
        ValueError, match="Either 'size_px' or both 'radius' and 'dl' must be provided."
    ):
        _get_kernel_size(None, None, None)


@pytest.mark.parametrize("radius", [1, 2, (1, 2)])
@pytest.mark.parametrize("dl", [0.1, 0.2, (0.1, 0.2)])
@pytest.mark.parametrize("size_px", [None, 5, (5, 7)])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("padding", PaddingType.__args__)
class TestMakeFilter:
    @pytest.mark.parametrize("filter_type", ["circular", "conic", "gaussian"])
    def test_make_filter(self, rng, filter_type, radius, dl, size_px, normalize, padding):
        """Test make_filter function for various parameters."""
        filter_func = make_filter(
            radius=radius,
            dl=dl,
            size_px=size_px,
            normalize=normalize,
            padding=padding,
            filter_type=filter_type,
        )
        array = rng.random((51, 51))
        result = filter_func(array)
        assert result.shape == array.shape

    def test_make_circular_filter(self, rng, radius, dl, size_px, normalize, padding):
        """Test make_circular_filter function for various parameters."""
        filter_func = make_circular_filter(
            radius=radius,
            dl=dl,
            size_px=size_px,
            normalize=normalize,
            padding=padding,
        )
        array = rng.random((51, 51))
        result = filter_func(array)
        assert result.shape == array.shape

    def test_make_conic_filter(self, rng, radius, dl, size_px, normalize, padding):
        """Test make_conic_filter function for various parameters."""
        filter_func = make_conic_filter(
            radius=radius,
            dl=dl,
            size_px=size_px,
            normalize=normalize,
            padding=padding,
        )
        array = rng.random((51, 51))
        result = filter_func(array)
        assert result.shape == array.shape

    def test_make_gaussian_filter(self, rng, radius, dl, size_px, normalize, padding):
        """Test make_gaussian_filter function for various parameters."""
        filter_func = make_gaussian_filter(
            radius=radius,
            dl=dl,
            size_px=size_px,
            normalize=normalize,
            padding=padding,
        )
        array = rng.random((51, 51))
        result = filter_func(array)
        assert result.shape == array.shape


def _conic_filter_reference(values: np.ndarray, coords: np.ndarray, radius: float) -> np.ndarray:
    diff = coords[1:] - coords[:-1]
    diff_left = np.pad(diff, (1, 0), mode="edge")
    diff_right = np.pad(diff, (0, 1), mode="edge")
    cell_sizes = 0.5 * (diff_left + diff_right)

    output = np.zeros_like(values, dtype=float)
    for i, coord in enumerate(coords):
        dist = np.abs(coords - coord)
        weights = np.maximum(0.0, 1.0 - dist / radius)
        weights = weights * cell_sizes
        output[i] = np.sum(weights * values)
    return output


def test_conic_filter_coords_nonuniform():
    coords = np.array([0.0, 0.12, 0.3, 0.55, 0.9, 1.4])
    values = np.array([0.1, 0.8, 0.2, 0.9, 0.4, 0.7])
    radius = 0.35

    filter_func = make_conic_filter(
        radius=radius, coords=(coords,), normalize=False, padding="constant"
    )
    result = filter_func(values)
    expected = _conic_filter_reference(values, coords, radius)

    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_make_filter_coords_size_px_conflict():
    coords = (np.linspace(0.0, 1.0, 5),)
    with pytest.raises(ValueError, match="coords"):
        make_filter(radius=0.2, coords=coords, size_px=5, filter_type="conic")


def test_make_filter_coords_dl_conflict():
    coords = (np.linspace(0.0, 1.0, 5),)
    with pytest.raises(ValueError, match="coords"):
        make_filter(radius=0.2, coords=coords, dl=0.1, filter_type="conic")
