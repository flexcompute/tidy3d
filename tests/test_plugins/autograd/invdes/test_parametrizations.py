from __future__ import annotations

import autograd.numpy as np
import numpy.testing as npt
import pytest

from tidy3d.plugins.autograd.invdes.parametrizations import make_filter_and_project
from tidy3d.plugins.autograd.invdes.symmetries import expand_mirror_symmetry
from tidy3d.plugins.autograd.types import PaddingType


@pytest.mark.parametrize("radius", [1, 2, (1, 2)])
@pytest.mark.parametrize("dl", [0.1, 0.2, (0.1, 0.2)])
@pytest.mark.parametrize("size_px", [None, 5, (5, 7)])
@pytest.mark.parametrize("beta", [1.0, 10.0])
@pytest.mark.parametrize("filter_type", ["circular", "conic", "gaussian"])
@pytest.mark.parametrize("padding", PaddingType.__args__)
@pytest.mark.parametrize("init_type", ("random", "binary_low_border", "binary_high_border"))
def test_make_filter_and_project(rng, radius, dl, size_px, beta, filter_type, padding, init_type):
    """Test make_filter_and_project function for various parameters."""
    filter_and_project_func = make_filter_and_project(
        radius=radius,
        dl=dl,
        size_px=size_px,
        beta=beta,
        eta=0.5,
        filter_type=filter_type,
        padding=padding,
    )
    if init_type == "random":
        array = rng.random((51, 51))
    elif init_type == "binary_low_border":
        array = np.zeros((100, 100))
        array[40:60, 40:60] = 1.0
    else:
        array = np.ones((100, 100))
        array[40:60, 40:60] = 0.0
    result = filter_and_project_func(array)
    assert result.shape == array.shape
    assert np.all(result >= 0) and np.all(result <= 1)


@pytest.mark.parametrize("symmetry", [("low", None), (None, "high"), ("low", "high")])
def test_make_filter_and_project_mirror_symmetry_matches_explicit_expansion(symmetry):
    """Mirror-aware filtering should match filtering an explicitly mirrored domain."""
    array = np.linspace(0.0, 1.0, 35).reshape((5, 7))

    filter_and_project_func = make_filter_and_project(
        radius=1,
        dl=0.2,
        beta=10.0,
        eta=0.5,
        symmetry=symmetry,
    )
    baseline_func = make_filter_and_project(radius=1, dl=0.2, beta=10.0, eta=0.5)

    expanded, crop_slices = expand_mirror_symmetry(array, symmetry=symmetry)
    expected = baseline_func(expanded)[crop_slices]
    result = filter_and_project_func(array)

    npt.assert_allclose(result, expected)
