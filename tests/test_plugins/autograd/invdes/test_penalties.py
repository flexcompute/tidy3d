from __future__ import annotations

import autograd.numpy as np
import pytest

from tidy3d.plugins.autograd.invdes.penalties import make_erosion_dilation_penalty
from tidy3d.plugins.autograd.invdes.symmetries import expand_mirror_symmetry
from tidy3d.plugins.autograd.types import PaddingType


@pytest.mark.parametrize("radius", [1, 2, (1, 2)])
@pytest.mark.parametrize("dl", [0.1, 0.2, (0.1, 0.2)])
@pytest.mark.parametrize("size_px", [None, 5, (5, 7)])
@pytest.mark.parametrize("padding", PaddingType.__args__)
def test_make_erosion_dilation_penalty(rng, radius, dl, size_px, padding):
    """Test make_erosion_dilation_penalty function for various parameters."""
    erosion_dilation_penalty_func = make_erosion_dilation_penalty(
        radius=radius,
        dl=dl,
        size_px=size_px,
        beta=10,
        eta=0.5,
        delta_eta=0.01,
        padding=padding,
    )
    array = rng.random((51, 51))
    result = erosion_dilation_penalty_func(array)
    assert isinstance(result, float)
    assert result >= 0


@pytest.mark.parametrize("symmetry", [("low", None), (None, "high"), ("low", "high")])
def test_make_erosion_dilation_penalty_mirror_symmetry_matches_explicit_expansion(symmetry):
    """Mirror-aware penalties should match explicit mirroring of the domain."""
    array = np.linspace(0.0, 1.0, 35).reshape((5, 7))

    penalty_func = make_erosion_dilation_penalty(
        radius=1,
        dl=0.2,
        beta=10.0,
        eta=0.5,
        delta_eta=0.01,
        symmetry=symmetry,
    )
    baseline_func = make_erosion_dilation_penalty(
        radius=1,
        dl=0.2,
        beta=10.0,
        eta=0.5,
        delta_eta=0.01,
    )

    expanded, _ = expand_mirror_symmetry(array, symmetry=symmetry)
    expected = baseline_func(expanded)
    result = penalty_func(array)

    assert result == pytest.approx(expected)
