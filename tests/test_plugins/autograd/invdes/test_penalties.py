from __future__ import annotations

import numpy as np
import pytest

from tidy3d.plugins.autograd.invdes.penalties import make_erosion_dilation_penalty
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


def test_make_erosion_dilation_penalty_coords_anisotropic(rng):
    dx, dy = 0.15, 0.05
    x = np.arange(25) * dx
    y = np.arange(15) * dy
    array = rng.random((x.size, y.size))

    penalty_func = make_erosion_dilation_penalty(radius=0.25, coords=(x, y))
    result = penalty_func(array)

    assert isinstance(result, float)
    assert result >= 0


def test_make_erosion_dilation_penalty_coords_size_px_conflict():
    coords = (np.linspace(0.0, 1.0, 5),)
    with pytest.raises(ValueError, match="coords"):
        make_erosion_dilation_penalty(radius=0.2, coords=coords, size_px=5)


def test_make_erosion_dilation_penalty_coords_dl_conflict():
    coords = (np.linspace(0.0, 1.0, 5),)
    with pytest.raises(ValueError, match="coords"):
        make_erosion_dilation_penalty(radius=0.2, coords=coords, dl=0.1)
