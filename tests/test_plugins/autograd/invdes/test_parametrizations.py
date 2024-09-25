import autograd.numpy as np
import pytest
from tidy3d.plugins.autograd.invdes.parametrizations import make_filter_and_project
from tidy3d.plugins.autograd.types import PaddingType


@pytest.mark.parametrize("radius", [1, 2, (1, 2)])
@pytest.mark.parametrize("dl", [0.1, 0.2, (0.1, 0.2)])
@pytest.mark.parametrize("size_px", [None, 5, (5, 7)])
@pytest.mark.parametrize("filter_type", ["circular", "conic"])
@pytest.mark.parametrize("padding", PaddingType.__args__)
def test_make_filter_and_project(rng, radius, dl, size_px, filter_type, padding):
    """Test make_filter_and_project function for various parameters."""
    filter_and_project_func = make_filter_and_project(
        radius=radius,
        dl=dl,
        size_px=size_px,
        beta=10,
        eta=0.5,
        filter_type=filter_type,
        padding=padding,
    )
    array = rng.random((51, 51))
    result = filter_and_project_func(array)
    assert result.shape == array.shape
    assert np.all(result >= 0) and np.all(result <= 1)
