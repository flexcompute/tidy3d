import pytest
from tidy3d.plugins.autograd.invdes.filters import (
    _get_kernel_size,
    make_circular_filter,
    make_conic_filter,
    make_filter,
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
    @pytest.mark.parametrize("filter_type", ["circular", "conic"])
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
