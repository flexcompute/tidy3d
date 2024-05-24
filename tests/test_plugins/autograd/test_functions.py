import pytest
import numpy as np
import numpy.testing as npt
import scipy.ndimage
from autograd.test_util import check_grads
from scipy.signal import convolve as convolve_sp

from tidy3d.plugins.autograd.functions import (
    _mode_to_scipy,
    convolve,
    grey_closing,
    grey_dilation,
    grey_erosion,
    grey_opening,
    morphological_gradient,
    pad,
)
from tidy3d.plugins.autograd.types import _pad_modes


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("pad_width", [0, 1, 2, (0, 0), (0, 1), (1, 0), (1, 2)])
@pytest.mark.parametrize("axis", [None, 0, -1])
class TestPad:
    def test_pad_val(self, rng, mode, size, ndim, pad_width, axis):
        x = rng.random((size,) * ndim)

        _pad_width = np.atleast_1d(pad_width)
        _axis = range(ndim) if axis is None else np.atleast_1d(axis)
        _axis = [(ax + ndim) % ndim for ax in _axis]  # Handle negative axes
        _pad = [(_pad_width[0], _pad_width[-1]) if ax in _axis else (0, 0) for ax in range(ndim)]

        pad_td = pad(x, pad_width, mode=mode, axis=axis)
        pad_np = np.pad(x, _pad, mode=mode)

        npt.assert_allclose(pad_td, pad_np)

    def test_pad_grad(self, rng, mode, size, ndim, pad_width, axis):
        x = rng.random((size,) * ndim)
        check_grads(pad, modes=["fwd", "rev"], order=2)(x, pad_width, mode=mode, axis=axis)


class TestPadExceptions:
    array = np.array([[1, 2], [3, 4]])

    def test_invalid_pad_width_size(self):
        with pytest.raises(ValueError, match="Padding width must have one or two elements"):
            pad(self.array, (1, 2, 3))

    def test_padding_larger_than_input_size(self):
        with pytest.raises(
            NotImplementedError, match="Padding larger than the input size is not supported"
        ):
            pad(self.array, (3, 3))

    def test_negative_padding(self):
        with pytest.raises(ValueError, match="Padding must be positive"):
            pad(self.array, (-1, 1))

    def test_unsupported_padding_mode(self):
        with pytest.raises(KeyError, match="Unsupported padding mode"):
            pad(self.array, (1, 1), mode="unsupported_mode")

    def test_axis_out_of_range(self):
        with pytest.raises(IndexError, match="Axis out of range"):
            pad(self.array, (1, 1), axis=2)

    def test_negative_axis_out_of_range(self):
        with pytest.raises(IndexError, match="Axis out of range"):
            pad(self.array, (1, 1), axis=-3)


# @pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("mode", ["valid", "same"])
@pytest.mark.parametrize("padding", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_convolve_val(rng, mode, padding, ary_size, kernel_size, ndim):
    def _conv(x, k):
        if mode == "valid":
            return convolve_sp(x, k, mode=mode)

        p = k.shape[-1] // 2
        x = np.pad(x, p, mode=padding)

        if mode == "same":
            _mode = "valid"

        return convolve_sp(x, k, mode=mode)

    x = rng.random((ary_size,) * ndim)
    k = rng.random((kernel_size,) * ndim)

    npt.assert_allclose(convolve(x, k, padding=padding, mode=mode), _conv(x, k))


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("ks", [1, 2, 3])
def test_convolve_grad(rng, mode, ary_size, ks):
    x = rng.random((ary_size, ary_size))
    k = rng.random((ks, ks))
    check_grads(convolve, modes=["rev"], order=2)(x, k, mode=mode)


@pytest.mark.parametrize(
    "op,sp_op",
    [
        (grey_dilation, scipy.ndimage.grey_dilation),
        (grey_erosion, scipy.ndimage.grey_erosion),
        (grey_opening, scipy.ndimage.grey_opening),
        (grey_closing, scipy.ndimage.grey_closing),
        (morphological_gradient, scipy.ndimage.morphological_gradient),
    ],
)
@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("ks", [1, 3])
@pytest.mark.parametrize(
    "kind",
    [
        "flat",
        pytest.param(
            "full", marks=pytest.mark.skip(reason="Full structuring elements are not supported.")
        ),
    ],
)
def test_morphology_val(rng, op, sp_op, mode, ary_size, ks, kind):
    x = rng.random((ary_size, ary_size))

    match kind:
        case "flat":
            s = np.ones((ks, ks))
        case "full":
            s = rng.randint(0, 2, (ks, ks))

    ndimg_mode = _mode_to_scipy[mode]
    npt.assert_allclose(op(x, s, mode=mode), sp_op(x, structure=s, mode=ndimg_mode))


@pytest.mark.parametrize(
    "op",
    [grey_dilation, grey_erosion, grey_opening, grey_closing, morphological_gradient],
)
@pytest.mark.parametrize("mode", ["reflect", "constant", "symmetric", "wrap"])
@pytest.mark.parametrize("ary_size", [10, 11])
@pytest.mark.parametrize("ks", [1, 3])
@pytest.mark.parametrize(
    "kind",
    [
        "flat",
        pytest.param(
            "full", marks=pytest.mark.skip(reason="Full structuring elements are not supported.")
        ),
    ],
)
def test_morphology_grad(rng, op, mode, ary_size, ks, kind):
    x = rng.random((ary_size, ary_size))

    match kind:
        case "flat":
            s = np.ones((ks, ks))
        case "full":
            s = rng.randint(0, 2, (ks, ks))

    check_grads(op, modes=["rev"], order=2)(x, s, mode=mode)
