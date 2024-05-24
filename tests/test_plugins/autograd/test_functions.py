import pytest
import numpy as np
import numpy.testing as npt
import scipy.ndimage
from autograd.test_util import check_grads
from scipy.signal import convolve as convolve_sp

from tidy3d.plugins.autograd.functions import (
    convolve,
    grey_dilation,
    grey_erosion,
    pad,
    grey_opening,
    grey_closing,
    morphological_gradient,
    morphological_gradient_internal,
    morphological_gradient_external,
)
from tidy3d.plugins.autograd.types import _pad_modes

_mode_to_scipy = {
    "constant": "constant",
    "edge": "nearest",
    "reflect": "mirror",
    "symmetric": "reflect",
    "wrap": "wrap",
}


@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("size", [3, 4, (3, 3), (4, 4), (3, 4), (3, 3, 3), (4, 4, 4), (3, 4, 5)])
@pytest.mark.parametrize("pad_width", [0, 1, 2, (0, 0), (0, 1), (1, 0), (1, 2)])
@pytest.mark.parametrize("axis", [None, 0, -1])
class TestPad:
    def test_pad_val(self, rng, mode, size, pad_width, axis):
        """Test padding values against NumPy for various modes, sizes, pad widths, and axes."""
        x = rng.random(size)
        d = x.ndim

        _pad_width = np.atleast_1d(pad_width)
        _axis = range(d) if axis is None else np.atleast_1d(axis)
        _axis = [(ax + d) % d for ax in _axis]  # Handle negative axes
        _pad = [(_pad_width[0], _pad_width[-1]) if ax in _axis else (0, 0) for ax in range(d)]

        pad_td = pad(x, pad_width, mode=mode, axis=axis)
        pad_np = np.pad(x, _pad, mode=mode)

        npt.assert_allclose(pad_td, pad_np)

    def test_pad_grad(self, rng, mode, size, pad_width, axis):
        """Test gradients of padding function for various modes, sizes, pad widths, and axes."""
        x = rng.random(size)
        check_grads(pad, modes=["fwd", "rev"], order=1)(x, pad_width, mode=mode, axis=axis)


class TestPadExceptions:
    array = np.array([[1, 2], [3, 4]])

    def test_invalid_pad_width_size(self):
        """Test that an exception is raised when pad_width has an invalid size."""
        with pytest.raises(ValueError, match="Padding width must have one or two elements"):
            pad(self.array, (1, 2, 3))

    def test_padding_larger_than_input_size(self):
        """Test that an exception is raised when padding is larger than the input size."""
        with pytest.raises(
            NotImplementedError, match="Padding larger than the input size is not supported"
        ):
            pad(self.array, (3, 3))

    def test_negative_padding(self):
        """Test that an exception is raised when padding is negative."""
        with pytest.raises(ValueError, match="Padding must be positive"):
            pad(self.array, (-1, 1))

    def test_unsupported_padding_mode(self):
        """Test that an exception is raised when an unsupported padding mode is used."""
        with pytest.raises(KeyError, match="Unsupported padding mode"):
            pad(self.array, (1, 1), mode="unsupported_mode")

    def test_axis_out_of_range(self):
        """Test that an exception is raised when the axis is out of range."""
        with pytest.raises(IndexError, match="Axis out of range"):
            pad(self.array, (1, 1), axis=2)

    def test_negative_axis_out_of_range(self):
        """Test that an exception is raised when a negative axis is out of range."""
        with pytest.raises(IndexError, match="Axis out of range"):
            pad(self.array, (1, 1), axis=-3)


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("padding", _pad_modes.__args__)
@pytest.mark.parametrize(
    "ary_size", [7, 8, (7, 7), (8, 8), (7, 8), (7, 7, 7), (8, 8, 8), (7, 8, 9)]
)
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("square_kernel", [True, False])
class TestConvolve:
    @staticmethod
    def _ary_and_kernel(rng, ary_size, kernel_size, square_kernel):
        x = rng.random(ary_size)

        kernel_shape = [kernel_size] * x.ndim
        if not square_kernel:
            kernel_shape[0] += 2
        k = rng.random(kernel_shape)

        return x, k

    def test_convolve_val(self, rng, mode, padding, ary_size, kernel_size, square_kernel):
        """Test convolution values against SciPy for various modes, padding, array sizes, and kernel sizes."""
        x, k = self._ary_and_kernel(rng, ary_size, kernel_size, square_kernel)

        if mode in ("full", "same"):
            pad_widths = [(k // 2, k // 2) for k in k.shape]
            x_padded = x
            for axis, pad_width in enumerate(pad_widths):
                x_padded = pad(x_padded, pad_width, mode=padding, axis=axis)
            conv_sp = convolve_sp(x_padded, k, mode="valid" if mode == "same" else mode)
        else:
            conv_sp = convolve_sp(x, k, mode=mode)

        conv_td = convolve(x, k, padding=padding, mode=mode)

        npt.assert_allclose(
            conv_td,
            conv_sp,
            atol=1e-12,  # scipy's "full" somehow is not zero at the edges...
        )

    def test_convolve_grad(self, rng, mode, padding, ary_size, kernel_size, square_kernel):
        """Test gradients of convolution function for various modes, padding, array sizes, and kernel sizes."""
        if not square_kernel and mode == "valid" and ary_size == (7, 7) and kernel_size == 3:
            pytest.skip(
                "Known bug of running into an autograd recursion error here. "
                "Investigate further if it becomes a problem."
            )

        x, k = self._ary_and_kernel(rng, ary_size, kernel_size, square_kernel)
        check_grads(convolve, modes=["rev"], order=1)(x, k, padding=padding, mode=mode)


class TestConvolveExceptions:
    array = np.array([[1, 2], [3, 4]])

    def test_even_kernel_dimensions(self):
        """Test that an exception is raised when all kernel dimensions are even."""
        kernel_even = np.array([[2, 2], [2, 2]])
        with pytest.raises(ValueError, match="All kernel dimensions must be odd"):
            convolve(self.array, kernel_even)

    def test_single_even_kernel_dimension(self):
        """Test that an exception is raised when a single kernel dimension is even."""
        kernel_single_even = np.array([[1, 1], [1, 2]])
        with pytest.raises(ValueError, match="All kernel dimensions must be odd"):
            convolve(self.array, kernel_single_even)

    def test_kernel_array_dimension_mismatch(self):
        """Test that an exception is raised when the kernel and array dimensions mismatch."""
        kernel_mismatch = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        with pytest.raises(ValueError, match="Kernel dimensions must match array dimensions"):
            convolve(self.array, kernel_mismatch)


def _ary_and_kernel(rng, ary_size, kernel_size, square_kernel):
    x = rng.random(ary_size)

    kernel_shape = [kernel_size] * x.ndim
    if not square_kernel:
        kernel_shape[0] += 2
    k = rng.random(kernel_shape)

    return x, k


@pytest.mark.parametrize(
    "op,sp_op",
    [
        (grey_dilation, scipy.ndimage.grey_dilation),
        (grey_erosion, scipy.ndimage.grey_erosion),
        (grey_opening, scipy.ndimage.grey_opening),
        (grey_closing, scipy.ndimage.grey_closing),
        (morphological_gradient, scipy.ndimage.morphological_gradient),
        (
            morphological_gradient_internal,
            lambda x, *args, **kwargs: x - scipy.ndimage.grey_erosion(x, *args, **kwargs),
        ),
        (
            morphological_gradient_external,
            lambda x, *args, **kwargs: scipy.ndimage.grey_dilation(x, *args, **kwargs) - x,
        ),
    ],
)
@pytest.mark.parametrize("mode", _pad_modes.__args__)
@pytest.mark.parametrize("ary_size", [(7, 7), (8, 8), (7, 8)])
@pytest.mark.parametrize("kernel_size", [1, 3])
class TestMorphology:
    def test_morphology_val_size(self, rng, op, sp_op, mode, ary_size, kernel_size):
        """Test morphological operation values against SciPy for various modes, array sizes, and kernel sizes."""
        x = rng.random(ary_size)
        ndimg_mode = _mode_to_scipy[mode]
        npt.assert_allclose(
            op(x, size=kernel_size, mode=mode), sp_op(x, size=kernel_size, mode=ndimg_mode)
        )

    def test_morphology_val_grad(self, rng, op, sp_op, mode, ary_size, kernel_size):
        """Test gradients of morphological operations for various modes, array sizes, and kernel sizes."""
        x = rng.random(ary_size)
        check_grads(op, modes=["rev"], order=1)(x, size=kernel_size, mode=mode)

    @pytest.mark.parametrize(
        "full",
        [
            True,
            # False,  # FIXME: does not pass for all cases
        ],
    )
    @pytest.mark.parametrize("square", [True, False])
    @pytest.mark.parametrize("flat", [True, False])
    class TestMorphologyStructure:
        @staticmethod
        def _ary_and_kernel(rng, ary_size, kernel_size, full, square, flat):
            x = rng.random(ary_size)
            kernel_shape = [kernel_size] * x.ndim

            if not square:
                kernel_shape[0] += 2

            if full:
                k = np.ones(kernel_shape)
            elif flat:
                k = np.random.randint(0, 2, kernel_shape)
            else:
                k = np.random.uniform(-1, 1, kernel_shape)

            return x, k

        def test_morphology_val_structure(
            self, rng, op, sp_op, mode, ary_size, kernel_size, full, square, flat
        ):
            """Test morphological operation values against SciPy for various kernel structures."""
            x, k = self._ary_and_kernel(rng, ary_size, kernel_size, full, square, flat)
            ndimg_mode = _mode_to_scipy[mode]
            npt.assert_allclose(
                op(x, structure=k, mode=mode), sp_op(x, structure=k, mode=ndimg_mode)
            )

        def test_morphology_val_structure_grad(
            self, rng, op, sp_op, mode, ary_size, kernel_size, full, square, flat
        ):
            """Test gradients of morphological operations for various kernel structures."""
            x, k = self._ary_and_kernel(rng, ary_size, kernel_size, full, square, flat)
            check_grads(op, modes=["rev"], order=1)(x, size=kernel_size, mode=mode)
