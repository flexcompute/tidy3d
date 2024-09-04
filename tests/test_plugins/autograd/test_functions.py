import numpy as np
import numpy.testing as npt
import pytest
import scipy.interpolate
import scipy.ndimage
from autograd.test_util import check_grads
from scipy.signal import convolve as convolve_sp
from tidy3d.plugins.autograd.functions import (
    add_at,
    convolve,
    grey_closing,
    grey_dilation,
    grey_erosion,
    grey_opening,
    interpn,
    morphological_gradient,
    morphological_gradient_external,
    morphological_gradient_internal,
    pad,
    rescale,
    threshold,
    trapz,
)
from tidy3d.plugins.autograd.types import PaddingType

_mode_to_scipy = {
    "constant": "constant",
    "edge": "nearest",
    "reflect": "mirror",
    "symmetric": "reflect",
    "wrap": "wrap",
}


@pytest.mark.parametrize("mode", PaddingType.__args__)
@pytest.mark.parametrize("size", [3, 4, (3, 3), (4, 4), (3, 4), (3, 3, 3), (4, 4, 4), (3, 4, 5)])
@pytest.mark.parametrize("pad_width", [0, 1, 2, 4, 5, (0, 0), (0, 1), (1, 0), (1, 2)])
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

    def test_negative_padding(self):
        """Test that an exception is raised when padding is negative."""
        with pytest.raises(ValueError, match="Padding must be non-negative"):
            pad(self.array, (-1, 1))

    def test_unsupported_padding_mode(self):
        """Test that an exception is raised when an unsupported padding mode is used."""
        with pytest.raises(ValueError, match="Unsupported padding mode"):
            pad(self.array, (1, 1), mode="unsupported_mode")

    def test_axis_out_of_range(self):
        """Test that an exception is raised when the axis is out of range."""
        with pytest.raises(IndexError, match="out of range"):
            pad(self.array, (1, 1), axis=2)

    def test_negative_axis_out_of_range(self):
        """Test that an exception is raised when a negative axis is out of range."""
        with pytest.raises(IndexError, match="out of range"):
            pad(self.array, (1, 1), axis=-3)


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("padding", PaddingType.__args__)
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
@pytest.mark.parametrize("mode", PaddingType.__args__)
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


@pytest.mark.parametrize(
    "array, out_min, out_max, in_min, in_max, expected",
    [
        (np.array([0, 0.5, 1]), 0, 10, 0, 1, np.array([0, 5, 10])),
        (np.array([0, 0.5, 1]), -1, 1, 0, 1, np.array([-1, 0, 1])),
        (np.array([0, 1, 2]), 0, 1, 0, 2, np.array([0, 0.5, 1])),
        (np.array([-1, 0, 1]), -10, 10, -1, 1, np.array([-10, 0, 10])),
        (np.array([-2, -1, 0]), -1, 1, -2, 0, np.array([-1, 0, 1])),
    ],
)
def test_rescale(array, out_min, out_max, in_min, in_max, expected):
    """Test rescale function for various input and output ranges."""
    result = rescale(array, out_min, out_max, in_min, in_max)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "array, out_min, out_max, in_min, in_max, expected_message",
    [
        (np.array([0, 0.5, 1]), 10, 0, 0, 1, "must be less than"),
        (np.array([0, 0.5, 1]), 0, 10, 1, 1, "must not be equal"),
        (np.array([0, 0.5, 1]), 0, 10, 1, 0, "must be less than"),
    ],
)
def test_rescale_exceptions(array, out_min, out_max, in_min, in_max, expected_message):
    """Test rescale function for expected exceptions."""
    with pytest.raises(ValueError, match=expected_message):
        rescale(array, out_min, out_max, in_min, in_max)


@pytest.mark.parametrize(
    "ary, vmin, vmax, level, expected",
    [
        (np.array([0, 0.5, 1]), 0, 1, 0.5, np.array([0, 1, 1])),
        (np.array([0, 0.5, 1]), 0, 1, None, np.array([0, 1, 1])),
        (np.array([0, 0.5, 1]), -1, 1, 0.5, np.array([-1, 1, 1])),
    ],
)
def test_threshold(ary, vmin, vmax, level, expected):
    """Test threshold function values for threshold levels and value ranges."""
    result = threshold(ary, vmin, vmax, level)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "array, vmin, vmax, level, expected_message",
    [
        (np.array([0, 0.5, 1]), 1, 0, None, "threshold range"),
        (np.array([0, 0.5, 1]), 0, 1, -0.5, "threshold level"),
        (np.array([0, 0.5, 1]), 0, 1, 1.5, "threshold level"),
    ],
)
def test_threshold_exceptions(array, vmin, vmax, level, expected_message):
    """Test threshold function for expected exceptions."""
    with pytest.raises(ValueError, match=expected_message):
        threshold(array, vmin, vmax, level)


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("method", ["linear", "nearest"])
class TestInterpn:
    @staticmethod
    def generate_points_values_xi(rng, dim):
        points = tuple(np.linspace(0, 1, 10) for _ in range(dim))
        values = rng.random([p.size for p in points])
        xi = tuple(np.linspace(0, 1, 5) for _ in range(dim))
        return points, values, xi

    def test_interpn_val(self, rng, dim, method):
        points, values, xi = self.generate_points_values_xi(rng, dim)
        xi_grid = np.meshgrid(*xi, indexing="ij")

        result_custom = interpn(points, values, tuple(xi_grid), method=method)
        result_scipy = scipy.interpolate.interpn(points, values, tuple(xi_grid), method=method)
        npt.assert_allclose(result_custom, result_scipy)

    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["fwd", "rev"])
    def test_interpn_values_grad(self, rng, dim, method, order, mode):
        points, values, xi = self.generate_points_values_xi(rng, dim)
        check_grads(lambda v: interpn(points, v, xi, method=method), modes=[mode], order=order)(
            values
        )


class TestInterpnExceptions:
    def test_invalid_method(self, rng):
        """Test that an exception is raised for an invalid interpolation method."""
        points, values, xi = TestInterpn.generate_points_values_xi(rng, 2)
        with pytest.raises(ValueError, match="interpolation method"):
            interpn(points, values, xi, method="invalid_method")


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("shape", [(10,), (10, 10)])
@pytest.mark.parametrize("use_x", [True, False])
class TestTrapz:
    @staticmethod
    def generate_y_x_dx(rng, shape, use_x):
        y = rng.uniform(-1, 1, shape)
        if use_x:
            x = rng.random(shape)
            dx = 1.0  # dx is not used when x is provided
        else:
            x = None
            dx = rng.random() + 0.1  # ensure dx is not zero
        return y, x, dx

    def test_trapz_val(self, rng, shape, axis, use_x):
        """Test trapz values against NumPy for different array dimensions and integration axes."""
        y, x, dx = self.generate_y_x_dx(rng, shape, use_x)
        result_custom = trapz(y, x=x, dx=dx, axis=axis)
        result_numpy = np.trapz(y, x=x, dx=dx, axis=axis)
        npt.assert_allclose(result_custom, result_numpy)

    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["fwd", "rev"])
    def test_trapz_grad(self, rng, shape, axis, use_x, order, mode):
        """Test gradients of trapz function for different array dimensions and integration axes."""
        y, x, dx = self.generate_y_x_dx(rng, shape, use_x)
        check_grads(lambda y: trapz(y, x=x, dx=dx, axis=axis), modes=[mode], order=order)(y)


@pytest.mark.parametrize("shape", [(10,), (10, 10)])
@pytest.mark.parametrize("indices", [(0,), (slice(3, 8),)])
class TestAddAt:
    @staticmethod
    def generate_x_y(rng, shape, indices):
        x = rng.uniform(-1, 1, shape)
        y = rng.uniform(-1, 1, x[tuple(indices)].shape)
        return x, y

    def test_add_at_val(self, rng, shape, indices):
        """Test add_at values against NumPy for different array dimensions and indices."""
        x, y = self.generate_x_y(rng, shape, indices)
        result_custom = add_at(x, indices, y)
        result_numpy = np.array(x)
        result_numpy[indices] += y
        npt.assert_allclose(result_custom, result_numpy)

    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["fwd", "rev"])
    def test_add_at_grad(self, rng, shape, indices, order, mode):
        """Test gradients of add_at function for different array dimensions and indices."""
        x, y = self.generate_x_y(rng, shape, indices)
        check_grads(lambda x: add_at(x, indices, y), modes=[mode], order=order)(x)
        check_grads(lambda y: add_at(x, indices, y), modes=[mode], order=order)(y)
