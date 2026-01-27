from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.interpolate
import scipy.ndimage
from autograd import grad
from autograd.test_util import check_grads
from scipy.signal import convolve as convolve_sp

from tidy3d.compat import np_trapezoid
from tidy3d.plugins.autograd import (
    add_at,
    convolve,
    grey_closing,
    grey_dilation,
    grey_erosion,
    grey_opening,
    interpn,
    least_squares,
    morphological_gradient,
    morphological_gradient_external,
    morphological_gradient_internal,
    pad,
    rescale,
    smooth_max,
    smooth_min,
    threshold,
    trapz,
)
from tidy3d.plugins.autograd.functions import _normalize_axes
from tidy3d.plugins.autograd.types import PaddingType

_mode_to_scipy = {
    "constant": "constant",
    "edge": "nearest",
    "reflect": "mirror",
    "symmetric": "reflect",
    "wrap": "wrap",
}

CONV_MODES = ["full", "same", "valid"]

_CONVOLVE_AXES_CASES = [
    ([0], [0]),
    ([1], [1]),
    ([1], [0]),
    ([-1], [-1]),
]


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
        check_grads(pad, modes=["fwd", "rev"], order=2)(x, pad_width, mode=mode, axis=axis)


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


@pytest.mark.parametrize("mode", CONV_MODES)
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

        conv_td = convolve(x, k, padding=padding, mode=mode)
        conv_sp = _reference_convolution(x, k, mode, padding, axes=None)

        npt.assert_allclose(conv_td, conv_sp, atol=1e-12)

    def test_convolve_grad(self, rng, mode, padding, ary_size, kernel_size, square_kernel):
        """Test gradients of convolution function for various modes, padding, array sizes, and kernel sizes."""
        if not square_kernel and mode == "valid" and ary_size == (7, 7) and kernel_size == 3:
            pytest.skip(
                "Known bug of running into an autograd recursion error here. "
                "Investigate further if it becomes a problem."
            )

        x, k = self._ary_and_kernel(rng, ary_size, kernel_size, square_kernel)
        check_grads(convolve, modes=["rev"], order=2)(x, k, padding=padding, mode=mode)


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


def _reference_convolve_with_axes(array, kernel, axes_array, axes_kernel, mode):
    """Construct a SciPy reference for convolutions with explicit axes."""

    array_batch_axes = tuple(ax for ax in range(array.ndim) if ax not in axes_array)
    kernel_batch_axes = tuple(ax for ax in range(kernel.ndim) if ax not in axes_kernel)

    array_perm = array_batch_axes + axes_array
    kernel_perm = kernel_batch_axes + axes_kernel

    array_reordered = np.transpose(array, array_perm)
    kernel_reordered = np.transpose(kernel, kernel_perm)

    len_array_batch = len(array_batch_axes)
    len_kernel_batch = len(kernel_batch_axes)

    array_batch_shape = array_reordered.shape[:len_array_batch]
    kernel_batch_shape = kernel_reordered.shape[:len_kernel_batch]

    sample_conv = convolve_sp(
        array_reordered[(0,) * len_array_batch],
        kernel_reordered[(0,) * len_kernel_batch],
        mode=mode,
    )
    conv_shape = sample_conv.shape

    expected = np.empty(array_batch_shape + kernel_batch_shape + conv_shape)

    for idx_array in np.ndindex(array_batch_shape):
        array_slice = array_reordered[idx_array]
        for idx_kernel in np.ndindex(kernel_batch_shape):
            kernel_slice = kernel_reordered[idx_kernel]
            expected[idx_array + idx_kernel] = convolve_sp(array_slice, kernel_slice, mode=mode)

    return expected


def _prepare_reference_inputs(array, kernel, mode, padding, axes):
    """Apply padding logic to match tidy3d's convolution before building a reference."""

    axes_array, axes_kernel = _normalize_axes(array.ndim, kernel.ndim, axes)

    working_array = array
    scipy_mode = mode

    if mode in ("same", "full"):
        for ax_array, ax_kernel in zip(axes_array, axes_kernel):
            pad_width = (
                kernel.shape[ax_kernel] // 2 if mode == "same" else kernel.shape[ax_kernel] - 1
            )
            if pad_width > 0:
                working_array = pad(
                    working_array, (pad_width, pad_width), mode=padding, axis=ax_array
                )
        scipy_mode = "valid"

    working_array_np = np.asarray(working_array)
    kernel_np = np.asarray(kernel)

    return working_array_np, kernel_np, axes_array, axes_kernel, scipy_mode


def _reference_convolution(array, kernel, mode, padding, axes):
    """Full reference that mimics tidy3d padding rules before SciPy convolution."""

    working_array_np, kernel_np, axes_array, axes_kernel, scipy_mode = _prepare_reference_inputs(
        array,
        kernel,
        mode,
        padding,
        axes,
    )

    return _reference_convolve_with_axes(
        working_array_np,
        kernel_np,
        axes_array,
        axes_kernel,
        scipy_mode,
    )


@pytest.mark.parametrize("mode", CONV_MODES)
@pytest.mark.parametrize("padding", PaddingType.__args__)
@pytest.mark.parametrize("axes", _CONVOLVE_AXES_CASES)
class TestConvolveAxes:
    def test_convolve_axes_val(self, rng, mode, padding, axes):
        """Test convolution with explicit axes against NumPy implementations."""
        array = rng.random((2, 5))
        kernel = rng.random((3, 3))

        conv_td = convolve(array, kernel, padding=padding, mode=mode, axes=axes)
        expected = _reference_convolution(array, kernel, mode, padding, axes)

        npt.assert_allclose(conv_td, expected, atol=1e-12)

    def test_convolve_axes_grad(self, rng, axes, mode, padding):
        """Test gradients of convolution when specific axes are provided."""
        array = rng.random((2, 5))
        kernel = rng.random((3, 3))
        check_grads(convolve, modes=["rev"], order=2)(
            array,
            kernel,
            padding=padding,
            mode=mode,
            axes=axes,
        )


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
            check_grads(op, modes=["rev"], order=1)(x, structure=k, mode=mode)


class TestMorphology1D:
    """Test morphological operations with 1D-like structuring elements."""

    @pytest.mark.parametrize("h, w", [(1, 3), (3, 1), (1, 5), (5, 1)])
    def test_1d_structuring_elements(self, rng, h, w):
        """Test grey dilation with 1D-like structuring elements on 2D arrays."""
        x = rng.random((8, 8))

        # Test with size parameter
        size_tuple = (h, w)
        result_size = grey_dilation(x, size=size_tuple)

        # Verify output shape matches input
        assert result_size.shape == x.shape

        # Verify that dilation actually increases values (or keeps them the same)
        assert np.all(result_size >= x)

        # Test that we can also use structure parameter with 1D-like arrays
        structure = np.ones((h, w))
        result_struct = grey_dilation(x, structure=structure)
        assert result_struct.shape == x.shape

    def test_1d_gradient_flow(self, rng):
        """Test gradient flow through 1D-like structuring elements."""
        x = rng.random((6, 6))

        # Test horizontal 1D structure
        check_grads(lambda x: grey_dilation(x, size=(1, 3)), modes=["rev"], order=1)(x)

        # Test vertical 1D structure
        check_grads(lambda x: grey_dilation(x, size=(3, 1)), modes=["rev"], order=1)(x)

        # Test with structure parameter
        struct_h = np.ones((1, 3))
        struct_v = np.ones((3, 1))
        check_grads(lambda x: grey_dilation(x, structure=struct_h), modes=["rev"], order=1)(x)
        check_grads(lambda x: grey_dilation(x, structure=struct_v), modes=["rev"], order=1)(x)


class TestMorphologyExceptions:
    """Test exceptions in morphological operations."""

    def test_no_size_or_structure(self, rng):
        """Test that an exception is raised when neither size nor structure is provided."""
        x = rng.random((5, 5))
        with pytest.raises(ValueError, match="Either size or structure must be provided"):
            grey_dilation(x)

    def test_even_structure_dimensions(self, rng):
        """Test that an exception is raised for even-dimensioned structuring elements."""
        x = rng.random((5, 5))
        k_even = np.ones((4, 4))
        with pytest.raises(ValueError, match="Structuring element dimensions must be odd"):
            grey_dilation(x, structure=k_even)

    def test_both_size_and_structure(self, rng):
        """Test that an exception is raised when both size and structure are provided."""
        x = rng.random((5, 5))
        k = np.ones((3, 3))
        with pytest.raises(ValueError, match="Cannot specify both size and structure"):
            grey_dilation(x, size=3, structure=k)


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


def test_rescale_clips_output_to_bounds():
    """Test that rescale clips output to [out_min, out_max] even when input is slightly outside [in_min, in_max].

    This is a regression test for a numerical precision issue where filter_project + tanh_projection
    could produce values slightly outside [0, 1] (e.g., -1e-15), causing rescale to produce
    permittivity values slightly below 1.0, which would fail CustomMedium validation.
    """
    # Simulate input slightly outside the expected [0, 1] range due to numerical precision
    array_with_numerical_error = np.array([-1e-15, 0.5, 1.0 + 1e-15])

    out_min, out_max = 1.0, 2.75
    in_min, in_max = 0.0, 1.0

    result = rescale(array_with_numerical_error, out_min, out_max, in_min, in_max)

    # Without clipping, result[0] would be slightly below 1.0 (e.g., 0.999999999999998)
    # and result[2] would be slightly above 2.75
    assert result.min() >= out_min, f"Output {result.min()} is below out_min={out_min}"
    assert result.max() <= out_max, f"Output {result.max()} is above out_max={out_max}"

    npt.assert_equal(result[0], out_min)
    npt.assert_equal(result[2], out_max)


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

    def test_interpn_values_grad(self, rng, dim, method):
        points, values, xi = self.generate_points_values_xi(rng, dim)
        check_grads(lambda v: interpn(points, v, xi, method=method), modes=["fwd", "rev"], order=2)(
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
        result_numpy = np_trapezoid(y, x=x, dx=dx, axis=axis)
        npt.assert_allclose(result_custom, result_numpy)

    def test_trapz_grad(self, rng, shape, axis, use_x):
        """Test gradients of trapz function for different array dimensions and integration axes."""
        y, x, dx = self.generate_y_x_dx(rng, shape, use_x)
        check_grads(lambda y: trapz(y, x=x, dx=dx, axis=axis), modes=["fwd", "rev"], order=2)(y)


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

    def test_add_at_grad(self, rng, shape, indices):
        """Test gradients of add_at function for different array dimensions and indices."""
        x, y = self.generate_x_y(rng, shape, indices)
        check_grads(lambda x: add_at(x, indices, y), modes=["fwd", "rev"], order=2)(x)
        check_grads(lambda y: add_at(x, indices, y), modes=["fwd", "rev"], order=2)(y)


def test_add_at_grad_kwargs(rng):
    """Test add_at function for different array dimensions and indices, with kwargs."""
    indices = (0,)
    x = rng.uniform(-1, 1, (10,))
    y = rng.uniform(-1, 1, x[tuple(indices)].shape)
    # this should not error
    grad(lambda y_: add_at(x=x, y=y_, indices_x=indices)[0])(y)


@pytest.mark.parametrize("shape", [(5,), (5, 5), (5, 5, 5)])
@pytest.mark.parametrize("tau", [1e-3, 1.0])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
class TestSmoothMax:
    def test_smooth_max_values(self, rng, shape, tau, axis):
        """Test `smooth_max` values for various shapes, tau, and axes."""

        if axis == 1 and len(shape) == 1:
            pytest.skip()

        x = rng.uniform(-10, 10, size=shape)
        result = smooth_max(x, tau=tau, axis=axis)

        expected = np.max(x, axis=axis)
        npt.assert_allclose(result, expected, atol=10 * tau)

    def test_smooth_max_grad(self, check_grads_with_tolerance, rng, shape, tau, axis):
        """Test gradients of `smooth_max` for various parameters."""

        if axis == 1 and len(shape) == 1:
            pytest.skip()

        x = rng.uniform(-1, 1, size=shape)
        func = lambda x: smooth_max(x, tau=tau, axis=axis)
        check_grads_with_tolerance(func, modes=["fwd", "rev"], order=2, tol=1e-5, rtol=1e-5)(x)


@pytest.mark.parametrize("shape", [(5,), (5, 5), (5, 5, 5)])
@pytest.mark.parametrize("tau", [1e-3, 1.0])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
class TestSmoothMin:
    def test_smooth_min_values(self, rng, shape, tau, axis):
        """Test `smooth_min` values for various shapes, tau, and axes."""

        if axis == 1 and len(shape) == 1:
            pytest.skip()

        x = rng.uniform(-10, 10, size=shape)
        result = smooth_min(x, tau=tau, axis=axis)

        expected = np.min(x, axis=axis)
        npt.assert_allclose(result, expected, atol=10 * tau)

    def test_smooth_min_grad(self, check_grads_with_tolerance, rng, shape, tau, axis):
        """Test gradients of `smooth_min` for various parameters."""

        if axis == 1 and len(shape) == 1:
            pytest.skip()

        x = rng.uniform(-1, 1, size=shape)
        func = lambda x: smooth_min(x, tau=tau, axis=axis)
        check_grads_with_tolerance(func, modes=["fwd", "rev"], order=2, tol=1e-5, rtol=1e-5)(x)


class TestLeastSquares:
    @pytest.mark.parametrize(
        "model, params_true, initial_guess, x, y",
        [
            (
                lambda x, a, b: a * x + b,
                np.array([2.0, -3.0]),
                (0.0, 0.0),
                np.linspace(0, 10, 50),
                2.0 * np.linspace(0, 10, 50) - 3.0,
            ),
            (
                lambda x, a, b, c: a * x**2 + b * x + c,
                np.array([1.0, -2.0, 1.0]),
                (0.0, 0.0, 0.0),
                np.linspace(-5, 5, 100),
                1.0 * np.linspace(-5, 5, 100) ** 2 - 2.0 * np.linspace(-5, 5, 100) + 1.0,
            ),
            (
                lambda x, a, b: a * np.exp(b * x),
                np.array([1.5, 0.5]),
                (1.0, 0.0),
                np.linspace(0, 2, 50),
                1.5 * np.exp(0.5 * np.linspace(0, 2, 50)),
            ),
        ],
    )
    def test_least_squares(self, model, params_true, initial_guess, x, y):
        """Test least_squares function with different models."""
        params_estimated = least_squares(model, x, y, initial_guess)
        npt.assert_allclose(params_estimated, params_true, rtol=1e-5)

    def test_least_squares_with_noise(self, rng):
        """Test least_squares function with noisy data."""

        model = lambda x, a, b: a * x + b
        a_true, b_true = -1.0, 4.0
        params_true = np.array([a_true, b_true])
        x = np.linspace(0, 10, 100)
        noise = rng.normal(scale=0.1, size=x.shape)
        y = a_true * x + b_true + noise
        initial_guess = (0.0, 0.0)

        params_estimated = least_squares(model, x, y, initial_guess)

        npt.assert_allclose(params_estimated, params_true, rtol=1e-1)

    def test_least_squares_no_convergence(self):
        """Test that least_squares function raises an error when not converging."""

        def constant_model(x, a):
            return a

        x = np.linspace(0, 10, 50)
        y = 2.0 * x - 3.0  # Linear data
        initial_guess = (0.0,)

        with pytest.raises(np.linalg.LinAlgError):
            least_squares(constant_model, x, y, initial_guess, max_iterations=10, tol=1e-12)

    def test_least_squares_gradient(self):
        """Test gradients of least_squares function with respect to parameters."""

        def linear_model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 50)
        y = 2.0 * x - 3.0
        initial_guess = (1.0, 0.0)

        check_grads(
            lambda params: least_squares(linear_model, x, y, params, max_iterations=1),
            modes=["fwd", "rev"],
            order=2,
        )(initial_guess)
