import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from tidy3d.exceptions import Tidy3dError
from tidy3d.plugins.autograd import (
    chain,
    get_kernel_size_px,
    make_kernel,
    scalar_objective,
    value_and_grad,
)


@pytest.mark.parametrize("size", [(3, 3), (4, 4), (5, 5)])
@pytest.mark.parametrize("normalize", [True, False])
class TestMakeKernel:
    def test_make_kernel_circular(self, size, normalize):
        """Test make_kernel function for circular kernel."""
        kernel = make_kernel("circular", size, normalize=normalize)
        assert kernel.shape == size
        if normalize:
            assert np.isclose(np.sum(kernel), 1.0)

        # Check that the corners of the circular kernel are zero
        assert all(kernel[i, j] == 0 for i in [0, -1] for j in [0, -1])

    def test_make_kernel_conic(self, size, normalize):
        """Test make_kernel function for conic kernel."""
        kernel = make_kernel("conic", size, normalize=normalize)
        assert kernel.shape == size
        if normalize:
            assert np.isclose(np.sum(kernel), 1.0)

        # Check that the corners of the conic kernel are zero
        assert all(kernel[i, j] == 0 for i in [0, -1] for j in [0, -1])


class TestMakeKernelExceptions:
    def test_make_kernel_invalid_type(self):
        """Test make_kernel function for invalid kernel type."""
        size = (5, 5)
        with pytest.raises(ValueError, match="Unsupported kernel type"):
            make_kernel("invalid_type", size)

    def test_make_kernel_invalid_size(self):
        """Test make_kernel function for invalid size."""
        size = (5, -5)
        with pytest.raises(ValueError, match="must be an iterable of positive integers"):
            make_kernel("circular", size)


@pytest.mark.parametrize(
    "radius, dl, expected",
    [
        (1, 0.1, 21),
        (1, [0.1, 0.2], [21, 11]),
        ([1, 2], 0.1, [21, 41]),
        ([1, 1], [0.1, 0.2], [21, 11]),
        ([1, 2], [0.1, 0.1], [21, 41]),
    ],
)
def test_get_kernel_size_px_with_radius_and_dl(radius, dl, expected):
    result = get_kernel_size_px(radius, dl)
    assert result == expected


def test_get_kernel_size_px_invalid_arguments():
    """Test get_kernel_size_px function with invalid arguments."""
    with pytest.raises(ValueError, match="must be provided"):
        get_kernel_size_px()


class TestChain:
    def test_chain_functions(self):
        """Test chain function with multiple functions."""

        def add_one(x):
            return x + 1

        def square(x):
            return x**2

        chained_func = chain(add_one, square)
        array = np.array([1, 2, 3])
        result = chained_func(array)
        expected = np.array([4, 9, 16])
        npt.assert_allclose(result, expected)

    def test_chain_single_iterable(self):
        """Test chain function with a single iterable of functions."""

        def add_one(x):
            return x + 1

        def square(x):
            return x**2

        funcs = [add_one, square]
        chained_func = chain(funcs)
        array = np.array([1, 2, 3])
        result = chained_func(array)
        expected = np.array([4, 9, 16])
        npt.assert_allclose(result, expected)

    def test_chain_invalid_function(self):
        """Test chain function with an invalid function in the list."""

        def add_one(x):
            return x + 1

        funcs = [add_one, "not_a_function"]
        with pytest.raises(TypeError, match="All elements in funcs must be callable"):
            chain(funcs)


class TestScalarObjective:
    def test_scalar_objective_no_aux(self):
        """Test scalar_objective decorator without auxiliary data."""

        @scalar_objective
        def objective(x):
            da = xr.DataArray(x)
            return da.sum()

        x = np.array([1.0, 2.0, 3.0])
        result, grad = value_and_grad(objective)(x)
        assert np.allclose(grad, np.ones_like(grad))
        assert np.isclose(result, 6.0)

    def test_scalar_objective_with_aux(self):
        """Test scalar_objective decorator with auxiliary data."""

        @scalar_objective(has_aux=True)
        def objective(x):
            da = xr.DataArray(x)
            return da.sum(), "auxiliary_data"

        x = np.array([1.0, 2.0, 3.0])
        (result, grad), aux_data = value_and_grad(objective, has_aux=True)(x)
        assert np.allclose(grad, np.ones_like(grad))
        assert np.isclose(result, 6.0)
        assert aux_data == "auxiliary_data"

    def test_scalar_objective_invalid_return(self):
        """Test scalar_objective decorator with invalid return value."""

        @scalar_objective
        def objective(x):
            da = xr.DataArray(x)
            return da  # Returning the array directly, not a scalar

        x = np.array([1, 2, 3])
        with pytest.raises(Tidy3dError, match="must be a scalar"):
            objective(x)

    def test_scalar_objective_float(self):
        """Test scalar_objective decorator with a Python float return value."""

        @scalar_objective
        def objective(x):
            return x**2

        result, grad = value_and_grad(objective)(3.0)
        assert np.isclose(grad, 6.0)
        assert np.isclose(result, 9.0)
