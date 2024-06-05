import pytest
import numpy as np
import numpy.testing as npt
from tidy3d.plugins.autograd.utilities import make_kernel, chain


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
