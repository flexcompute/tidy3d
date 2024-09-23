import numpy as np
import pytest
from tidy3d.plugins.expressions.functions import Cos, Exp, Log, Log10, Sin, Sqrt, Tan
from tidy3d.plugins.expressions.variables import Constant

FUNCTIONS = [
    (Sin, np.sin),
    (Cos, np.cos),
    (Tan, np.tan),
    (Exp, np.exp),
    (Log, np.log),
    (Log10, np.log10),
    (Sqrt, np.sqrt),
]


@pytest.fixture(params=[1, 2.5, 1 + 2j, np.array([1, 2, 3]), np.array([1 + 2j, 3 - 4j, 5 + 6j])])
def value(request):
    return request.param


@pytest.mark.parametrize("tidy3d_func, numpy_func", FUNCTIONS)
def test_functions_evaluate(tidy3d_func, numpy_func, value):
    constant = Constant(value)
    func = tidy3d_func(constant)
    result = func.evaluate(constant.evaluate())
    np.testing.assert_allclose(result, numpy_func(value))


@pytest.mark.parametrize("tidy3d_func, numpy_func", FUNCTIONS)
def test_functions_type(tidy3d_func, numpy_func, value):
    constant = Constant(value)
    func = tidy3d_func(constant)
    result = func.evaluate(constant.evaluate())
    assert isinstance(result, type(numpy_func(value)))
