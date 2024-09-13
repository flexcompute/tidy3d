import numpy as np
import pytest
from tidy3d.plugins.metrics.constants import Constant


@pytest.fixture(params=[1, 2.5, 1 + 2j, np.array([1, 2, 3])])
def value(request):
    return request.param


def test_constant_evaluate(value):
    constant = Constant(value)
    result = constant.evaluate()
    np.testing.assert_allclose(result, value)


def test_constant_type(value):
    constant = Constant(value)
    result = constant.evaluate()
    assert isinstance(result, type(value))
