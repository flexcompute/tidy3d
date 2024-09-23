import numpy as np
import pytest
from tidy3d.plugins.expressions.variables import Constant, Variable


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


def test_variable_evaluate_positional(value):
    variable = Variable()
    result = variable.evaluate(value)
    np.testing.assert_allclose(result, value)


def test_variable_evaluate_named(value):
    variable = Variable(name="x")
    result = variable.evaluate(x=value)
    np.testing.assert_allclose(result, value)


def test_variable_missing_positional():
    variable = Variable()
    with pytest.raises(ValueError, match="No positional argument provided for unnamed variable."):
        variable.evaluate()


def test_variable_missing_named():
    variable = Variable(name="x")
    with pytest.raises(ValueError, match="Variable 'x' not provided."):
        variable.evaluate()


def test_variable_wrong_named():
    variable = Variable(name="x")
    with pytest.raises(ValueError, match="Variable 'x' not provided."):
        variable.evaluate(y=5)


def test_variable_repr():
    variable_unnamed = Variable()
    variable_named = Variable(name="x")
    assert repr(variable_unnamed) == "Variable()"
    assert repr(variable_named) == "x"


def test_variable_in_expression_positional(value):
    variable = Variable()
    expr = variable + 2
    result = expr(value)
    expected = value + 2
    np.testing.assert_allclose(result, expected)


def test_variable_in_expression_named(value):
    variable = Variable(name="x")
    expr = variable + 2
    result = expr(x=value)
    expected = value + 2
    np.testing.assert_allclose(result, expected)


def test_variable_mixed_args():
    variable_unnamed = Variable()
    variable_named = Variable(name="x")
    expr = variable_unnamed + variable_named
    result = expr(5, x=3)
    expected = 5 + 3
    np.testing.assert_allclose(result, expected)


def test_variable_missing_args():
    variable_unnamed = Variable()
    variable_named = Variable(name="x")
    expr = variable_unnamed + variable_named
    with pytest.raises(ValueError, match="No positional argument provided for unnamed variable."):
        expr(x=3)
    with pytest.raises(ValueError, match="Variable 'x' not provided."):
        expr(5)


def test_variable_multiple_positional_args():
    variable1 = Variable()
    variable2 = Variable()
    expr = variable1 + variable2
    with pytest.raises(ValueError, match="Multiple positional arguments"):
        expr(5, 3)


def test_single_unnamed_variable_multiple_args():
    variable = Variable()
    expr = variable * 2
    with pytest.raises(ValueError, match="Multiple positional arguments"):
        expr(5, 3)


def test_multiple_unnamed_variables():
    variable1 = Variable()
    variable2 = Variable()
    expr = variable1 + variable2
    assert expr(5) == 10
