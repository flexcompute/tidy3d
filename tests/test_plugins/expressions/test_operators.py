import operator

import numpy as np
import pytest
from tidy3d.plugins.expressions.operators import (
    Abs,
    Add,
    Divide,
    FloorDivide,
    MatMul,
    Modulus,
    Multiply,
    Negate,
    Power,
    Subtract,
)

UNARY_OPS = [
    (Abs, operator.abs),
    (Negate, operator.neg),
]

BINARY_OPS = [
    (Add, operator.add),
    (Subtract, operator.sub),
    (Multiply, operator.mul),
    (Divide, operator.truediv),
    (FloorDivide, operator.floordiv),
    (Modulus, operator.mod),
    (Power, operator.pow),
    (MatMul, operator.matmul),
]


@pytest.mark.parametrize("tidy3d_op, python_op", BINARY_OPS)
@pytest.mark.parametrize(
    "x, y", [(1, 2), (2.5, 3.7), (1 + 2j, 3 - 4j), (np.array([1, 2, 3]), np.array([4, 5, 6]))]
)
def test_binary_operators(tidy3d_op, python_op, x, y):
    if any(isinstance(p, complex) for p in (x, y)) and any(
        tidy3d_op == op for op in (FloorDivide, Modulus)
    ):
        pytest.skip("operation undefined for complex inputs")
    if tidy3d_op == MatMul and (np.isscalar(x) or np.isscalar(y)):
        pytest.skip("matmul operation undefined for scalar inputs")

    tidy3d_result = tidy3d_op(left=x, right=y).evaluate(x)
    python_result = python_op(x, y)

    np.testing.assert_allclose(tidy3d_result, python_result)


@pytest.mark.parametrize("tidy3d_op, python_op", UNARY_OPS)
@pytest.mark.parametrize("x", [1, 2.5, 1 + 2j, np.array([1 + 2j, 3 - 4j, 5 + 6j])])
def test_unary_operators(tidy3d_op, python_op, x):
    tidy3d_result = tidy3d_op(operand=x).evaluate(x)
    python_result = python_op(x)

    np.testing.assert_allclose(tidy3d_result, python_result)
