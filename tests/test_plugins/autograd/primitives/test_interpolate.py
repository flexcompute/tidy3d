from __future__ import annotations

import autograd.numpy as np
import numpy.testing as npt
import pytest
from autograd import grad
from autograd.test_util import check_grads

from tidy3d.plugins.autograd import interpolate_spline

from ....utils import AssertLogLevel


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("num_points", [5, 10])
@pytest.mark.parametrize(
    "endpoint_derivs",
    [
        (None, None),
        (0.0, None),
        (None, 0.0),
        (-1.0, 1.0),
    ],
)
@pytest.mark.parametrize(
    "x_distribution",
    [
        (lambda n, rng: np.linspace(0, 1, n)),  # uniform ascending
        (lambda n, rng: np.sort(rng.random(n))),  # random ascending
        (lambda n, rng: np.linspace(1, 0, n)),  # uniform descending
    ],
)
def test_interpolate_spline_grads(rng, order, num_points, endpoint_derivs, x_distribution):
    """Test gradients of interpolate_spline function."""
    x = x_distribution(num_points, rng)
    y = rng.random(num_points)

    check_grads(interpolate_spline, modes=["rev"], order=1, argnum=1)(
        x, y, num_points, order, endpoint_derivs
    )


def test_interpolate_spline_grads_kwargs(rng):
    """Test interpolate_spline function can be called with kwargs."""
    x = np.linspace(0, 1, 10)
    y = rng.random(x.size)
    # this should not error
    grad(
        lambda y_: interpolate_spline(
            x_points=x,
            y_points=y_,
            num_points=10,
            order=3,
            endpoint_derivatives=(None, None),
        )[1][0]
    )(y)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("x", [np.linspace(0, 1, 10), np.linspace(1, 0, 10)])
def test_interpolate_spline_vals(rng, order, x):
    """Test values of interpolate_spline function (ascending and descending)."""
    y = rng.random(10)

    x_interp, y_interp = interpolate_spline(x, y, len(x), order=order)

    npt.assert_allclose(x_interp, x)
    npt.assert_allclose(y_interp, y)


@pytest.mark.parametrize("invalid_order", [0, 4])
def test_interpolate_spline_invalid_order(rng, invalid_order):
    """Test that interpolate_spline raises exception for invalid orders."""
    x = rng.random(10)
    y = rng.random(10)

    with pytest.raises(NotImplementedError):
        interpolate_spline(x, y, 10, order=invalid_order)


@pytest.mark.parametrize(
    "order, endpoint_derivs",
    [
        (1, (1.0, None)),
        (1, (None, 1.0)),
        (1, (1.0, 1.0)),
        (2, (None, 1.0)),
        (2, (1.0, 1.0)),
    ],
)
def test_interpolate_spline_warnings(rng, order, endpoint_derivs):
    """Test that interpolate_spline raises warnings for unused endpoint derivatives."""
    x = np.linspace(0, 1, 10)
    y = rng.random(10)

    with AssertLogLevel("WARNING", contains_str="ignored"):
        interpolate_spline(x, y, 10, order, endpoint_derivs)


def test_interpolate_spline_shape_validation(rng):
    """Test that interpolate_spline validates that x and y have the same shape."""
    x = np.linspace(0, 1, 10)
    y = rng.random(11)  # Different length than x

    with pytest.raises(ValueError, match="shape"):
        interpolate_spline(x, y, 10, order=3)
