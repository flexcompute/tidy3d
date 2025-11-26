from __future__ import annotations

import autograd
import numpy as np
import pytest
from autograd.test_util import check_grads

from tidy3d.plugins.autograd.invdes.filters import ConicFilter
from tidy3d.plugins.autograd.invdes.projections import smoothed_projection, tanh_projection


def create_circle(nx, ny, radius):
    # 1. Initialize array
    arr = np.zeros((nx, ny), dtype=float)

    # 2. Logic to create circle
    center_x, center_y = nx / 2, ny / 2
    x = np.arange(nx)
    y = np.arange(ny)
    # Note: indexing='ij' ensures x corresponds to rows (nx) and y to cols (ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    arr[distance <= radius] = 1
    return arr


def test_smoothed_projection_beta_inf():
    nx, ny = 50, 50
    radius = 10

    arr = create_circle(nx, ny, radius)

    filter = ConicFilter(kernel_size=5)
    arr_filtered = filter(arr)

    result = smoothed_projection(
        array=arr_filtered,
        beta=np.inf,
        eta=0.5,
    )
    assert not np.any(np.isinf(result) | np.isnan(result))
    assert np.isclose(result[round(nx / 2), round(ny / 2)], 1)
    assert np.isclose(result[0, -1], 0)
    assert np.isclose(result[0, 0], 0)
    assert np.isclose(result[-1, 0], 0)
    assert np.isclose(result[-1, -1], 0)

    # fully discrete input should lead to fully discrete output
    discrete_result = smoothed_projection(
        array=arr,
        beta=np.inf,
        eta=0.5,
    )
    assert np.all(np.isclose(discrete_result, 0) | np.isclose(discrete_result, 1))


def test_smoothed_projection_beta_non_inf():
    nx, ny = 50, 50
    radius = 10

    arr = create_circle(nx, ny, radius)

    # fully discrete input should still be fully discrete output
    discrete_result = smoothed_projection(
        array=arr,
        beta=1.0,
        eta=0.5,
    )
    assert np.all(np.isclose(discrete_result, 0) | np.isclose(discrete_result, 1))

    filter = ConicFilter(kernel_size=11)
    arr_filtered = filter(arr)

    smooth_result = smoothed_projection(
        array=arr_filtered,
        beta=1.0,
        eta=0.5,
    )
    # for sufficiently smooth input, the result should be the same as tanh projection
    tanh_result = tanh_projection(
        array=arr_filtered,
        beta=1.0,
        eta=0.5,
    )
    assert np.isclose(smooth_result, tanh_result, rtol=0, atol=1e-4).all()


def test_smoothed_projection_initialization():
    # test that for initialization at eta=0.5, projection returns simply 0.5
    arr = np.zeros((5, 5), dtype=float) + 0.5
    result = smoothed_projection(array=arr, beta=1.0, eta=0.5)
    assert np.all(np.isclose(result, 0.5))


def test_projection_gradient():
    # test that gradient is finite
    arr = np.zeros((5, 5), dtype=float) + 0.5

    def _helper_fn(x):
        return smoothed_projection(array=x, beta=1.0, eta=0.5).mean()

    val, grad = autograd.value_and_grad(_helper_fn)(arr)
    assert val == 0.5
    assert np.all(~(np.isnan(grad) | np.isinf(grad)))


@pytest.mark.parametrize("beta", [0, 5, np.inf])
@pytest.mark.parametrize("size", [30, 50])
@pytest.mark.parametrize("radius", [10, 15, 20])
@pytest.mark.parametrize("smoothing_radius", [3, 5, 7])
def test_projection_gradient_correctness(beta, size, radius, smoothing_radius):
    arr = create_circle(size, size, radius)
    filter = ConicFilter(kernel_size=smoothing_radius)
    arr = filter(arr)

    def _helper_fn(x):
        return smoothed_projection(x, beta=beta, eta=0.5).mean()

    check_grads(_helper_fn, modes=["fwd", "rev"], order=2)(arr)
