from __future__ import annotations

import autograd.numpy as anp
import torch
from autograd import elementwise_grad
from numpy.testing import assert_allclose

from tidy3d.plugins.pytorch.wrapper import to_torch


def test_to_torch_no_kwargs(rng):
    x_np = rng.uniform(-1, 1, 10).astype("f4")
    x_torch = torch.tensor(x_np, requires_grad=True)

    def f_np(x):
        return x * anp.sin(x) ** 2

    f_torch = to_torch(f_np)

    val = f_torch(x_torch)
    val.backward(torch.ones(x_torch.shape))

    grad = x_torch.grad.numpy()
    expected_grad = elementwise_grad(f_np)(x_np)

    assert_allclose(grad, expected_grad)


def test_to_torch_with_kwargs(rng):
    x_np = rng.uniform(-1, 1, 10).astype("f4")
    y_np = rng.uniform(-1, 1, 10).astype("f4")
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)

    def f_np(x, y):
        return y * anp.sin(x) ** 2

    f_torch = to_torch(f_np)

    val = f_torch(y=y_torch, x=x_torch)
    val.backward(torch.ones(x_torch.shape))

    grad = x_torch.grad.numpy(), y_torch.grad.numpy()
    expected_grad = elementwise_grad(f_np, argnum=[0, 1])(x_np, y_np)

    assert_allclose(grad, expected_grad)


def test_to_torch_array_valued_function(rng):
    """Test that gradients are computed correctly for functions returning arrays with different shapes than input."""
    x_np = rng.uniform(-1, 1, (2, 2)).astype("f4")
    x_torch = torch.tensor(x_np, requires_grad=True)

    # define a function that returns a different shape than input
    # this function maps (2,2) -> (2,3)
    def f_np(x):
        return anp.stack([x.sum(axis=1), x.mean(axis=1) * 2, x[:, 0] * x[:, 1]], axis=1)

    f_torch = to_torch(f_np)

    output = f_torch(x_torch)
    assert output.shape == (2, 3)

    # create upstream gradient (simulating backprop from a loss)
    grad_output = torch.ones_like(output)  # shape (2, 3)

    output.backward(grad_output)

    h = 1e-5
    expected_grad = anp.zeros_like(x_np)

    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            x_plus = x_np.copy()
            x_plus[i, j] += h
            x_minus = x_np.copy()
            x_minus[i, j] -= h

            f_plus = f_np(x_plus)
            f_minus = f_np(x_minus)

            expected_grad[i, j] = anp.sum((f_plus - f_minus) / (2 * h))

    computed_grad = x_torch.grad.numpy()
    assert_allclose(computed_grad, expected_grad, rtol=1e-3, atol=1e-3)
