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
