from __future__ import annotations

import autograd.numpy as np
import pytest
from autograd.test_util import check_grads

from tidy3d.plugins.autograd.primitives import gaussian_filter


@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("sigma", [1, 2])
@pytest.mark.parametrize("mode", ["constant", "nearest", "mirror", "reflect", "wrap"])
def test_gaussian_filter_grad(rng, size, ndim, sigma, mode):
    x = rng.random((size,) * ndim)
    check_grads(lambda x: gaussian_filter(x, sigma=sigma, mode=mode), modes=["rev"], order=1)(x)


@pytest.mark.parametrize("shape, axis", [((100,), -1), ((10, 12), 0), ((10, 12), 1)])
@pytest.mark.parametrize("period", [np.pi, 2 * np.pi])
@pytest.mark.parametrize("discont", [None, 0.6])
def test_unwrap_grad(rng, shape, axis, period, discont):
    """Test the gradient of the unwrap function with various arguments."""
    if discont is not None:
        # discont must be > period / 2 to have an effect
        discont = discont * period

    x = rng.uniform(-4 * period, 4 * period, shape)

    check_grads(
        lambda x: np.unwrap(x, discont=discont, axis=axis, period=period), modes=["fwd", "rev"]
    )(x)
