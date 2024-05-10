import pytest
from autograd.test_util import check_grads

from tidy3d.plugins.autograd.primitives import gaussian_filter


@pytest.mark.parametrize("size", [10, 11])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("sigma", [1, 2])
@pytest.mark.parametrize(
    "mode",
    [
        "constant",
        "reflect",
        "wrap",
        pytest.param("nearest", marks=pytest.mark.skip(reason="Grads not implemented.")),
        pytest.param("mirror", marks=pytest.mark.skip(reason="Grads not implemented.")),
    ],
)
def test_gaussian_filter_grad(rng, size, ndim, sigma, mode):
    x = rng.random((size,) * ndim)
    check_grads(lambda x: gaussian_filter(x, sigma=sigma, mode=mode), modes=["rev"], order=2)(x)
