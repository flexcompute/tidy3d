import autograd.numpy as np
import pytest
from autograd import grad as grad_ag
from autograd import value_and_grad as value_and_grad_ag
from numpy.testing import assert_allclose
from tidy3d.components.data.data_array import DataArray
from tidy3d.plugins.autograd import grad, value_and_grad


@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize("has_aux", [True, False])
def test_grad(rng, argnum, has_aux):
    """Test the custom value_and_grad function against autograd's implementation"""
    x = rng.random(10)
    y = rng.random(10)
    aux_val = "aux"

    def f(x, y):
        ret = DataArray(x * y).sum()  # still DataArray
        if has_aux:
            return ret, aux_val
        return ret

    grad_fun = grad(f, argnum=argnum, has_aux=has_aux)
    grad_fun_ag = grad_ag(
        lambda x, y: f(x, y)[0].item() if has_aux else f(x, y).item(), argnum=argnum
    )

    if has_aux:
        g, aux = grad_fun(x, y)
        assert aux == aux_val
    else:
        g = grad_fun(x, y)
    g_ag = grad_fun_ag(x, y)

    assert_allclose(g, g_ag)


@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize("has_aux", [True, False])
def test_value_and_grad(rng, argnum, has_aux):
    """Test the custom value_and_grad function against autograd's implementation"""
    x = rng.random(10)
    y = rng.random(10)
    aux_val = "aux"

    def f(x, y):
        ret = DataArray(np.linalg.norm(x * y)).sum()  # still DataArray
        if has_aux:
            return ret, aux_val
        return ret

    vg_fun = value_and_grad(f, argnum=argnum, has_aux=has_aux)
    vg_fun_ag = value_and_grad_ag(
        lambda x, y: f(x, y)[0].item() if has_aux else f(x, y).item(), argnum=argnum
    )

    if has_aux:
        (v, g), aux = vg_fun(x, y)
        assert aux == aux_val
    else:
        v, g = vg_fun(x, y)
    v_ag, g_ag = vg_fun_ag(x, y)

    assert_allclose(v, v_ag)
    assert_allclose(g, g_ag)
