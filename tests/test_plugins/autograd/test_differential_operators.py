import autograd.numpy as np
from autograd import value_and_grad as value_and_grad_ag
from numpy.testing import assert_allclose
from tidy3d.plugins.autograd.differential_operators import value_and_grad


def test_value_and_grad(rng):
    """Test the custom value_and_grad function against autograd's implementation"""
    x = rng.random(10)
    aux_val = "aux"

    vg_fun = value_and_grad(lambda x: (np.linalg.norm(x), aux_val), has_aux=True)
    vg_fun_ag = value_and_grad_ag(lambda x: np.linalg.norm(x))

    (v, g), aux = vg_fun(x)
    v_ag, g_ag = vg_fun_ag(x)

    # assert that values and gradients match
    assert_allclose(v, v_ag)
    assert_allclose(g, g_ag)

    # check that auxiliary output is correctly returned
    assert aux == aux_val
