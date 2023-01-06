import pytest
import numpy as np

from tidy3d.material_library.material_library import material_library
from tidy3d.material_library.parametric_materials import Graphene
import tidy3d as td

from numpy.random import default_rng


# tolerance
ATOL = 1e-5


@pytest.mark.parametrize("rng_seed", np.arange(0, 4))
def test_graphene(rng_seed):
    """test graphene for range of physical parameters"""
    rng = default_rng(rng_seed)
    gamma_min = 1e-5
    gamma_max = 1e-2
    mu_min = 0
    mu_max = 0.5
    temp_min = 200
    temp_max = 1000

    freqs = np.linspace(1e12, 1e15, 1000)

    gamma = gamma_min + (gamma_max - gamma_min) * rng.random()
    mu_c = mu_min + (mu_max - mu_min) * rng.random()
    temp = temp_min + (temp_max - temp_min) * rng.random()

    graphene = Graphene(gamma=gamma, mu_c=mu_c, temp=temp)
    sigma1 = graphene.medium.sigma_model(freqs)
    sigma2 = graphene.numerical_conductivity(freqs)

    assert np.allclose(sigma1, sigma2, rtol=0, atol=ATOL)

    graphene = Graphene(gamma=gamma, mu_c=mu_c, temp=temp, include_interband=False)
    sigma1 = graphene.medium.sigma_model(freqs)
    sigma2 = graphene.intraband_drude.sigma_model(freqs)
    assert np.allclose(sigma1, sigma2, rtol=0, atol=ATOL)
