import pytest
import pydantic
import numpy as np

from tidy3d.material_library.material_library import material_library
from tidy3d.material_library.parametric_materials import (
    GRAPHENE_FIT_FREQ_MIN,
    GRAPHENE_FIT_FREQ_MAX,
    GRAPHENE_FIT_NUM_FREQS,
    GRAPHENE_FIT_ATOL,
)
from tidy3d.material_library.parametric_materials import Graphene
import tidy3d as td

from numpy.random import default_rng

# bounds for MU_C
GRAPHENE_MU_C_MIN = 0
GRAPHENE_MU_C_MAX = 3
# bounds for temp
GRAPHENE_TEMP_MIN = 100
GRAPHENE_TEMP_MAX = 1000
# bounds for gamma
GRAPHENE_GAMMA_MIN = 0
GRAPHENE_GAMMA_MAX = 0.03


def test_graphene_defaults():
    freqs = np.linspace(GRAPHENE_FIT_FREQ_MIN, GRAPHENE_FIT_FREQ_MAX, GRAPHENE_FIT_NUM_FREQS)
    graphene = Graphene()
    sigma1 = graphene.medium.sigma_model(freqs)
    sigma2 = graphene.numerical_conductivity(freqs)


@pytest.mark.parametrize("rng_seed", np.arange(0, 15))
def test_graphene(rng_seed):
    """test graphene for range of physical parameters"""
    rng = default_rng(rng_seed)
    gamma_min = GRAPHENE_GAMMA_MIN
    gamma_max = GRAPHENE_GAMMA_MAX
    mu_min = GRAPHENE_MU_C_MIN
    mu_max = GRAPHENE_MU_C_MAX
    temp_min = GRAPHENE_TEMP_MIN
    temp_max = GRAPHENE_TEMP_MAX

    freqs = np.linspace(GRAPHENE_FIT_FREQ_MIN, GRAPHENE_FIT_FREQ_MAX, GRAPHENE_FIT_NUM_FREQS)

    gamma = gamma_min + (gamma_max - gamma_min) * rng.random()
    mu_c = mu_min + (mu_max - mu_min) * rng.random()
    temp = temp_min + (temp_max - temp_min) * rng.random()

    print(f"Graphene(gamma='{gamma:.6f}', mu_c='{mu_c:.2f}', temp='{temp:.0f}')")

    graphene = Graphene(gamma=gamma, mu_c=mu_c, temp=temp)
    sigma1 = graphene.medium.sigma_model(freqs)
    sigma2 = graphene.numerical_conductivity(freqs)

    assert np.allclose(sigma1, sigma2, rtol=0, atol=GRAPHENE_FIT_ATOL)

    graphene = Graphene(gamma=gamma, mu_c=mu_c, temp=temp, include_interband=False)
    sigma1 = graphene.medium.sigma_model(freqs)
    sigma2 = graphene.intraband_drude.sigma_model(freqs)
    assert np.allclose(sigma1, sigma2, rtol=0, atol=GRAPHENE_FIT_ATOL)
