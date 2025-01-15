import numpy as np
import pytest


@pytest.fixture
def rng():
    seed = 36523525
    return np.random.default_rng(seed)
