import matplotlib.pyplot as plt
import numpy as np
import pytest
import tidy3d as td
from tidy3d.log import DEFAULT_LEVEL, set_logging_console, set_logging_level


@pytest.fixture
def rng():
    seed = 36523525
    return np.random.default_rng(seed)


@pytest.fixture(autouse=True, scope="module")
def close_matplotlib():
    plt.close()


@pytest.fixture(autouse=True, scope="module")
def reset_logger():
    """Reset logger state at the beginning of each module."""
    if "console" in td.log.handlers:
        del td.log.handlers["console"]
    set_logging_console()
    set_logging_level(DEFAULT_LEVEL)
