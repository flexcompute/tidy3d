import os

import autograd
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pytest
import tidy3d as td
from autograd.test_util import check_grads
from autograd.wrap_util import unary_to_nary
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


@pytest.fixture
def check_grads_with_tolerance(monkeypatch):
    @unary_to_nary
    def check_grads_with_tolerance_(f, x, modes=None, order=2, tol=1e-6, rtol=1e-6):
        """Wrap autograd's check_grads function so we can override the hardcoded tolerances."""
        if not modes:
            modes = ["fwd", "rev"]

        with monkeypatch.context() as m:
            m.setattr(autograd.test_util, "TOL", tol)
            m.setattr(autograd.test_util, "RTOL", rtol)
            check_grads(f, modes=modes, order=order)(x)

    return check_grads_with_tolerance_


def pytest_xdist_auto_num_workers(config):
    """Return the number of workers for pytest-xdist auto mode based on CPU cores and available memory.

    Each worker requires approximately 1GB of memory, so the number of workers is limited by both
    the number of physical CPU cores and available system memory.
    """
    try:
        cores = psutil.cpu_count(logical=False)
    except Exception:
        cores = os.cpu_count()

    available_mem_gb = psutil.virtual_memory().available / (1024**3)

    # allow 1.2gb per core to provide some buffer
    mem_limited_cores = int(available_mem_gb / 1.2)

    cores = min(cores, mem_limited_cores)

    if os.getenv("GITHUB_ACTIONS"):
        return cores
    return max(1, cores - 1)
