from __future__ import annotations

import os
import sys
from pathlib import Path

import autograd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pytest
from autograd.test_util import check_grads
from autograd.wrap_util import unary_to_nary

import tidy3d as td
from tidy3d.log import DEFAULT_LEVEL, set_logging_console, set_logging_level


@pytest.fixture
def rng():
    seed = 36523525
    return np.random.default_rng(seed)


@pytest.fixture(autouse=True)
def close_matplotlib():
    plt.close("all")


@pytest.fixture(autouse=True, scope="module")
def reset_logger():
    """Reset logger state at the beginning of each module."""
    if "console" in td.log.handlers:
        del td.log.handlers["console"]
    set_logging_console()
    set_logging_level(DEFAULT_LEVEL)


@pytest.fixture(autouse=True)
def clear_log_cache():
    """Ensure log-once cache does not leak between tests."""
    td.log._static_cache.clear()
    yield


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
        if os.getenv("RUNNER_ENVIRONMENT") == "self-hosted":
            MAX_SELF_HOSTED_CORES = 8
            return min(MAX_SELF_HOSTED_CORES, cores)
        return cores
    return max(1, cores - 1)


@pytest.fixture
def mpl_config_noninteractive():
    """Configure matplotlib non-interactive backend for all tests in this module."""
    original_backend = mpl.get_backend()
    mpl.use("Agg")
    yield
    plt.close("all")
    mpl.use(original_backend)


@pytest.fixture
def mpl_config_interactive():
    """Configure matplotlib interactive backend for all tests in this module."""
    original_backend = mpl.get_backend()
    mpl.use("TkAgg")
    yield
    mpl.use(original_backend)


@pytest.fixture
def dir_name(request):
    return request.param


@pytest.fixture
def create_directory(dir_name):
    if dir_name is not None:
        directory = Path(dir_name).mkdir(parents=True, exist_ok=True)


class OutputTee:
    """Helper class to write to two streams at once."""

    def __init__(self, original_stdout, stderr):
        self.original_stdout = original_stdout
        self.stderr = stderr

    def write(self, message):
        # Write to the original stdout (so pytest capture works)
        self.original_stdout.write(message)
        # Write to stderr (so you see it immediately)
        # We generally want to flush immediately for debug prints
        self.stderr.write(message)
        self.stderr.flush()

    def flush(self):
        self.original_stdout.flush()
        self.stderr.flush()

    def __getattr__(self, attr):
        # Pass any other method calls (like isatty) to the original stream
        return getattr(self.original_stdout, attr)


@pytest.fixture()
def redirect_stdout_to_stderr(request):
    """
    Automatically wraps sys.stdout to write to both stdout and stderr.
    This ensures output is visible during parallel execution without
    breaking pytest capturing.
    """
    # 1. Capture the current stdout (which might be pytest's capture buffer)
    original_stdout = sys.stdout

    # 2. Replace stdout with our Tee
    sys.stdout = OutputTee(original_stdout, sys.stderr)

    yield

    # 3. Restore original stdout after test finishes
    sys.stdout = original_stdout
