"""Tests webapi bits that dont require authentication."""
import pytest

import tidy3d as td
import tidy3d.web as web
from tidy3d.log import DataError

from .utils import clear_tmp


def make_sim():
    """Makes a simulation."""
    return td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1.0), run_time=1e-12)


def _test_job():
    """tests creation of a job."""
    sim = make_sim()
    j = web.Job(simulation=sim, task_name="test")


def _test_batch():
    """tests creation of a batch."""
    sim = make_sim()
    b = web.Batch(simulations={"test": sim})


@clear_tmp
def _test_batchdata_load():
    """Tests loading of a batch data from file."""
    sim = make_sim()
    b = web.Batch(simulations={"test": sim})
    b.to_file("tests/tmp/batch.json")
    with pytest.raises(DataError):
        web.BatchData.load(path_dir="tests/tmp")
