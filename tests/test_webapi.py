"""Tests webapi bits that dont require authentication."""
import tidy3d as td
import tidy3d.web as web


def make_sim():
    """Makes a simulation."""
    return td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1.0), run_time=1e-12)


def test_job():
    """tests creation of a job."""
    sim = make_sim()
    j = web.Job(simulation=sim, task_name="test")


def test_batch():
    """tests creation of a batch."""
    sim = make_sim()
    b = web.Batch(simulations={"test": sim})
