# Test the inverse design plugin

import pytest

import numpy as np
import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda
import tidy3d.plugins.invdes as tdi
import matplotlib.pyplot as plt

from .test_adjoint import use_emulated_run
from ..utils import run_emulated

FREQ0 = 1e14
L_SIM = 2.0
MNT_NAME = "mnt_name"
PARAMS_SHAPE = (18, 19, 20)
PARAMS_0 = np.random.random(PARAMS_SHAPE)
HISTORY_FNAME = "tests/data/invdes_history.pkl"

td.config.logging_level = "ERROR"

mnt = td.FieldMonitor(
    center=(L_SIM / 3.0, 0, 0), size=(0, td.inf, td.inf), freqs=[FREQ0], name=MNT_NAME
)

simulation = td.Simulation(
    size=(L_SIM, L_SIM, L_SIM),
    grid_spec=td.GridSpec.auto(wavelength=td.C_0 / FREQ0),
    sources=[
        td.PointDipole(
            center=(0, 0, 0),
            source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FREQ0 / 10),
            polarization="Ez",
        )
    ],
    run_time=1e-12,
    monitors=[mnt],
)


def test_design_region():
    """make a design region and test some functions."""
    design_region = tdi.TopologyDesignRegion(
        size=(0.4 * L_SIM, 0.4 * L_SIM, 0.4 * L_SIM),
        center=(0, 0, 0),
        eps_bounds=(1.0, 4.0),
        params_shape=PARAMS_SHAPE,
        pixel_size=0.1,
        transformations=[tdi.FilterProject(radius=0.2, beta=2.0)],
        penalties=[
            tdi.ErosionDilationPenalty(length_scale=0.2),
        ],
    )

    # test some design region functions
    design_region.material_density(PARAMS_0)
    design_region.penalty_value(PARAMS_0)
    return design_region


def post_process_fn(sim_data: tda.JaxSimulationData, **kwargs) -> float:
    """Define a post-processing function"""
    intensity = sim_data.get_intensity(MNT_NAME)
    return jnp.sum(intensity.values)


def test_invdes():
    """make an inverse design"""
    invdes = tdi.InverseDesign(
        simulation=simulation,
        design_region=test_design_region(),
        # output_monitor_names=[MNT_NAME],
        post_process_fn=post_process_fn,
        task_name="test",
    )
    return invdes


def test_optimizer():
    """Make an optimizer"""
    design = test_invdes()
    optimizer = tdi.AdamOptimizer(
        design=design,
        history_save_fname=HISTORY_FNAME,
        learning_rate=0.2,
        num_steps=2,
    )
    return optimizer


def test_run(use_emulated_run):
    """Test running the optimization defined in the ``InverseDesign`` object."""
    optimizer = test_optimizer()
    result = optimizer.run(params0=PARAMS_0)
    return result


def test_continue_run(use_emulated_run):
    """Test continuing an already run inverse design."""
    result_orig = test_run(use_emulated_run)
    optimizer = test_optimizer()
    result_full = optimizer.continue_run(result=result_orig)

    num_steps_orig = len(result_orig.history["params"])
    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == num_steps_orig + optimizer.num_steps
    ), "wrong number of elements in the combined run history."


def test_complete_run_from_file(use_emulated_run):
    """Test continuing an already run inverse design from file."""
    result_orig = test_run(use_emulated_run)
    optimizer_orig = test_optimizer()
    optimizer = optimizer_orig.updated_copy(num_steps=optimizer_orig.num_steps + 1)
    result_full = optimizer.complete_run_from_file(HISTORY_FNAME)

    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == optimizer_orig.num_steps + optimizer.num_steps
    ), "wrong number of elements in the combined run history."


def test_result(use_emulated_run, tmp_path):
    """Test methods of the ``InverseDesignResult`` object."""
    result = test_run(use_emulated_run)

    with pytest.raises(KeyError):
        _ = result.get_final("blah")

    val_final1 = result.final["params"]
    val_final2 = result.get_final("params")
    assert np.allclose(val_final1, val_final2)

    result.plot_optimization()

    gds_fname = str(tmp_path / "sim_final.gds")
    gds_layer_dtype_map = {td.Medium(permittivity=4.0): (2, 1), td.Medium(): (1, 0)}
    result.sim_final.to_gds_file(gds_fname, z=0, gds_layer_dtype_map=gds_layer_dtype_map)

    sim_data_final = result.sim_data_final(task_name="final")


def test_invdes_io(tmp_path, use_emulated_run):
    """Test saving a loading invdes instances to file."""

    result = test_run(use_emulated_run)
    optimizer = test_optimizer()
    design = optimizer.design

    for obj in (design, optimizer, result):
        obj.json()

        path = str(tmp_path / "obj.pkl")
        obj.to_file(path)

        obj2 = obj.from_file(path)

        assert obj2.json() == obj.json()
