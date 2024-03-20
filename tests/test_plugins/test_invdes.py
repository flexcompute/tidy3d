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


td.config.logging_level = "ERROR"

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
)

mnt = td.FieldMonitor(
    center=(L_SIM / 3.0, 0, 0), size=(0, td.inf, td.inf), freqs=[FREQ0], name=MNT_NAME
)

# class custom_transformation(tdi.Transformation):
#     def evaluate(self, data: jnp.ndarray) -> jnp.ndarray:
#         return data / jnp.mean(data)

# class custom_penalty(tdi.Penalty):
#     def evaluate(self, data: jnp.ndarray) -> float:
#         return jnp.mean(data)


def test_design_region():
    """make a design region and test some functions."""
    design_region = tdi.TopologyDesignRegion(
        size=(0.4 * L_SIM, 0.4 * L_SIM, 0.4 * L_SIM),
        center=(0, 0, 0),
        eps_bounds=(1.0, 4.0),
        symmetry=(0, 1, -1),
        params_shape=PARAMS_SHAPE,
        transformations=[
            tdi.CircularFilter(radius=0.2, design_region_dl=0.1),
            tdi.BinaryProjector(beta=2.0, vmin=0.0, vmax=1.0),
            # custom_transformation, # TODO: fix these
            tdi.ConicFilter(radius=0.2, design_region_dl=0.1),
        ],
        penalties=[
            tdi.ErosionDilationPenalty(length_scale=0.2, pixel_size=0.1),
            # custom_penalty, # TODO: fix these
        ],
        penalty_weights=[0.2],
    )

    # test some design region functions
    design_region.material_density(PARAMS_0)
    design_region.penalty_value(PARAMS_0)
    design_region.updated_copy(penalty_weights=None).penalty_value(PARAMS_0)
    design_region_no_penalties = design_region.updated_copy(penalties=[], penalty_weights=None)
    with pytest.raises(ValueError):
        design_region_no_penalties._penalty_weights
    assert design_region_no_penalties.penalty_value(PARAMS_0) == 0.0

    return design_region


def test_optimizer():
    """Make an optimizer"""
    optimizer = tdi.Optimizer(
        learning_rate=0.2,
        num_steps=3,
    )
    return optimizer


def test_invdes():
    """make an inverse design"""
    invdes = tdi.InverseDesign(
        simulation=simulation,
        design_region=test_design_region(),
        output_monitors=[mnt],
        optimizer=test_optimizer(),
        params0=np.random.random(PARAMS_SHAPE).tolist(),
        history_save_fname="tests/data/invdes_history.pkl",
    )
    return invdes


def post_process_fn(sim_data: tda.JaxSimulationData, scale: float = 2.0) -> float:
    """Define a postprocessing function"""
    intensity = sim_data.get_intensity(MNT_NAME)
    return scale * jnp.sum(intensity.values)


def test_run(use_emulated_run):
    """Test running the optimization defined in the ``InverseDesign`` object."""
    invdes = test_invdes()
    result = invdes.run(post_process_fn, task_name="blah")
    return invdes, result


def test_continue_run(use_emulated_run):
    """Test continuing an already run inverse design."""
    invdes, result_orig = test_run(use_emulated_run)
    result_full = invdes.continue_run(result_orig, post_process_fn, task_name="blah")
    num_steps_orig = len(result_orig.history["params"])
    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == num_steps_orig + invdes.optimizer.num_steps
    ), "wrong number of elements in the combined run history."


def test_result(use_emulated_run, tmp_path):
    """Test methods of the ``OptimizeResult`` object."""
    _, result = test_run(use_emulated_run)

    with pytest.raises(KeyError):
        _ = result.get_final("blah")

    val_final1 = result.final["params"]
    val_final2 = result.get_final("params")
    assert np.allclose(val_final1, val_final2)

    result.plot_optimization()

    gds_fname = str(tmp_path / "sim_final.gds")
    gds_layer_dtype_map = {td.Medium(permittivity=4.0): (2, 1), td.Medium(): (1, 0)}
    result.to_gds_file(gds_fname, z=0, gds_layer_dtype_map=gds_layer_dtype_map)

    sim_data_final = result.sim_data_final(task_name="final")
