# Test the inverse design plugin

import pytest
import builtins

import numpy as np
import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda
import tidy3d.plugins.invdes as tdi
import matplotlib.pyplot as plt

from .test_adjoint import use_emulated_run
from ..utils import run_emulated, log_capture, assert_log_level, AssertLogLevel

FREQ0 = 1e14
L_SIM = 2.0
MNT_NAME = "mnt_name"
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


def make_design_region():
    return tdi.TopologyDesignRegion(
        size=(0.4 * L_SIM, 0.4 * L_SIM, 0.4 * L_SIM),
        center=(0, 0, 0),
        eps_bounds=(1.0, 4.0),
        transformations=[tdi.FilterProject(radius=0.2, beta=2.0)],
        penalties=[
            tdi.ErosionDilationPenalty(length_scale=0.2),
        ],
        pixel_size=1.0 / 4.0,
    )


def test_region_params():
    """make a design region and test some functions."""

    design_region = make_design_region()

    PARAMS_0 = np.random.random(design_region.params_shape)
    PARAMS_0 = design_region.params_random
    PARAMS_0 = design_region.params_ones
    PARAMS_0 = design_region.params_zeros


def test_region_penalties():
    region = make_design_region()

    PARAMS_0 = region.params_random

    # test some design region functions
    region.material_density(PARAMS_0)
    region.penalty_value(PARAMS_0)

    region.updated_copy(penalties=[]).penalty_value(PARAMS_0)


def test_region_to_structure():
    region = make_design_region()

    PARAMS_0 = region.params_ones

    _ = region.to_jax_structure(PARAMS_0)
    _ = region.to_structure(PARAMS_0)


def test_region_params_bounds():
    region = make_design_region()

    PARAMS_0 = region.params_ones

    with pytest.raises(ValueError):
        region.penalty_value(2 * PARAMS_0)

    with pytest.raises(ValueError):
        region.penalty_value(-1 * PARAMS_0)


def test_region_inf_size():
    region = make_design_region()
    inf_size = list(region.size)
    inf_size[1] = td.inf
    region = region.updated_copy(size=inf_size)
    params_0_inf = region.params_zeros
    _ = region.to_structure(params_0_inf)


def test_penalty_pixel_size(log_capture):
    with AssertLogLevel(log_capture, "WARNING"):
        _ = tdi.ErosionDilationPenalty(pixel_size=1.0, length_scale=2.0)

    _ = tdi.ErosionDilationPenalty(length_scale=2.0)


def post_process_fn(sim_data: tda.JaxSimulationData, **kwargs) -> float:
    """Define a post-processing function"""
    intensity = sim_data.get_intensity(MNT_NAME)
    return jnp.sum(intensity.values)


def post_process_fn_kwargless(sim_data: tda.JaxSimulationData) -> float:
    """Define a post-processing function"""
    intensity = sim_data.get_intensity(MNT_NAME)
    return jnp.sum(intensity.values)


def make_invdes():
    """make an inverse design"""
    return tdi.InverseDesign(
        simulation=simulation,
        design_region=make_design_region(),
        post_process_fn=post_process_fn,
        task_name="test",
    )


class MockDataArray:
    values = jnp.linspace(0, 1, 10)


class MockSimData:
    def get_intensity(self, name: str) -> jnp.ndarray:
        return MockDataArray()


def test_invdes_kwarglfull():
    """Test postprocess function defined with kwargs works."""
    invdes = make_invdes()
    invdes.post_process_fn(MockSimData(), kwarg1="hi")


def test_invdes_kwargless():
    """Test postprocess function defined without kwargs still works."""
    invdes = make_invdes()
    invdes = invdes.updated_copy(post_process_fn=post_process_fn_kwargless)
    invdes.post_process_fn(MockSimData(), kwarg1="hi")


def test_invdes_simulation_data():
    invdes = make_invdes()
    params = invdes.design_region.params_random
    invdes.to_simulation_data(params=params, task_name="test")


def test_invdes_output_monitor_names(use_emulated_run):
    invdes = make_invdes()
    invdes = invdes.updated_copy(output_monitor_names=[MNT_NAME])

    params = invdes.design_region.params_random
    invdes.objective_fn(params, kwarg1="hi")


def test_invdes_mesh_override():
    invdes = make_invdes()
    params = invdes.design_region.params_random
    region_override_structure = invdes.design_region.to_mesh_override_structure()

    # if override_structure_dl left to None, use the design region override structure
    # with a dl corresponding to the pixel size
    invdes = invdes.updated_copy(override_structure_dl=None)
    sim = invdes.to_simulation(params)
    sim_override_structure = sim.grid_spec.override_structures[-1]
    assert sim_override_structure == region_override_structure
    assert all(dl == invdes.design_region.pixel_size for dl in sim_override_structure.dl)

    # if override_structure_dl set to False, should be no override structure
    invdes = invdes.updated_copy(override_structure_dl=False)
    sim = invdes.to_simulation(params)
    assert not sim.grid_spec.override_structures

    # if override_structure_dl set directly, ensure the override structure has the value
    dl = 0.1234
    invdes = invdes.updated_copy(override_structure_dl=dl)
    sim = invdes.to_simulation(params)
    assert sim.grid_spec.override_structures[-1].dl == (dl, dl, dl)


def make_optimizer():
    """Make an optimizer"""
    design = make_invdes()
    return tdi.AdamOptimizer(
        design=design,
        results_cache_fname=HISTORY_FNAME,
        learning_rate=0.2,
        num_steps=1,
    )


def make_result(use_emulated_run):
    """Test running the optimization defined in the ``InverseDesign`` object."""
    optimizer = make_optimizer()

    PARAMS_0 = np.random.random(optimizer.design.design_region.params_shape)

    return optimizer.run(params0=PARAMS_0)


def test_continue_run_fns(use_emulated_run):
    """Test continuing an already run inverse design."""
    result_orig = make_result(use_emulated_run)
    optimizer = make_optimizer()
    result_full = optimizer.continue_run(result=result_orig)

    num_steps_orig = len(result_orig.history["params"])
    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == num_steps_orig + optimizer.num_steps
    ), "wrong number of elements in the combined run history."


def test_continue_run_from_file(use_emulated_run):
    """Test continuing an already run inverse design from file."""
    result_orig = make_result(use_emulated_run)
    optimizer_orig = make_optimizer()
    optimizer = optimizer_orig.updated_copy(num_steps=optimizer_orig.num_steps + 1)
    result_full = optimizer.continue_run_from_file(HISTORY_FNAME)
    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == optimizer_orig.num_steps + optimizer.num_steps
    ), "wrong number of elements in the combined run history."
    result_full = optimizer.continue_run_from_history()


def test_result(use_emulated_run, tmp_path):
    """Test methods of the ``InverseDesignResult`` object."""
    result = make_result(use_emulated_run)

    with pytest.raises(KeyError):
        _ = result.get_final("blah")

    val_final1 = result.final["params"]
    val_final2 = result.get_final("params")
    assert np.allclose(val_final1, val_final2)

    result.plot_optimization()

    # gds_fname = str(tmp_path / "sim_final.gds")
    # gds_layer_dtype_map = {td.Medium(permittivity=4.0): (2, 1), td.Medium(): (1, 0)}
    # result.sim_final.to_gds_file(gds_fname, z=0, gds_layer_dtype_map=gds_layer_dtype_map)

    sim_data_final = result.sim_data_final(task_name="final")


def test_result_empty():
    result_empty = tdi.InverseDesignResult(design=make_invdes())
    with pytest.raises(ValueError):
        result_empty.get_final("params")


def test_invdes_io(tmp_path, log_capture, use_emulated_run):
    """Test saving a loading invdes instances to file."""

    result = make_result(use_emulated_run)
    optimizer = make_optimizer()
    design = optimizer.design

    for obj in (design, optimizer, result):
        obj.json()

        path = str(tmp_path / "obj.pkl")
        obj.to_file(path)
        obj2 = obj.from_file(path)

        # warn if not pickle extension
        path = str(tmp_path / "obj.other")
        with AssertLogLevel(log_capture, "WARNING"):
            obj.to_file(path)

        assert obj2.json() == obj.json()


@pytest.fixture
def hide_jax(monkeypatch, request):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in ["jaxlib.xla_extension"]:
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


def try_array_impl_import() -> None:
    """Try importing `tidy3d.plugins.adjoint.components.types`."""
    from importlib import reload
    import tidy3d

    reload(tidy3d.plugins.invdes.base)


@pytest.mark.usefixtures("hide_jax")
def test_jax_array_impl_import_fail(tmp_path, log_capture):
    """Make sure if import error with JVPTracer, a warning is logged and module still imports."""
    try_array_impl_import()
    assert_log_level(log_capture, "WARNING")


def test_jax_array_impl_import_pass(tmp_path, log_capture):
    """Make sure if no import error with JVPTracer, nothing is logged and module imports."""
    try_array_impl_import()
    assert_log_level(log_capture, None)


@pytest.mark.parametrize("exception, ok", [(TypeError, True), (OSError, True), (ValueError, False)])
def test_fn_source_error(monkeypatch, exception, ok):
    """Make sure type errors are caught."""

    import inspect

    def getsource_error(*args, **kwargs):
        raise exception

    monkeypatch.setattr("inspect.getsource", getsource_error)

    def test():
        return None

    # shouldnt raise exception as it's caught internally
    if ok:
        tdi.base.InvdesBaseModel._get_fn_source(test)

    # should raise exception as it's not caught internally
    else:
        with pytest.raises(exception):
            tdi.base.InvdesBaseModel._get_fn_source(test)
