# Test the inverse design plugin

import pytest
import builtins

import numpy as np
import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda
import tidy3d.plugins.invdes as tdi
import matplotlib.pyplot as plt

from .test_adjoint import use_emulated_run, use_emulated_run_async
from ..utils import run_emulated, log_capture, assert_log_level, AssertLogLevel

FREQ0 = 1e14
L_SIM = 2.0
MNT_NAME1 = "mnt_name1"
MNT_NAME2 = "mnt_name2"
HISTORY_FNAME = "tests/data/invdes_history.pkl"

td.config.logging_level = "ERROR"

mnt1 = td.FieldMonitor(
    center=(L_SIM / 3.0, 0, 0), size=(0, td.inf, td.inf), freqs=[FREQ0], name=MNT_NAME1
)

mnt2 = td.ModeMonitor(
    center=(-L_SIM / 3.0, 0, 0),
    size=(0, td.inf, td.inf),
    freqs=[FREQ0],
    name=MNT_NAME2,
    mode_spec=td.ModeSpec(num_modes=1),
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
    monitors=[mnt1, mnt2],
)


def make_design_region():
    """Make a ``TopologyDesignRegion``."""

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
    """Test creation of parameter arrays from a ``TopologyDesignRegion``."""

    design_region = make_design_region()

    PARAMS_0 = np.random.random(design_region.params_shape)
    PARAMS_0 = design_region.params_random
    PARAMS_0 = design_region.params_ones
    PARAMS_0 = design_region.params_zeros


def test_region_penalties():
    """Test evaluation of penalties of a ``TopologyDesignRegion``."""

    region = make_design_region()

    PARAMS_0 = region.params_random

    # test some design region functions
    region.material_density(PARAMS_0)
    region.penalty_value(PARAMS_0)

    region.updated_copy(penalties=[]).penalty_value(PARAMS_0)


def test_region_to_structure():
    """Test converting ``TopologyDesignRegion`` to jax and regular structure."""

    region = make_design_region()

    PARAMS_0 = region.params_ones

    _ = region.to_jax_structure(PARAMS_0)
    _ = region.to_structure(PARAMS_0)


def test_region_params_bounds():
    """Test supplying parameters out of bounds to ``TopologyDesignRegion``."""

    region = make_design_region()

    PARAMS_0 = region.params_ones

    with pytest.raises(ValueError):
        region.penalty_value(2 * PARAMS_0)

    with pytest.raises(ValueError):
        region.penalty_value(-1 * PARAMS_0)


def test_region_inf_size():
    """Test that a structure can be created with an infinite size edge case."""

    region = make_design_region()
    inf_size = list(region.size)
    inf_size[1] = td.inf
    region = region.updated_copy(size=inf_size)
    params_0_inf = region.params_zeros
    _ = region.to_structure(params_0_inf)


def test_penalty_pixel_size(log_capture):
    """Test that warning is raised if ``pixel_size`` is supplied."""

    _ = tdi.ErosionDilationPenalty(length_scale=2.0)

    with AssertLogLevel(log_capture, "WARNING"):
        _ = tdi.ErosionDilationPenalty(pixel_size=1.0, length_scale=2.0)


def post_process_fn(sim_data: tda.JaxSimulationData, **kwargs) -> float:
    """Define a post-processing function with extra kwargs (recommended)."""
    intensity = sim_data.get_intensity(MNT_NAME1)
    return jnp.sum(intensity.values)


def post_process_fn_kwargless(sim_data: tda.JaxSimulationData) -> float:
    """Define a post-processing function with no kwargs specified."""
    intensity = sim_data.get_intensity(MNT_NAME1)
    return jnp.sum(intensity.values)


def post_process_fn_multi(sim_data_list: list[tda.JaxSimulationData], **kwargs) -> float:
    """Define a post-processing function for batch."""
    val = 0.0
    for sim_data in sim_data_list:
        intensity_i = sim_data.get_intensity(MNT_NAME1)
        val += jnp.sum(intensity_i.values)
        power_i = tdi.utils.sum_abs_squared(tdi.utils.get_amps(sim_data, MNT_NAME2))
        val += power_i
    return val


def make_invdes():
    """Make an inverse design"""
    return tdi.InverseDesign(
        simulation=simulation,
        design_region=make_design_region(),
        post_process_fn=post_process_fn,
        task_name="test",
    )


class MockDataArray:
    """Pretends to be a ``JaxDataArray`` with ``.values``."""

    values = jnp.linspace(0, 1, 10)


class MockSimData:
    """Pretends to be a ``JaxSimulationData``, returns a data array with ``.get_intensity()``."""

    def get_intensity(self, name: str) -> MockDataArray:
        return MockDataArray()

    def __getitem__(self, name: str) -> MockDataArray:
        return MockDataArray()


def test_invdes_kwarglfull():
    """Test that a postprocess function works when defined with kwargs in the signature."""

    invdes = make_invdes()
    invdes.post_process_fn(MockSimData(), kwarg1="hi")


def test_invdes_kwargless():
    """Test that a postprocess function works when defined with no kwargs in the signature."""

    invdes = make_invdes()
    invdes = invdes.updated_copy(post_process_fn=post_process_fn_kwargless)
    invdes.post_process_fn(MockSimData(), kwarg1="hi")


def test_invdes_simulation_data(use_emulated_run):
    """Test convenience function to convert ``InverseDesign`` to simulation and run it."""

    invdes = make_invdes()
    params = invdes.design_region.params_random
    invdes.to_simulation_data(params=params, task_name="test")


def test_invdes_output_monitor_names(use_emulated_run):
    """test the inverse design objective function with user-specified output monitor names."""
    invdes = make_invdes()
    invdes = invdes.updated_copy(output_monitor_names=[MNT_NAME1, MNT_NAME2])

    params = invdes.design_region.params_random
    invdes.objective_fn(params, kwarg1="hi")


def test_invdes_mesh_override():
    """Test all edge cases of mesh override structure in the ``JaxSimulation``."""

    region = make_design_region()
    invdes = make_invdes()
    params = invdes.design_region.params_random

    # if ``override_structure_dl`` left of ``None`` (default), use an override structure
    # defined by the design region, with a ``dl`` corresponding to the ``pixel_size``.
    region = region.updated_copy(override_structure_dl=None)
    invdes = invdes.updated_copy(design_region=region)
    sim = invdes.to_simulation(params)
    sim_override_structure = sim.grid_spec.override_structures[-1]
    region_override_structure = invdes.design_region.mesh_override_structure
    assert sim_override_structure == region_override_structure
    assert all(dl == invdes.design_region.pixel_size for dl in sim_override_structure.dl)

    # if ``override_structure_dl`` set to ``False``, should be no override structure
    region = region.updated_copy(override_structure_dl=False)
    invdes = invdes.updated_copy(design_region=region)
    sim = invdes.to_simulation(params)
    assert not sim.grid_spec.override_structures

    # if ``override_structure_dl`` set directly, ensure the override structure has this value
    dl = 0.1234
    region = region.updated_copy(override_structure_dl=dl)
    invdes = invdes.updated_copy(design_region=region)
    sim = invdes.to_simulation(params)
    assert sim.grid_spec.override_structures[-1].dl == (dl, dl, dl)


def make_invdes_multi():
    n = 7

    region = make_design_region()

    simulations = n * [simulation]
    # post_process_fns = n * [post_process_fn]

    invdes = tdi.InverseDesignMulti(
        design_region=region,
        simulations=simulations,
        # post_process_fns=post_process_fns,
        post_process_fn=post_process_fn_multi,
        task_name="base",
    )

    return invdes


def test_invdes_multi_same_length():
    """Test validator ensuring all multi fields have the same lengths, if applicable."""

    invdes = make_invdes_multi()
    n = len(invdes.simulations)

    output_monitor_names = (n + 1) * [["test"]]

    with pytest.raises(ValueError):
        _ = invdes.updated_copy(output_monitor_names=output_monitor_names)

    output_monitor_names = [([MNT_NAME1, MNT_NAME2], None)[i % 2] for i in range(n)]
    invdes = invdes.updated_copy(output_monitor_names=output_monitor_names)

    ds = invdes.designs


def make_optimizer():
    """Make a ``tdi.Optimizer``."""
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


def make_result_multi(use_emulated_run_async):
    """Test running the optimization defined in the ``InverseDesignMulti`` object."""

    optimizer = make_optimizer()
    design = make_invdes_multi()

    optimizer = optimizer.updated_copy(design=design)

    PARAMS_0 = np.random.random(optimizer.design.design_region.params_shape)

    return optimizer.run(params0=PARAMS_0)


def test_result_store_full_results_is_false(use_emulated_run):
    """Test running the optimization defined in the ``InverseDesign`` object."""

    optimizer = make_optimizer()
    optimizer = optimizer.updated_copy(store_full_results=False, num_steps=3)

    PARAMS_0 = np.random.random(optimizer.design.design_region.params_shape)

    result = optimizer.run(params0=PARAMS_0)

    # these store at the very beginning and at the end of every iteration
    # but when ``store_full_results == False``, they only store the last one
    for key in ("params", "grad", "opt_state"):
        assert len(result.history[key]) == 1

    # these store at the end of each iteration
    for key in ("penalty", "objective_fn_val", "post_process_val"):
        assert len(result.history[key]) == optimizer.num_steps

    # this should still work, even if ``store_full_results == False``
    val_last1 = result.last["params"]


def test_continue_run_fns(use_emulated_run):
    """Test continuing an already run inverse design from result."""
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
    num_steps_orig = len(result_orig.history["params"])
    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == num_steps_orig + optimizer.num_steps
    ), "wrong number of elements in the combined run history."

    # test the convenience function to load it from file
    result_full = optimizer.continue_run_from_history()


def test_result(use_emulated_run, use_emulated_run_async, tmp_path):
    """Test methods of the ``InverseDesignResult`` object."""

    result = make_result(use_emulated_run)

    with pytest.raises(KeyError):
        _ = result.get_last("blah")

    val_last1 = result.last["params"]
    val_last2 = result.get_last("params")
    assert np.allclose(val_last1, val_last2)

    result.plot_optimization()
    sim_data_last = result.sim_data_last(task_name="last")


def test_result_data(use_emulated_run):
    """Test methods of the ``InverseDesignResult`` object."""

    result = make_result(use_emulated_run)
    sim_last = result.sim_last
    sim_data_last = result.sim_data_last(task_name="last")


def test_result_data_multi(use_emulated_run_async, tmp_path):
    result_multi = make_result_multi(use_emulated_run_async)
    sim_last = result_multi.sim_last
    sim_data_last = result_multi.sim_data_last(task_name="last")


def test_result_empty():
    """Assert that things error with empty results."""
    result_empty = tdi.InverseDesignResult(design=make_invdes())
    with pytest.raises(ValueError):
        result_empty.get_last("params")


def test_invdes_io(tmp_path, log_capture, use_emulated_run):
    """Test saving a loading ``invdes`` components to file."""

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
    """Force an ``ImportError`` when trying to import ``jaxlib.xla_extension``."""
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in ["jaxlib.xla_extension"]:
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


def try_array_impl_import() -> None:
    """Try importing ``tidy3d.plugins.invdes.base``."""

    from importlib import reload
    import tidy3d

    reload(tidy3d.plugins.invdes.base)


@pytest.mark.usefixtures("hide_jax")
def test_jax_array_impl_import_fail(tmp_path, log_capture):
    """Make sure if import error with ArrayImpl, a warning is logged and module still imports."""
    try_array_impl_import()
    assert_log_level(log_capture, "WARNING")


def test_jax_array_impl_import_pass(tmp_path, log_capture):
    """Make sure if no import error with ArrayImpl, nothing is logged and module imports."""
    try_array_impl_import()
    assert_log_level(log_capture, None)


@pytest.mark.parametrize("exception, ok", [(TypeError, True), (OSError, True), (ValueError, False)])
def test_fn_source_error(monkeypatch, exception, ok):
    """Make sure type errors are caught when grabbing function source code."""

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


def test_objective_utilities(use_emulated_run):
    """Test objective function helpers."""

    sim_data = run_emulated(simulation, task_name="test")

    value = 0.0

    utils = tdi.utils

    amps_i = utils.get_amps(sim_data, MNT_NAME2)
    value += utils.sum_abs_squared(amps_i)

    phase = utils.get_phase(amps_i)
    value += utils.sum_array(phase)

    intensity = utils.get_intensity(sim_data, MNT_NAME1)
    value += utils.sum_array(intensity)

    ex = utils.get_field_component(sim_data, MNT_NAME1, "Ex")
    value += utils.sum_abs_squared(ex)

    with pytest.raises(ValueError):
        utils.get_intensity(sim_data, MNT_NAME2)

    with pytest.raises(ValueError):
        utils.get_field_component(sim_data, MNT_NAME2, "Ex")

    with pytest.raises(ValueError):
        utils.get_amps(sim_data, MNT_NAME1)


def test_pixel_size_warn_validator(log_capture):
    """test that pixel size validator warning is raised if too large."""

    with AssertLogLevel(log_capture, None):
        invdes = make_invdes()

    wvl_mat_min = invdes.simulation.wvl_mat_min
    region_too_coarse = invdes.design_region.updated_copy(pixel_size=wvl_mat_min)
    with AssertLogLevel(log_capture, "WARNING", contains_str="pixel_size"):
        invdes = invdes.updated_copy(design_region=region_too_coarse)

    with AssertLogLevel(log_capture, None):
        invdes_multi = make_invdes_multi()

    with AssertLogLevel(log_capture, "WARNING", contains_str="pixel_size"):
        invdes_multi = invdes_multi.updated_copy(design_region=region_too_coarse)
