# Test the inverse design plugin

import autograd.numpy as anp
import numpy as np
import pytest
import tidy3d as td
import tidy3d.plugins.invdes as tdi

from ..utils import AssertLogLevel, run_async_emulated, run_emulated

# use single threading pipeline
from .test_adjoint import use_emulated_run, use_emulated_run_async  # noqa: F401

FREQ0 = 1e14
L_SIM = 1.0
MNT_NAME1 = "mnt_name1"
MNT_NAME2 = "mnt_name2"
HISTORY_FNAME = "tests/data/invdes_history.json"


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


@pytest.fixture
def use_emulated_run_autograd(monkeypatch):
    """Emulate the tidy3d web function used in autograd web API."""
    import tidy3d.web.api.autograd.autograd as web_ag

    monkeypatch.setattr(web_ag, "_run_tidy3d", run_emulated)


@pytest.fixture
def use_emulated_run_autograd_async(monkeypatch):
    """Emulate the tidy3d web function used in autograd web API."""
    import tidy3d.web.api.autograd.autograd as web_ag

    monkeypatch.setattr(web_ag, "_run_async_tidy3d", run_async_emulated)


def make_design_region():
    """Make a ``TopologyDesignRegion``."""

    return tdi.TopologyDesignRegion(
        size=(0.4 * L_SIM, 0.4 * L_SIM, 0.4 * L_SIM),
        center=(0, 0, 0),
        eps_bounds=(1.0, 4.0),
        transformations=[tdi.FilterProject(radius=0.05, beta=2.0)],
        penalties=[
            tdi.ErosionDilationPenalty(length_scale=0.05),
        ],
        pixel_size=0.02,
    )


def test_region_params():
    """Test creation of parameter arrays from a ``TopologyDesignRegion``."""

    design_region = make_design_region()

    _ = np.random.random(design_region.params_shape)
    _ = design_region.params_random
    _ = design_region.params_ones
    _ = design_region.params_zeros


def test_region_penalties():
    """Test evaluation of penalties of a ``TopologyDesignRegion``."""

    region = make_design_region()

    PARAMS_0 = region.params_random

    # test some design region functions
    region.material_density(PARAMS_0)
    region.penalty_value(PARAMS_0)


def test_region_to_structure():
    """Test converting ``TopologyDesignRegion`` to jax and regular structure."""

    region = make_design_region()

    PARAMS_0 = region.params_ones

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


def post_process_fn(sim_data: td.SimulationData, **kwargs) -> float:
    """Define a post-processing function with extra kwargs (recommended)."""
    intensity = sim_data.get_intensity(MNT_NAME1)
    return anp.sum(intensity.values)


postprocess_obj = tdi.WeightedSum(
    powers=[
        tdi.GetPowerMode(monitor_name=MNT_NAME2, direction="+", mode_index=0, weight=0.5),
    ]
)


def post_process_fn_kwargless(sim_data: td.SimulationData) -> float:
    """Define a post-processing function with no kwargs specified."""
    intensity = sim_data.get_intensity(MNT_NAME1)
    return anp.sum(intensity.values)


def post_process_fn_multi(batch_data: dict[str, td.SimulationData], **kwargs) -> float:
    """Define a post-processing function for batch."""
    val = 0.0
    for _, sim_data in batch_data.items():
        intensity_i = sim_data.get_intensity(MNT_NAME1)
        val += anp.sum(intensity_i.values)
        power_i = tdi.utils.sum_abs_squared(tdi.utils.get_amps(sim_data, MNT_NAME2))
        val += power_i
    return val


def post_process_fn_untraced(sim_data: td.SimulationData, **kwargs) -> float:
    """Define a post-processing function with extra kwargs (recommended)."""
    return 1.0


def make_invdes():
    """Make an inverse design"""
    return tdi.InverseDesign(
        simulation=simulation,
        design_region=make_design_region(),
        task_name="test",
    )


class MockDataArray:
    """Pretends to be a ``JaxDataArray`` with ``.values``."""

    values = anp.linspace(0, 1, 10)


class MockSimData:
    """Pretends to be a ``JaxSimulationData``, returns a data array with ``.get_intensity()``."""

    def get_intensity(self, name: str) -> MockDataArray:
        return MockDataArray()

    def __getitem__(self, name: str) -> MockDataArray:
        return MockDataArray()


def test_invdes_simulation_data(use_emulated_run):  # noqa: F811
    """Test convenience function to convert ``InverseDesign`` to simulation and run it."""

    invdes = make_invdes()
    params = invdes.design_region.params_random
    invdes.to_simulation_data(params=params, task_name="test")


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

    _ = invdes.designs


def make_optimizer():
    """Make a ``tdi.Optimizer``."""
    design = make_invdes()
    return tdi.AdamOptimizer(
        design=design,
        results_cache_fname=HISTORY_FNAME,
        learning_rate=0.2,
        num_steps=1,
    )


def make_result(use_emulated_run_autograd, fn_postprocess: bool = True):
    """Test running the optimization defined in the ``InverseDesign`` object."""

    optimizer = make_optimizer()

    PARAMS_0 = np.random.random(optimizer.design.design_region.params_shape)

    if fn_postprocess:
        return optimizer.run(params0=PARAMS_0, post_process_fn=post_process_fn)
    else:
        optimizer = optimizer.updated_copy(postprocess=postprocess_obj, path="design")
        return optimizer.run(params0=PARAMS_0)


def test_default_params(use_emulated_run_autograd):  # noqa: F811
    """Test default paramns running the optimization defined in the ``InverseDesign`` object."""

    optimizer = make_optimizer()

    _ = np.random.random(optimizer.design.design_region.params_shape)

    optimizer.run(post_process_fn=post_process_fn)


def test_warn_zero_grad(log_capture, use_emulated_run_autograd):
    """Test default paramns running the optimization defined in the ``InverseDesign`` object."""

    optimizer = make_optimizer()
    with AssertLogLevel(
        log_capture, "WARNING", contains_str="All elements of the gradient are almost zero"
    ):
        _ = optimizer.run(post_process_fn=post_process_fn_untraced)


def make_result_multi(use_emulated_run_autograd_async):
    """Test running the optimization defined in the ``InverseDesignMulti`` object."""

    optimizer = make_optimizer()
    design = make_invdes_multi()

    optimizer = optimizer.updated_copy(design=design)

    PARAMS_0 = np.random.random(optimizer.design.design_region.params_shape)

    return optimizer.run(params0=PARAMS_0, post_process_fn=post_process_fn_multi)


def test_result_store_full_results_is_false(use_emulated_run_autograd):  # noqa: F811
    """Test running the optimization defined in the ``InverseDesign`` object."""

    optimizer = make_optimizer()
    optimizer = optimizer.updated_copy(store_full_results=False, num_steps=3)

    PARAMS_0 = np.random.random(optimizer.design.design_region.params_shape)

    result = optimizer.run(params0=PARAMS_0, post_process_fn=post_process_fn)

    # these store at the very beginning and at the end of every iteration
    # but when ``store_full_results == False``, they only store the last one
    for key in ("params", "grad", "opt_state"):
        assert len(result.history[key]) == 1

    # these store at the end of each iteration
    for key in ("penalty", "objective_fn_val", "post_process_val"):
        assert len(result.history[key]) == optimizer.num_steps

    # this should still work, even if ``store_full_results == False``
    _ = result.last["params"]


def test_continue_run_fns(use_emulated_run_autograd):
    """Test continuing an already run inverse design from result."""
    result_orig = make_result(use_emulated_run_autograd)
    optimizer = make_optimizer()
    result_full = optimizer.continue_run(result=result_orig, post_process_fn=post_process_fn)

    num_steps_orig = len(result_orig.history["params"])
    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == num_steps_orig + optimizer.num_steps
    ), "wrong number of elements in the combined run history."


def test_continue_run_from_file(use_emulated_run_autograd):
    """Test continuing an already run inverse design from file."""
    result_orig = make_result(use_emulated_run_autograd)
    optimizer_orig = make_optimizer()
    optimizer = optimizer_orig.updated_copy(num_steps=optimizer_orig.num_steps + 1)
    result_full = optimizer.continue_run_from_file(HISTORY_FNAME, post_process_fn=post_process_fn)
    num_steps_orig = len(result_orig.history["params"])
    num_steps_full = len(result_full.history["params"])
    assert (
        num_steps_full == num_steps_orig + optimizer.num_steps
    ), "wrong number of elements in the combined run history."

    # test the convenience function to load it from file
    result_full = optimizer.continue_run_from_history(post_process_fn=post_process_fn)


def test_postprocess_init(use_emulated_run):  # noqa: F811
    """Test the intiialization of an ``InverseDesign`` with different ``postprocess`` options."""

    power_obj = tdi.GetPowerMode(monitor_name=MNT_NAME2, direction="+", mode_index=0, weight=0.5)
    postprocess_obj = tdi.WeightedSum(powers=[power_obj])

    def fn(sim_data):
        return power_obj.evaluate(sim_data)

    # basline
    invdes = make_invdes()
    params = invdes.design_region.params_random
    sim_data = invdes.to_simulation_data(params=params, task_name="test")
    obj_fn_val = fn(sim_data)

    # supply postprocess object directly
    invdes_obj = invdes.updated_copy(postprocess=postprocess_obj)

    # supply invdes with a function from classmethod constructor
    invdes_from_fn = invdes.from_function(
        fn,
        **invdes.dict(
            exclude={
                "postprocess",
            }
        ),
    )

    # use the custom postprocess operation.
    invdes_from_custom = invdes.updated_copy(
        postprocess=tdi.CustomPostprocessOperation.from_function(fn)
    )

    # check that they all give the same objective function values, and calling evaluate works
    for invdes_ in (invdes_obj, invdes_from_fn, invdes_from_custom):
        obj_fn_val_ = invdes_.postprocess.evaluate(sim_data)
        assert np.isclose(obj_fn_val, obj_fn_val_)


@pytest.mark.parametrize("fn_postprocess", (True, False))
def test_result(
    use_emulated_run,  # noqa: F811
    use_emulated_run_autograd,
    use_emulated_run_autograd_async,
    tmp_path,
    fn_postprocess,
):
    """Test methods of the ``InverseDesignResult`` object."""

    result = make_result(use_emulated_run_autograd, fn_postprocess=fn_postprocess)

    with pytest.raises(KeyError):
        _ = result.get_last("blah")

    val_last1 = result.last["params"]
    val_last2 = result.get_last("params")
    assert np.allclose(val_last1, val_last2)

    result.plot_optimization()
    _ = result.sim_data_last(task_name="last")


def test_result_data(use_emulated_run, use_emulated_run_autograd):  # noqa: F811
    """Test methods of the ``InverseDesignResult`` object."""

    result = make_result(use_emulated_run_autograd)
    _ = result.sim_last
    _ = result.sim_data_last(task_name="last")


def test_result_data_multi(
    use_emulated_run_async,  # noqa: F811
    use_emulated_run_autograd_async,
    use_emulated_run_autograd,
    tmp_path,
):
    result_multi = make_result_multi(use_emulated_run_autograd_async)
    _ = result_multi.sim_last
    _ = result_multi.sim_data_last()


def test_result_empty():
    """Assert that things error with empty results."""
    result_empty = tdi.InverseDesignResult(design=make_invdes())
    with pytest.raises(ValueError):
        result_empty.get_last("params")


def test_invdes_io(tmp_path, log_capture, use_emulated_run_autograd):
    """Test saving a loading ``invdes`` components to file."""

    result = make_result(use_emulated_run_autograd)
    optimizer = make_optimizer()
    design = optimizer.design

    for obj in (design, optimizer, result):
        obj.json()

        path = str(tmp_path / "obj.hdf5")
        obj.to_file(path)
        obj2 = obj.from_file(path)

        assert obj2.json() == obj.json()


def test_objective_utilities(use_emulated_run_autograd):
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
