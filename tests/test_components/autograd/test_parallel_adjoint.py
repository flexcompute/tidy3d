from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web.api.autograd.autograd as autograd_api
from tidy3d.components.autograd.parallel_adjoint_bases import (
    DiffractionAdjointBasis,
    ModeAdjointBasis,
    PointFieldAdjointBasis,
)
from tidy3d.components.autograd.source_factory import (
    adjoint_fwidth_from_simulation,
    point_current_source_from_simulation,
)
from tidy3d.config import config
from tidy3d.web import run, run_async
from tidy3d.web.api.autograd.parallel_adjoint import (
    _outgoing_mode_direction,
    apply_parallel_adjoint,
    prepare_parallel_adjoint,
)

from ...utils import AssertLogStr
from .test_autograd import (
    FREQ0,
    SIM_BASE,
    AssertLogLevel,
    get_functions,
    make_monitors,
    make_structures,
    params0,
    use_emulated_run,  # noqa: F401
)


@pytest.mark.parametrize("monitor_key", ("mode", "diff", "field_point"))
def test_parallel_adjoint_matches_sequential(use_emulated_run, monitor_key, monkeypatch):  # noqa: F811
    """Ensure parallel adjoint gradients match the sequential local-gradient path."""

    fn_dict = get_functions("medium", monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        sim = make_sim(*args)
        data = run(
            sim,
            task_name="parallel_adjoint_test",
            verbose=False,
            local_gradient=True,
        )
        return postprocess(data)

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )
    monkeypatch.setattr(config.adjoint, "parallel_all_port", False)
    val_seq, grad_seq = ag.value_and_grad(objective)(params0)
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    val_par, grad_par = ag.value_and_grad(objective)(params0)

    assert np.isclose(val_seq, val_par)
    assert np.allclose(grad_seq, grad_par)


def test_parallel_adjoint_fallback_unsupported(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure parallel adjoint is disabled when no eligible monitors are present."""

    monitors = make_monitors()
    field_vol_monitor = monitors["field_vol"][0]

    def make_sim(*args):
        structures = make_structures(*args)
        return SIM_BASE.updated_copy(
            structures=[structures["medium"]], monitors=[field_vol_monitor]
        )

    def objective(*args):
        sim = make_sim(*args)
        data = run(
            sim,
            task_name="parallel_adjoint_fallback",
            verbose=False,
            local_gradient=True,
        )
        field_data = data[field_vol_monitor.name]
        return anp.sum(anp.abs(field_data.field_components["Ex"].values))

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    with AssertLogLevel("WARNING", contains_str="unsupported monitors"):
        _, grad = ag.value_and_grad(objective)(params0)

    assert anp.any(grad != 0.0)


def test_parallel_adjoint_partial_subset(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure parallel adjoint is disabled when mixed monitor support is present."""

    monitors = make_monitors()
    mode_monitor = monitors["mode"][0]
    field_vol_monitor = monitors["field_vol"][0]

    def make_sim(*args):
        structures = make_structures(*args)
        return SIM_BASE.updated_copy(
            structures=[structures["medium"]],
            monitors=[mode_monitor, field_vol_monitor],
        )

    def objective(*args):
        sim = make_sim(*args)
        data = run(
            sim,
            task_name="parallel_adjoint_subset",
            verbose=False,
            local_gradient=True,
        )
        mode_data = data[mode_monitor.name]
        field_data = data[field_vol_monitor.name]
        mode_val = anp.sum(anp.abs(mode_data.amps.values) ** 2)
        field_val = anp.sum(anp.abs(field_data.field_components["Ex"].values))
        return mode_val + field_val

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )
    monkeypatch.setattr(config.adjoint, "parallel_all_port", False)
    val_seq, grad_seq = ag.value_and_grad(objective)(params0)
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    with AssertLogLevel("WARNING", contains_str="unsupported monitors"):
        val_par, grad_par = ag.value_and_grad(objective)(params0)

    assert np.isclose(val_seq, val_par)
    assert np.allclose(grad_seq, grad_par)


def test_parallel_adjoint_fallback_warning(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure mixed monitor support disables parallel adjoint."""

    monitors = make_monitors()
    mode_monitor = monitors["mode"][0]
    field_vol_monitor = monitors["field_vol"][0]

    def make_sim(*args):
        structures = make_structures(*args)
        return SIM_BASE.updated_copy(
            structures=[structures["medium"]],
            monitors=[mode_monitor, field_vol_monitor],
        )

    def objective(*args):
        sim = make_sim(*args)
        data = run(
            sim,
            task_name="parallel_adjoint_fallback_warning",
            verbose=False,
            local_gradient=True,
        )
        mode_data = data[mode_monitor.name]
        field_data = data[field_vol_monitor.name]
        mode_val = anp.sum(anp.abs(mode_data.amps.values) ** 2)
        field_val = anp.sum(anp.abs(field_data.field_components["Ex"].values))
        return mode_val + field_val

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)

    with AssertLogStr("WARNING", contains_str="unsupported monitors"):
        ag.value_and_grad(objective)(params0)


def test_parallel_adjoint_matches_sequential_multifreq_mode(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure parallel adjoint matches sequential for multi-frequency mode objectives."""

    freqs = [0.95 * FREQ0, FREQ0, 1.05 * FREQ0]
    mode_monitor = make_monitors()["mode"][0].updated_copy(freqs=freqs)

    def make_sim(*args):
        structures = make_structures(*args)
        return SIM_BASE.updated_copy(
            structures=[structures["medium"]],
            monitors=[mode_monitor],
        )

    def objective(*args):
        sim = make_sim(*args)
        data = run(
            sim,
            task_name="parallel_adjoint_multifreq_mode",
            verbose=False,
            local_gradient=True,
        )
        mode_data = data[mode_monitor.name]
        return anp.sum(anp.abs(mode_data.amps.values) ** 2)

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(config.adjoint, "parallel_adjoint_mode_direction_policy", "assume_outgoing")
    monkeypatch.setattr(config.adjoint, "parallel_all_port", False)
    val_seq, grad_seq = ag.value_and_grad(objective)(params0)
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    val_par, grad_par = ag.value_and_grad(objective)(params0)

    assert np.isclose(val_seq, val_par)
    assert np.allclose(grad_seq, grad_par)


def test_parallel_adjoint_limit_error(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure an error is raised when the parallel adjoint limit is zero."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = {"1", "2"}

    def objective(*args):
        sims = {task_name: make_sim(*args) for task_name in task_names}
        batch_data = run_async(
            sims,
            verbose=False,
            local_gradient=True,
            max_num_adjoint_per_fwd=0,
        )
        values = []
        for _, sim_data in batch_data.items():
            values.append(postprocess(sim_data))
        return 0 * sum(values)

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )
    with pytest.raises(td.exceptions.AdjointError, match="Number of parallel adjoint simulations"):
        ag.grad(objective)(params0)


def test_parallel_adjoint_mode_direction_policy_assume_outgoing(monkeypatch):
    """Ensure assume_outgoing keeps only the expected mode direction."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]
    sim = make_sim(params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )

    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    monkeypatch.setattr(config.adjoint, "parallel_adjoint_mode_direction_policy", "assume_outgoing")

    payload = prepare_parallel_adjoint(
        simulation=sim.to_static(),
        sim_fields_keys=list(sim_fields.keys()),
        task_name="parallel_mode_direction",
        max_num_adjoint_per_fwd=100,
    )

    assert payload is not None
    mode_monitor = next(m for m in sim.monitors if m.name == "mode")
    axis = mode_monitor.normal_axis
    expected_dir = "+" if mode_monitor.center[axis] >= sim.center[axis] else "-"
    directions = {
        desc.direction for desc in payload.basis_specs if isinstance(desc, ModeAdjointBasis)
    }
    assert directions == {expected_dir}


def test_parallel_adjoint_mode_direction_policy_no_parallel(monkeypatch):
    """Ensure no_parallel disables parallel adjoint for mode monitors only."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]
    sim = make_sim(params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )

    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    monkeypatch.setattr(config.adjoint, "parallel_adjoint_mode_direction_policy", "no_parallel")

    payload = prepare_parallel_adjoint(
        simulation=sim.to_static(),
        sim_fields_keys=list(sim_fields.keys()),
        task_name="parallel_mode_direction_disabled",
        max_num_adjoint_per_fwd=100,
    )

    assert payload is not None
    assert all(not isinstance(basis, ModeAdjointBasis) for basis in payload.basis_specs)


def test_parallel_adjoint_diffraction_bases(monkeypatch):
    """Ensure diffraction monitors expose parallel adjoint bases."""

    fn_dict = get_functions("medium", "diff")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )

    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)

    payload = prepare_parallel_adjoint(
        simulation=sim.to_static(),
        sim_fields_keys=list(sim_fields.keys()),
        task_name="parallel_diffraction_bases",
        max_num_adjoint_per_fwd=100,
    )

    assert payload is not None
    assert any(isinstance(basis, DiffractionAdjointBasis) for basis in payload.basis_specs)


def test_parallel_adjoint_mode_bases(monkeypatch):
    """Ensure mode monitors expose parallel adjoint bases."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )

    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)

    payload = prepare_parallel_adjoint(
        simulation=sim.to_static(),
        sim_fields_keys=list(sim_fields.keys()),
        task_name="parallel_mode_bases",
        max_num_adjoint_per_fwd=100,
    )

    assert payload is not None
    assert any(isinstance(basis, ModeAdjointBasis) for basis in payload.basis_specs)


def test_parallel_adjoint_point_field_bases(monkeypatch):
    """Ensure point field monitors expose parallel adjoint bases."""

    fn_dict = get_functions("medium", "field_point")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )

    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)

    payload = prepare_parallel_adjoint(
        simulation=sim.to_static(),
        sim_fields_keys=list(sim_fields.keys()),
        task_name="parallel_point_field_bases",
        max_num_adjoint_per_fwd=100,
    )

    assert payload is not None
    assert any(isinstance(basis, PointFieldAdjointBasis) for basis in payload.basis_specs)


def test_parallel_adjoint_unused_warning(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure a warning is emitted when parallel adjoint bases are unused."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]

    def objective(*args):
        sim = make_sim(*args)
        data = run(
            sim,
            task_name="parallel_adjoint_unused",
            verbose=False,
            local_gradient=True,
        )
        monitor = next(m for m in sim.monitors if isinstance(m, td.ModeMonitor))
        mode_data = data[monitor.name]
        outgoing_dir = _outgoing_mode_direction(sim, monitor)
        directions = [str(direction) for direction in mode_data.amps.coords["direction"].values]
        outgoing_index = directions.index(outgoing_dir)
        amp = mode_data.amps.values[outgoing_index, 0, 0]
        return anp.abs(amp) ** 2

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)

    with AssertLogLevel("WARNING", contains_str="unused"):
        ag.grad(objective)(params0)


def test_point_current_source_from_simulation(use_emulated_run):  # noqa: F811
    """Ensure point-current adjoint sources can be generated from simulation data."""

    fn_dict = get_functions("medium", "field_point")
    sim = fn_dict["sim"](params0)
    monitor = next(m for m in sim.monitors if isinstance(m, td.FieldMonitor))
    freq = float(monitor.freqs[0])
    fwidth = adjoint_fwidth_from_simulation(sim)

    source = point_current_source_from_simulation(
        simulation=sim,
        monitor=monitor,
        component="Ex",
        freq=freq,
        coefficient=1.0 + 0.5j,
        fwidth=fwidth,
    )

    assert source is not None
    assert source.current_dataset is not None
    assert np.any(source.current_dataset.field_components["Ex"].values != 0.0)


def test_apply_parallel_adjoint_assume_outgoing_mode_vjp(use_emulated_run, monkeypatch, tmp_path):  # noqa: F811
    """Ensure assume_outgoing reports incoming-mode VJP entries."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    sim_data = run(
        sim,
        task_name="parallel_apply_assume_outgoing",
        path=tmp_path / "parallel_apply_assume_outgoing.hdf5",
        verbose=False,
    )

    monitor_index, monitor = next(
        (i, m) for i, m in enumerate(sim.monitors) if isinstance(m, td.ModeMonitor)
    )
    mode_data = sim_data[monitor.name]
    bases = monitor.parallel_adjoint_bases(sim, monitor_index)

    outgoing_dir = _outgoing_mode_direction(sim, monitor)
    directions = [str(direction) for direction in mode_data.amps.coords["direction"].values]
    outgoing_index = directions.index(outgoing_dir)
    incoming_index = 1 - outgoing_index

    outgoing_basis = next(
        desc
        for desc in bases
        if isinstance(desc, ModeAdjointBasis)
        and desc.direction == outgoing_dir
        and desc.freq == float(mode_data.amps.coords["f"].values[0])
        and desc.mode_index == int(mode_data.amps.coords["mode_index"].values[0])
    )

    vjp = np.zeros_like(mode_data.amps.values, dtype=complex)
    vjp[outgoing_index, 0, 0] = 2.0 + 0.0j
    vjp[incoming_index, 0, 0] = 3.0 + 0.0j
    data_path = ("data", monitor_index, "amps")
    data_fields_vjp = {data_path: vjp}

    basis_maps = {
        outgoing_basis: {
            "real": {("structures", 0, "medium", "permittivity"): np.array(5.0)},
            "imag": {("structures", 0, "medium", "permittivity"): np.array(7.0)},
        }
    }
    parallel_info = {
        "basis_maps": basis_maps,
        "basis_specs": [outgoing_basis],
        "task_name": "assume_outgoing",
    }

    monkeypatch.setattr(config.adjoint, "parallel_adjoint_mode_direction_policy", "assume_outgoing")

    vjp_parallel, fallback = apply_parallel_adjoint(
        data_fields_vjp=data_fields_vjp,
        parallel_info=parallel_info,
        sim_data_orig=sim_data,
    )

    assert np.isclose(
        vjp_parallel[("structures", 0, "medium", "permittivity")],
        10.0,
    )
    fallback_vjp = fallback[data_path]
    assert fallback_vjp[outgoing_index, 0, 0] == 0.0
    assert fallback_vjp[incoming_index, 0, 0] != 0.0


def test_parallel_adjoint_launches_parallel_tasks(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure the forward batch includes canonical parallel-adjoint tasks."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]
    task_names = {"pa_task_1", "pa_task_2"}

    captured_task_names: set[str] = set()
    orig_run_async = autograd_api._run_async_tidy3d

    def _run_async_capture(simulations, **run_kwargs):
        captured_task_names.update(simulations.keys())
        return orig_run_async(simulations, **run_kwargs)

    monkeypatch.setattr(autograd_api, "_run_async_tidy3d", _run_async_capture)
    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "parallel_all_port", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )

    def objective(*args):
        sims = {task_name: make_sim(*args) for task_name in task_names}
        batch_data = run_async(sims, verbose=False, local_gradient=True)
        values = [postprocess(sim_data) for sim_data in batch_data.values()]
        return 0 * sum(values)

    ag.grad(objective)(params0)

    assert captured_task_names.issuperset(task_names)
    assert any("_parallel_adj_" in task_name for task_name in captured_task_names)
