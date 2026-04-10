from __future__ import annotations

from typing import TYPE_CHECKING

import autograd as ag
import autograd.numpy as anp
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web.api.autograd.autograd as autograd_api
import tidy3d.web.api.autograd.parallel_adjoint as parallel_adjoint_api
from tidy3d.components.autograd.parallel_adjoint_bases import (
    DiffractionAdjointBasis,
    ModeAdjointBasis,
    PointFieldAdjointBasis,
)
from tidy3d.components.autograd.source_factory import point_current_source_from_simulation
from tidy3d.components.autograd.utils import adjoint_fwidth_from_simulation
from tidy3d.components.diffraction import COS_THETA_THRESH, diffraction_angle_is_propagating
from tidy3d.config import config
from tidy3d.web import run, run_async
from tidy3d.web.api.autograd.context import AutogradContext, ParallelAdjointState
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

if TYPE_CHECKING:
    from typing import Any, Optional


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
    monkeypatch.setattr(config.adjoint, "parallel_run", False)
    val_seq, grad_seq = ag.value_and_grad(objective)(params0)
    monkeypatch.setattr(config.adjoint, "parallel_run", True)
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
    monkeypatch.setattr(config.adjoint, "parallel_run", True)
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
    monkeypatch.setattr(config.adjoint, "parallel_run", False)
    val_seq, grad_seq = ag.value_and_grad(objective)(params0)
    monkeypatch.setattr(config.adjoint, "parallel_run", True)
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
    monkeypatch.setattr(config.adjoint, "parallel_run", True)

    with AssertLogStr("WARNING", contains_str="unsupported monitors"):
        ag.value_and_grad(objective)(params0)


def test_warn_parallel_adjoint_fallback_ignores_empty_state(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(td.log, "warning", lambda message: warnings.append(message))

    parallel_adjoint_api._warn_parallel_adjoint_fallback(
        parallel_info=None,
        sims_adj=[SIM_BASE],
        task_name="parallel_empty_state",
    )

    assert warnings == []


def test_warn_parallel_adjoint_fallback_warns_for_nonempty_state(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(td.log, "warning", lambda message: warnings.append(message))
    basis = ModeAdjointBasis(
        monitor_index=0,
        monitor_name="monitor",
        freq=1.0,
        direction="+",
        mode_index=0,
        data_path=("data", 0, "amps"),
    )

    parallel_adjoint_api._warn_parallel_adjoint_fallback(
        parallel_info=ParallelAdjointState(
            task_name="task",
            num_sims=1,
            basis_specs=[basis],
            basis_maps={},
            basis_task_map={basis: "task"},
        ),
        sims_adj=[SIM_BASE],
        task_name="parallel_nonempty_state",
    )

    assert len(warnings) == 1
    assert "Parallel adjoint incomplete" in warnings[0]


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
    monkeypatch.setattr(config.adjoint, "parallel_run", False)
    val_seq, grad_seq = ag.value_and_grad(objective)(params0)
    monkeypatch.setattr(config.adjoint, "parallel_run", True)
    val_par, grad_par = ag.value_and_grad(objective)(params0)

    assert np.isclose(val_seq, val_par)
    assert np.allclose(grad_seq, grad_par)


def test_parallel_adjoint_limit_fallback(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure parallel-adjoint cap disables parallel path and falls back sequentially."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        sim = make_sim(*args)
        sim_data = run(
            sim,
            task_name="parallel_adjoint_limit_fallback",
            verbose=False,
            local_gradient=True,
            max_num_adjoint_per_fwd=1,
        )
        return postprocess(sim_data)

    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )
    monkeypatch.setattr(config.adjoint, "parallel_run", False)
    val_seq, grad_seq = ag.value_and_grad(objective)(params0)
    monkeypatch.setattr(config.adjoint, "parallel_run", True)
    with AssertLogLevel("WARNING", contains_str="exceeds max_adjoint_per_fwd=1"):
        val_par, grad_par = ag.value_and_grad(objective)(params0)

    assert np.isclose(val_seq, val_par)
    assert np.allclose(grad_seq, grad_par)


def test_parallel_adjoint_mode_direction_policy_assume_outgoing(monkeypatch):
    """Ensure assume_outgoing keeps only the expected mode direction."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]
    sim = make_sim(params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )

    monkeypatch.setattr(config.adjoint, "parallel_run", True)
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


def test_parallel_adjoint_unresolved_residual_vjp_fails(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure unresolved non-zero residual VJP entries fail parallel fallback."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )
    sim_data = run(
        sim,
        task_name="parallel_unresolved_residual_vjp",
        verbose=False,
    )
    monitor_index, monitor = next(
        (i, m) for i, m in enumerate(sim.monitors) if isinstance(m, td.ModeMonitor)
    )
    mode_data = sim_data[monitor.name]
    vjp = np.zeros_like(mode_data.amps.values, dtype=complex)
    vjp[0, 0, 0] = 1.0 + 0.0j

    monkeypatch.setattr(autograd_api, "setup_adj", lambda **_: [])
    with pytest.raises(
        td.exceptions.AdjointError,
        match="could not resolve remaining non-zero VJP entries",
    ):
        autograd_api._prepare_adjoints_from_vjp(
            data_fields_vjp={("data", monitor_index, "amps"): vjp},
            sim_fields_original=sim_fields,
            sim_data_orig=sim_data,
            sim_fields_keys=list(sim_fields.keys()),
            max_num_adjoint_per_fwd=100,
            parallel_info=ParallelAdjointState(
                task_name="parallel_unresolved_residual_vjp",
                num_sims=0,
                basis_specs=[],
                basis_maps={},
                basis_task_map={},
            ),
            task_name="parallel_unresolved_residual_vjp",
        )


def test_nonparallel_unresolved_vjp_returns_zero_map(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure legacy non-parallel path keeps zero-gradient fallback when setup_adj returns no sims."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )
    sim_data = run(
        sim,
        task_name="nonparallel_unresolved_vjp_zero_map",
        verbose=False,
    )
    monitor_index, monitor = next(
        (i, m) for i, m in enumerate(sim.monitors) if isinstance(m, td.ModeMonitor)
    )
    mode_data = sim_data[monitor.name]
    vjp = np.zeros_like(mode_data.amps.values, dtype=complex)
    vjp[0, 0, 0] = 1.0 + 0.0j

    monkeypatch.setattr(autograd_api, "setup_adj", lambda **_: [])
    with AssertLogLevel("WARNING", contains_str="contains no sources"):
        vjp_traced_fields, sims_adj, has_adj_sources = autograd_api._prepare_adjoints_from_vjp(
            data_fields_vjp={("data", monitor_index, "amps"): vjp},
            sim_fields_original=sim_fields,
            sim_data_orig=sim_data,
            sim_fields_keys=list(sim_fields.keys()),
            max_num_adjoint_per_fwd=100,
            parallel_info=None,
            task_name="nonparallel_unresolved_vjp_zero_map",
        )

    assert not sims_adj
    assert has_adj_sources is False
    assert set(vjp_traced_fields.keys()) == set(sim_fields.keys())
    for key in sim_fields:
        expected = (
            type(sim_fields[key])(0 * x for x in sim_fields[key])
            if isinstance(sim_fields[key], (list, tuple))
            else 0 * sim_fields[key]
        )
        assert np.all(vjp_traced_fields[key] == expected)


def test_parallel_adjoint_diffraction_bases(monkeypatch):
    """Ensure diffraction monitors expose parallel adjoint bases."""

    fn_dict = get_functions("medium", "diff")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )

    monkeypatch.setattr(config.adjoint, "parallel_run", True)

    payload = prepare_parallel_adjoint(
        simulation=sim.to_static(),
        sim_fields_keys=list(sim_fields.keys()),
        task_name="parallel_diffraction_bases",
        max_num_adjoint_per_fwd=100,
    )

    assert payload is not None
    assert any(isinstance(basis, DiffractionAdjointBasis) for basis in payload.basis_specs)


@pytest.mark.parametrize(
    ("angle_theta", "expected"),
    (
        (np.nan, False),
        (0.0, True),
        (np.arccos(COS_THETA_THRESH), False),
    ),
)
def test_diffraction_angle_is_propagating(angle_theta, expected):
    """Ensure the shared diffraction-angle predicate matches basis/source cutoff behavior."""

    assert diffraction_angle_is_propagating(angle_theta) == expected


@pytest.mark.parametrize(
    ("monitor_key", "basis_type"),
    (
        ("mode", ModeAdjointBasis),
        ("field_point", PointFieldAdjointBasis),
    ),
)
@pytest.mark.usefixtures("use_emulated_run")
def test_parallel_adjoint_missing_index_keeps_vjp_for_fallback(monitor_key, basis_type):
    """Missing basis coordinates should not raise and should preserve VJP for sequential fallback."""

    fn_dict = get_functions("medium", monitor_key)
    sim = fn_dict["sim"](params0)
    sim_data = run(
        sim,
        task_name=f"parallel_missing_index_{monitor_key}",
        verbose=False,
    )

    monitor_index, monitor = next(
        (i, m)
        for i, m in enumerate(sim.monitors)
        if (
            isinstance(m, td.ModeMonitor)
            if basis_type is ModeAdjointBasis
            else isinstance(m, td.FieldMonitor) and all(size == 0.0 for size in m.size)
        )
    )
    basis = next(
        b for b in monitor.parallel_adjoint_bases(sim, monitor_index) if isinstance(b, basis_type)
    )
    missing_freq = float(basis.freq) * 1.123456789

    if isinstance(basis, ModeAdjointBasis):
        missing_basis = ModeAdjointBasis(
            monitor_index=basis.monitor_index,
            monitor_name=basis.monitor_name,
            freq=missing_freq,
            direction=basis.direction,
            mode_index=basis.mode_index,
            data_path=basis.data_path,
        )
        values = sim_data[monitor.name].amps.values
    else:
        missing_basis = PointFieldAdjointBasis(
            monitor_index=basis.monitor_index,
            monitor_name=basis.monitor_name,
            freq=missing_freq,
            component=basis.component,
            data_path=basis.data_path,
        )
        values = sim_data[monitor.name].field_components[basis.component].values

    data_fields_vjp = {basis.data_path: np.ones_like(values, dtype=complex)}
    original = data_fields_vjp[basis.data_path].copy()

    with AssertLogStr("WARNING", contains_str="basis_metadata="):
        assert missing_basis.vjp_value(data_fields_vjp, sim_data) == 0.0 + 0.0j
        missing_basis.zero_vjp_entry(data_fields_vjp, sim_data)

    assert np.array_equal(data_fields_vjp[basis.data_path], original)


def test_parallel_adjoint_mode_bases(monkeypatch):
    """Ensure mode monitors expose parallel adjoint bases."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )

    monkeypatch.setattr(config.adjoint, "parallel_run", True)

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
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )

    monkeypatch.setattr(config.adjoint, "parallel_run", True)

    payload = prepare_parallel_adjoint(
        simulation=sim.to_static(),
        sim_fields_keys=list(sim_fields.keys()),
        task_name="parallel_point_field_bases",
        max_num_adjoint_per_fwd=100,
    )

    assert payload is not None
    assert any(isinstance(basis, PointFieldAdjointBasis) for basis in payload.basis_specs)


def test_parallel_adjoint_unused_debug_log(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure a debug log is emitted when parallel adjoint bases are unused."""

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
    monkeypatch.setattr(config.adjoint, "parallel_run", True)

    with AssertLogStr("DEBUG", contains_str="unused"):
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


def test_adjoint_fwidth_from_simulation_requires_sources():
    """Ensure adjoint fwidth helper raises a clear error when a simulation has no sources."""

    class _NoSourceSimulation:
        normalize_index = None
        sources = ()

    with pytest.raises(ValueError, match="has no sources"):
        adjoint_fwidth_from_simulation(_NoSourceSimulation())


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
    parallel_info = parallel_adjoint_api.ParallelAdjointState(
        task_name="assume_outgoing",
        num_sims=1,
        basis_maps=basis_maps,
        basis_specs=[outgoing_basis],
        basis_task_map={outgoing_basis: "assume_outgoing_parallel_adj_0"},
    )

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
    monkeypatch.setattr(config.adjoint, "parallel_run", True)
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


def test_local_backward_batch_does_not_pass_path(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure local adjoint batch calls pass path_dir only (not path)."""

    fn_dict = get_functions("medium", "mode")
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]
    orig_run_async = autograd_api._run_async_tidy3d

    def _run_async_capture(simulations, **run_kwargs):
        if simulations and all("_adjoint_" in task_name for task_name in simulations):
            assert "path" not in run_kwargs
        return orig_run_async(simulations, **run_kwargs)

    monkeypatch.setattr(autograd_api, "_run_async_tidy3d", _run_async_capture)
    monkeypatch.setattr(config.adjoint, "local_gradient", True)
    monkeypatch.setattr(config.adjoint, "parallel_run", False)
    monkeypatch.setattr(config.adjoint, "max_adjoint_per_fwd", 100)
    monkeypatch.setattr(
        config.adjoint, "parallel_adjoint_mode_direction_policy", "run_both_directions"
    )

    def objective(*args):
        sim = make_sim(*args)
        sim_data = run(
            sim,
            task_name="local_backward_no_path_kwarg",
            verbose=False,
            local_gradient=True,
        )
        return postprocess(sim_data)

    ag.grad(objective)(params0)


def test_group_parallel_adjoint_bases_no_extra_fwidth_adjust(monkeypatch):
    """Ensure basis grouping does not re-run adjoint source-width normalization."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    monitor_index, monitor = next(
        (i, m) for i, m in enumerate(sim.monitors) if isinstance(m, td.ModeMonitor)
    )
    basis = next(
        basis
        for basis in monitor.parallel_adjoint_bases(sim, monitor_index)
        if isinstance(basis, ModeAdjointBasis)
    )
    source_info = parallel_adjoint_api.make_source_info_from_simulation(
        simulation=sim,
        basis=basis,
        coefficient=1.0 + 0.0j,
    )
    source = source_info.sources[0]

    def _fail(*_args, **_kwargs):
        raise AssertionError("_adjoint_src_width_single should not be called during grouping.")

    monkeypatch.setattr(td.SimulationData, "_adjoint_src_width_single", _fail)
    port_groups = td.SimulationData._group_adjoint_sources_by_port(
        adj_srcs=[source],
        metadata=[basis],
        adjust_fwidth=False,
    )
    grouped = parallel_adjoint_api._group_parallel_adjoint_bases_by_port(
        simulation=sim.to_static(),
        basis_sources=[(basis, source)],
    )

    assert len(port_groups) == 1
    assert port_groups[0].metadata == (basis,)
    assert len(grouped) == 1
    _, grouped_source_info = grouped[0]
    assert np.isclose(grouped_source_info.sources[0].source_time.fwidth, source.source_time.fwidth)


def test_populate_parallel_adjoint_bases_passes_custom_vjp(use_emulated_run, monkeypatch):  # noqa: F811
    """Ensure parallel basis map postprocessing receives custom VJP and numerical map overrides."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )
    sim_data = run(
        sim,
        task_name="parallel_basis_custom_vjp",
        verbose=False,
    )
    monitor_index, monitor = next(
        (i, m) for i, m in enumerate(sim.monitors) if isinstance(m, td.ModeMonitor)
    )
    basis = next(
        basis
        for basis in monitor.parallel_adjoint_bases(sim, monitor_index)
        if isinstance(basis, ModeAdjointBasis)
    )

    payload = parallel_adjoint_api.ParallelAdjointPayload(
        task_name="parallel_custom_vjp_task",
        basis_specs=[basis],
        sims_adj={},
        task_map={"parallel_custom_vjp_task_parallel_adj_0": [basis]},
    )

    custom_marker = (object(),)
    numerical_map_marker = {0: object()}
    captured: list[tuple[Optional[dict[int, Any]], Optional[tuple[Any, ...]]]] = []

    def _postprocess_capture(
        *,
        sim_data_adj,
        sim_data_orig,
        sim_data_fwd,
        sim_fields_keys,
        numerical_structure_map=None,
        custom_vjp=None,
    ):
        captured.append((numerical_structure_map, custom_vjp))
        return {("structures", 0, "medium", "permittivity"): np.array(2.0)}

    monkeypatch.setattr(parallel_adjoint_api, "postprocess_adj", _postprocess_capture)
    monkeypatch.setattr(
        parallel_adjoint_api, "_adjoint_post_norm_for_basis", lambda _sim_data_adj, _basis: None
    )
    monkeypatch.setattr(
        parallel_adjoint_api, "_select_sim_data_freq", lambda sim_data_adj, _freq: sim_data_adj
    )
    monkeypatch.setattr(
        parallel_adjoint_api, "_with_post_norm", lambda sim_data_adj, _post_norm: sim_data_adj
    )
    monkeypatch.setattr(
        parallel_adjoint_api, "_scale_adjoint_field_data", lambda sim_data_adj, _scale: sim_data_adj
    )
    context = AutogradContext()
    context.simulation_data_original = sim_data
    context.simulation_data_forward = sim_data
    parallel_adjoint_api._populate_parallel_adjoint_bases(
        batch_data={
            "parallel_custom_vjp_task": sim_data,
            "parallel_custom_vjp_task_parallel_adj_0": sim_data,
        },
        task_name="parallel_custom_vjp_task",
        payload=payload,
        sim_fields_keys=list(sim_fields.keys()),
        context=context,
        numerical_structure_map=numerical_map_marker,
        custom_vjp=custom_marker,
    )

    assert len(captured) == 2
    assert all(numerical_map is numerical_map_marker for numerical_map, _ in captured)
    assert all(custom_vjp is custom_marker for _, custom_vjp in captured)
    assert context.parallel_adjoint_state is not None
    basis_map = context.parallel_adjoint_state.basis_maps[basis]
    key = ("structures", 0, "medium", "permittivity")
    assert np.allclose(basis_map["real"][key], 2.0)
    assert np.allclose(basis_map["imag"][key], 2.0)


def test_populate_parallel_adjoint_bases_missing_batch_data_raises(use_emulated_run):  # noqa: F811
    """Missing canonical batch data is a hard error."""

    fn_dict = get_functions("medium", "mode")
    sim = fn_dict["sim"](params0)
    sim_fields = sim._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",),)
    )
    sim_data = run(
        sim,
        task_name="parallel_basis_missing_batch_data",
        verbose=False,
    )
    monitor_index, monitor = next(
        (i, m) for i, m in enumerate(sim.monitors) if isinstance(m, td.ModeMonitor)
    )
    basis = next(
        basis
        for basis in monitor.parallel_adjoint_bases(sim, monitor_index)
        if isinstance(basis, ModeAdjointBasis)
    )

    payload = parallel_adjoint_api.ParallelAdjointPayload(
        task_name="parallel_missing_batch_task",
        basis_specs=[basis],
        sims_adj={},
        task_map={"parallel_missing_batch_task_parallel_adj_0": [basis]},
    )
    context = AutogradContext(simulation_data_original=sim_data, simulation_data_forward=sim_data)

    with pytest.raises(td.exceptions.AdjointError, match="unexpectedly missing"):
        parallel_adjoint_api._populate_parallel_adjoint_bases(
            batch_data={"parallel_missing_batch_task": sim_data},
            task_name="parallel_missing_batch_task",
            payload=payload,
            sim_fields_keys=list(sim_fields.keys()),
            context=context,
        )
