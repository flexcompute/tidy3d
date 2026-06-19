"""Flux monitor autograd integration tests."""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np
import numpy.testing as npt
import pytest

import tidy3d as td
from tidy3d._testing.synthetic_monitor_data import make_simulation_data
from tidy3d.components.autograd.flux_monitor import is_flux_adjoint_helper_name
from tidy3d.exceptions import AdjointError
from tidy3d.web import run
from tidy3d.web.api.autograd import io_utils as autograd_io_utils
from tidy3d.web.api.autograd import strategy as autograd_strategy
from tidy3d.web.api.autograd.autograd import run_async_custom, run_custom
from tidy3d.web.api.autograd.context import AdjointTaskContext, AutogradContext, ForwardTaskContext
from tidy3d.web.api.autograd.flux_monitor import expand_flux_monitor_vjps
from tidy3d.web.api.autograd.types import NumericalStructureConfig

from ...utils import AssertLogStr
from .test_autograd import (
    FREQ0,
    FWIDTH,
    SIM_BASE,
    make_structures,
    params0,
    use_emulated_run,  # noqa: F401
)

pytestmark = pytest.mark.usefixtures("use_emulated_run")


def _field_monitors_for_flux_monitor(
    flux_monitor: td.FluxMonitor,
) -> tuple[tuple[td.FieldMonitor, ...], tuple[int, ...]]:
    """FieldMonitor.flux monitors equivalent to one FluxMonitor (same integration scheme)."""
    surfaces = flux_monitor.integration_surfaces
    signs = tuple(1 if surface.normal_dir == "+" else -1 for surface in surfaces)
    field_monitors = tuple(
        td.FieldMonitor(
            size=surface.size,
            center=surface.center,
            freqs=flux_monitor.freqs,
            name=f"field_{surface_index}",
            colocate=flux_monitor.use_colocated_integration,
            use_colocated_integration=flux_monitor.use_colocated_integration,
        )
        for surface_index, surface in enumerate(surfaces)
    )
    return field_monitors, signs


def _simulation_for_flux_monitor(flux_monitor: td.FluxMonitor) -> td.Simulation:
    """Use a coarse 3D grid only for box-flux equivalence coverage."""
    if flux_monitor.size.count(0.0) > 0:
        return SIM_BASE
    return SIM_BASE.updated_copy(
        size=(3.5, SIM_BASE.size[1], SIM_BASE.size[2]),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        grid_spec=td.GridSpec.uniform(dl=0.5),
    )


def _numerical_box_config(parameters) -> NumericalStructureConfig:
    """Numerical structure config with traced parameters but simple static geometry."""

    def create(params):
        permittivity = 2.0 + float(np.asarray(params).flatten()[0])
        return td.Structure(
            geometry=td.Box(size=(0.4, 0.4, 0.4), center=(0, 0, 0)),
            medium=td.Medium(permittivity=permittivity),
        )

    def compute_derivatives(parameters, derivative_info, derivative_helper):
        return dict.fromkeys(derivative_info.paths, 1.0)

    return NumericalStructureConfig(
        create=create,
        compute_derivatives=compute_derivatives,
        parameters=parameters,
    )


def _run_with_numerical_structure(
    sim: td.Simulation,
    parameters,
    use_run_async: bool,
) -> td.SimulationData:
    """Run one simulation through the numerical-structure autograd path."""
    numerical_structure = _numerical_box_config(parameters)
    if use_run_async:
        return run_async_custom(
            {"test_a": sim},
            numerical_structures={"test_a": numerical_structure},
            local_gradient=True,
        )["test_a"]
    return run_custom(
        sim,
        numerical_structures=numerical_structure,
        local_gradient=True,
    )


@pytest.mark.parametrize(
    ("case_name", "flux_monitor"),
    [
        (
            "planar_plus",
            td.FluxMonitor(
                size=(1, 1, 0),
                center=(0, 0, 0),
                freqs=[FREQ0],
                name="flux",
                normal_dir="+",
                enable_adjoint=True,
            ),
        ),
        (
            "planar_minus",
            td.FluxMonitor(
                size=(1, 1, 0),
                center=(0, 0, 0),
                freqs=[FREQ0],
                name="flux",
                normal_dir="-",
                enable_adjoint=True,
            ),
        ),
        (
            "box_exclude_zminus",
            td.FluxMonitor(
                size=(1, 1, 1),
                center=(0, 0, 0),
                freqs=[FREQ0],
                name="flux",
                exclude_surfaces=("z-",),
                enable_adjoint=True,
            ),
        ),
    ],
)
@pytest.mark.parametrize("use_colocated_integration", [True, False])
def test_flux_monitor_adjoint_matches_field_monitor_flux(
    case_name, flux_monitor, use_colocated_integration
):
    """Make sure FluxMonitor adjoints match the established FieldMonitor.flux path, for both
    the colocated and the staggered integration scheme (the hidden helpers follow the parent's
    scheme, so the differentiated functional is the same one in both objectives)."""

    flux_monitor = flux_monitor.updated_copy(use_colocated_integration=use_colocated_integration)
    field_monitors, field_signs = _field_monitors_for_flux_monitor(flux_monitor)
    sim_base = _simulation_for_flux_monitor(flux_monitor)

    def make_objective(monitor_name):
        def objective(params):
            structure_traced = make_structures(params)["medium"]
            monitors = (flux_monitor,) if monitor_name == "flux" else field_monitors
            sim = sim_base.updated_copy(
                structures=(structure_traced,),
                monitors=monitors,
            )
            data = run(
                sim,
                task_name=f"flux_adjoint_{monitor_name}_{case_name}",
                local_gradient=True,
            )
            if monitor_name == "flux":
                return anp.sum(data["flux"].flux.values)
            return sum(
                sign * anp.sum(data[field_monitor.name].flux.values)
                for field_monitor, sign in zip(field_monitors, field_signs)
            )

        return objective

    flux_grad = ag.grad(make_objective("flux"))(params0)
    field_grad = ag.grad(make_objective("field"))(params0)

    assert np.all(np.isfinite(flux_grad))
    npt.assert_allclose(flux_grad, field_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_colocated_integration", [True, False])
def test_flux_monitor_remote_adjoint_matches_field_monitor_flux(use_colocated_integration):
    """Make sure remote FluxMonitor adjoints match the FieldMonitor.flux path, for both
    integration schemes."""

    flux_monitor = td.FluxMonitor(
        size=(1, 1, 0),
        center=(0, 0, 0),
        freqs=[FREQ0],
        name="flux",
        enable_adjoint=True,
        use_colocated_integration=use_colocated_integration,
    )
    field_monitor = td.FieldMonitor(
        size=flux_monitor.size,
        center=flux_monitor.center,
        freqs=flux_monitor.freqs,
        name="field",
        colocate=use_colocated_integration,
        use_colocated_integration=use_colocated_integration,
    )

    def make_objective(monitor_name):
        def objective(params):
            structure_traced = make_structures(params)["medium"]
            monitor = flux_monitor if monitor_name == "flux" else field_monitor
            sim = SIM_BASE.updated_copy(
                structures=(structure_traced,),
                monitors=(monitor,),
            )
            data = run(
                sim,
                task_name=f"flux_equivalence_{monitor_name}_remote",
                local_gradient=False,
            )
            return anp.sum(data[monitor_name].flux.values)

        return objective

    flux_grad = ag.grad(make_objective("flux"))(params0)
    field_grad = ag.grad(make_objective("field"))(params0)

    assert np.all(np.isfinite(flux_grad))
    npt.assert_allclose(flux_grad, field_grad, rtol=1e-12, atol=1e-12)


def test_flux_monitor_vjp_expansion_uses_raw_symmetric_helper_data():
    """Flux helper VJPs should keep stored-grid symmetry metadata for source construction."""

    flux_monitor = td.FluxMonitor(
        size=(1, 1, 0),
        center=(0, 0, 0),
        freqs=[FREQ0],
        name="flux",
        enable_adjoint=True,
    )
    sim = SIM_BASE.updated_copy(
        symmetry=(0, -1, 0),
        monitors=(flux_monitor,),
    )
    sim_data_combined = make_simulation_data(sim._with_adjoint_monitors([]))
    sim_data_orig, sim_data_fwd = sim_data_combined._split_original_fwd(num_mnts_original=1)

    helper_data = sim_data_fwd.data[0].updated_copy(
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        deep=False,
        validate=False,
    )
    sim_data_fwd = sim_data_fwd.updated_copy(data=(helper_data,), deep=False, validate=False)

    _, sim_data_for_adj = expand_flux_monitor_vjps(
        data_fields_vjp={("data", 0, "flux"): np.ones_like(sim_data_orig.data[0].flux.values)},
        sim_data_orig=sim_data_orig,
        sim_data_fwd=sim_data_fwd,
    )

    helper_vjp_data = sim_data_for_adj.data[len(sim_data_orig.data)]
    assert helper_vjp_data.symmetry == sim.symmetry
    for component_name, component_data in helper_vjp_data.field_components.items():
        assert component_data.shape == helper_data.field_components[component_name].shape


def test_flux_monitor_source_gradient_uses_source_adjoint_monitor(monkeypatch):
    """Differentiating flux from a traced source should use source-adjoint monitors."""

    flux_monitor = td.FluxMonitor(
        size=(1, 1, 0),
        center=(0, 0, 0),
        freqs=[FREQ0],
        name="flux",
        enable_adjoint=True,
    )
    coords = {"x": [0.0], "y": [0.0], "z": [0.0], "f": [FREQ0]}
    setup_adj_orig = autograd_strategy.setup_adj
    captured = {"monitor_names": []}

    def setup_adj_capture(*args, **kwargs):
        sims_adj = setup_adj_orig(*args, **kwargs)
        captured["monitor_names"] = [
            monitor.name for sim_adj in sims_adj for monitor in sim_adj.monitors
        ]
        return sims_adj

    monkeypatch.setattr(autograd_strategy, "setup_adj", setup_adj_capture)

    def objective(source_amp):
        source_data = td.ScalarFieldDataArray(
            anp.ones((1, 1, 1, 1)) * source_amp[0],
            coords=coords,
        )
        source = td.CustomCurrentSource(
            center=(0, 0, 0),
            size=(0, 0, 0),
            source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
            current_dataset=td.FieldDataset(Ex=source_data),
        )
        sim = SIM_BASE.updated_copy(
            sources=(source,),
            structures=(),
            monitors=(flux_monitor,),
        )
        data = run(sim, task_name="flux_source_gradient", local_gradient=True)
        return anp.sum(data["flux"].flux.values)

    grad = ag.grad(objective)(anp.array([1.0]))

    assert np.all(np.isfinite(grad))
    assert "source_adjoint_0" in captured["monitor_names"]


def test_remote_flux_monitor_mixed_objective_uses_grouped_adjoint_sim(
    monkeypatch,
):
    """Remote FluxMonitor VJPs should group with regular field-source VJPs."""

    flux_monitor = td.FluxMonitor(
        size=(1, 1, 0),
        center=(0, 0, 0),
        freqs=[FREQ0],
        name="flux",
        enable_adjoint=True,
    )
    field_monitor = td.FieldMonitor(
        size=flux_monitor.size,
        center=flux_monitor.center,
        freqs=flux_monitor.freqs,
        name="field",
        colocate=True,
        use_colocated_integration=True,
    )

    setup_adj_orig = autograd_strategy.setup_adj
    captured = {"num_sims_adj": None}

    def setup_adj_capture(*args, **kwargs):
        sims_adj = setup_adj_orig(*args, **kwargs)
        captured["num_sims_adj"] = len(sims_adj)
        return sims_adj

    monkeypatch.setattr(autograd_strategy, "setup_adj", setup_adj_capture)

    def objective(params):
        structure_traced = make_structures(params)["medium"]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(flux_monitor, field_monitor),
        )
        data = run(sim, task_name="flux_remote_mixed_grouping", local_gradient=False)
        return anp.sum(data["flux"].flux.values + data["field"].flux.values)

    grad = ag.grad(objective)(params0)

    assert np.all(np.isfinite(grad))
    assert captured["num_sims_adj"] == 1, captured


def test_error_flux_enable_adjoint_false():
    """Make sure FluxMonitor adjoint reports the opt-in flag when helper fields are not stored."""

    def objective(params, include_field_monitor):
        structure_traced = make_structures(params)["medium"]
        monitors = [
            td.FluxMonitor(
                size=(1, 1, 0),
                center=(0, 0, 0),
                freqs=[FREQ0],
                name="flux",
            ),
        ]
        if include_field_monitor:
            monitors.append(
                td.FieldMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0], name="field")
            )
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=tuple(monitors),
        )
        data = run(sim, task_name="flux_error", local_gradient=True)
        return anp.sum(data["flux"].flux.values)

    with pytest.raises(AdjointError, match=r"enable_adjoint=True"):
        ag.grad(lambda params: objective(params, include_field_monitor=True))(params0)

    with pytest.raises(AdjointError, match=r"enable_adjoint=True"):
        ag.grad(lambda params: objective(params, include_field_monitor=False))(params0)


@pytest.mark.parametrize("use_run_async", [False, True])
def test_error_flux_enable_adjoint_false_numerical_structures(use_run_async):
    """Numerical-structure autograd should still report the FluxMonitor opt-in flag."""

    def objective(params):
        flux_monitor = td.FluxMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[FREQ0],
            name="flux",
        )
        sim = SIM_BASE.updated_copy(structures=(), monitors=(flux_monitor,))
        data = _run_with_numerical_structure(sim, params, use_run_async)
        return anp.sum(data["flux"].flux.values)

    with pytest.raises(AdjointError, match=r"enable_adjoint=True"):
        ag.grad(objective)(params0)


@pytest.mark.parametrize("use_run_async", [False, True])
def test_flux_monitor_default_false_does_not_warn_when_unused_numerical_structures(use_run_async):
    """Numerical-structure autograd should not warn for unused observational FluxMonitors."""

    def objective(params):
        flux_monitor = td.FluxMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[FREQ0],
            name="flux_untracked_warning_numerical",
        )
        field_monitor = td.FieldMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[FREQ0],
            name="field",
        )
        sim = SIM_BASE.updated_copy(structures=(), monitors=(flux_monitor, field_monitor))
        data = _run_with_numerical_structure(sim, params, use_run_async)
        return anp.sum(data["field"].flux.values)

    with AssertLogStr("WARNING", excludes_str="enable_adjoint"):
        grad = ag.grad(objective)(params0)

    assert np.all(np.isfinite(grad))


def test_flux_monitor_default_false_does_not_warn_when_unused():
    """Autograd should not warn for default FluxMonitors unused by the objective."""

    flux_monitor = td.FluxMonitor(
        size=(1, 1, 0),
        center=(0, 0, 0),
        freqs=[FREQ0],
        name="flux_untracked_warning",
    )
    field_monitor = td.FieldMonitor(
        size=(1, 1, 0),
        center=(0, 0, 0),
        freqs=[FREQ0],
        name="field",
    )

    def objective(params):
        structure_traced = make_structures(params)["medium"]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(flux_monitor, field_monitor),
        )
        data = run(sim, task_name="flux_untracked_warning", local_gradient=True)
        return anp.sum(data["field"].flux.values)

    with AssertLogStr("WARNING", excludes_str="enable_adjoint"):
        grad = ag.grad(objective)(params0)

    assert np.all(np.isfinite(grad))


def test_flux_monitor_adjoint_frequency_inclusion():
    """Check that FluxMonitor frequencies are included only when opted into adjoint fields."""

    monitors_just_field = (
        td.FieldMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[FREQ0],
            name="field",
        ),
    )

    monitors_with_flux = (
        td.FieldMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0], name="field"),
        td.FluxMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[FREQ0 - FWIDTH, FREQ0 + FWIDTH],
            name="flux",
            enable_adjoint=True,
        ),
    )
    monitors_with_flux_untracked = (
        td.FieldMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0], name="field"),
        td.FluxMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[FREQ0 - FWIDTH, FREQ0 + FWIDTH],
            name="flux",
        ),
    )

    def objective_with_monitors(monitors, expected_freqs):
        def objective(params):
            structure_traced = make_structures(params)["medium"]
            sim = SIM_BASE.updated_copy(structures=(structure_traced,), monitors=monitors)
            data = run(sim, task_name="adjoint_freq_test")
            assert data.simulation._freqs_adjoint == expected_freqs
            return anp.sum(data["field"].flux.values)

        return objective

    grad_no_flux_monitors = ag.grad(objective_with_monitors(monitors_just_field, [FREQ0]))(params0)
    grad_with_flux_monitors = ag.grad(
        objective_with_monitors(monitors_with_flux, [FREQ0 - FWIDTH, FREQ0, FREQ0 + FWIDTH])
    )(params0)
    grad_with_flux_untracked = ag.grad(
        objective_with_monitors(monitors_with_flux_untracked, [FREQ0])
    )(params0)
    assert np.all(np.isfinite(grad_no_flux_monitors))
    assert np.all(np.isfinite(grad_with_flux_monitors))
    assert np.all(np.isfinite(grad_with_flux_untracked))
    npt.assert_allclose(grad_with_flux_untracked, grad_no_flux_monitors)


def test_get_autograd_flux_forward_data_missing_artifact_raises_adjoint_error(monkeypatch):
    """Missing hidden flux helper data should report an actionable adjoint error."""

    def fail_download(*args, **kwargs):
        raise FileNotFoundError("missing artifact")

    monkeypatch.setattr(autograd_io_utils, "download_file", fail_download)

    with pytest.raises(
        AdjointError,
        match=r"hidden FluxMonitor forward data artifact.*task_fwd",
    ):
        autograd_io_utils.get_autograd_flux_forward_data("task_fwd", verbose=False)


def test_remote_flux_monitor_untracked_vjp_does_not_download_helper_artifact(monkeypatch):
    """Remote flux VJPs should fail from monitor metadata when adjoint is not enabled."""
    flux_monitor = td.FluxMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[FREQ0],
        name="flux",
    )
    sim = SIM_BASE.updated_copy(monitors=(flux_monitor,))
    context = AutogradContext(
        simulation_data_original=make_simulation_data(sim),
        forward_task_id="task_fwd",
    )
    task_context = AdjointTaskContext.from_inputs(
        task_name="remote_untracked_flux",
        sim_fields_original={},
        context=context,
        max_num_adjoint_per_fwd=1,
        numerical_structures={},
        custom_vjp=None,
        local_gradient=False,
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("helper artifact should not be downloaded")

    monkeypatch.setattr(autograd_strategy, "get_autograd_flux_forward_data", fail_download)

    with pytest.raises(AdjointError, match=r"enable_adjoint=True"):
        autograd_strategy._prepare_adjoints_from_vjp(
            task_context=task_context,
            data_fields_vjp={("data", 0, "flux"): 1.0},
        )


def test_remote_forward_cache_hit_validates_combined_flux_helpers(monkeypatch):
    """Remote cache hits should still validate hidden FluxMonitor helper fields."""
    flux_monitor = td.FluxMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[FREQ0],
        name="flux",
        enable_adjoint=True,
    )
    sim = SIM_BASE.updated_copy(monitors=(flux_monitor,))
    sim_data_orig = make_simulation_data(sim)
    task_context = ForwardTaskContext.from_inputs(
        task_name="cached_flux_forward",
        sim_fields={},
        sim_original=sim,
        context=AutogradContext(),
        max_num_adjoint_per_fwd=1,
        numerical_structures={},
        custom_vjp=None,
    )
    validated_monitor_names = []

    def validate_pre_upload_capture(self):
        validated_monitor_names.append(tuple(monitor.name for monitor in self.monitors))

    monkeypatch.setattr(td.Simulation, "validate_pre_upload", validate_pre_upload_capture)
    monkeypatch.setattr(
        autograd_strategy.webapi,
        "restore_simulation_if_cached",
        lambda **kwargs: ("cached_flux.hdf5", "task_fwd"),
    )
    monkeypatch.setattr(
        autograd_strategy.webapi,
        "load",
        lambda **kwargs: sim_data_orig,
    )

    autograd_strategy.RemoteClientSourceStrategy().run_forward(
        task_context=task_context,
        run_kwargs={},
    )

    assert any(
        any(is_flux_adjoint_helper_name(name) for name in monitor_names)
        for monitor_names in validated_monitor_names
    )


def test_flux_monitor_forward_data_keeps_helpers_internal_in_validation():
    """Helper-only forward data should not re-emit public warnings for hidden helpers."""
    flux_monitor = td.FluxMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[FREQ0],
        name="flux",
        enable_adjoint=True,
    )
    source = td.PlaneWave(
        center=(0, 0, -0.5),
        size=(1, 1, 0),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        direction="+",
        use_colocated_integration=False,
    )
    sim = SIM_BASE.updated_copy(
        sources=(source,),
        monitors=(flux_monitor,),
    )
    sim_data_fwd = make_simulation_data(sim._with_adjoint_monitors([]))

    with AssertLogStr("WARNING", excludes_str="__tidy3d_flux_adjoint"):
        sim_data_flux_fwd = autograd_io_utils.flux_monitor_forward_data(sim_data_fwd)

    assert sim_data_flux_fwd.data
    assert all(is_flux_adjoint_helper_name(data.monitor.name) for data in sim_data_flux_fwd.data)
