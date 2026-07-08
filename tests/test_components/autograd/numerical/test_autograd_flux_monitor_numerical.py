from __future__ import annotations

import sys
from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static

from .numerical_test_helpers import (
    EvaluationData,
    case_identity_id,
    condition_metric,
    evaluate_allclose_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)
from .result_models import Metric


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


WAVELENGTH = 1.55
FREQ0 = td.C_0 / WAVELENGTH
FWIDTH = FREQ0 / 10
RUN_TIME = 1.5e-12
PARAMS0 = anp.array([2.2])

SIM_SIZE = (2.4 * WAVELENGTH, 2.4 * WAVELENGTH, 3.2 * WAVELENGTH)
BOX_SIZE = (0.7 * WAVELENGTH, 0.7 * WAVELENGTH, 0.45 * WAVELENGTH)
MONITOR_CENTER = (0.0, 0.0, 0.85 * WAVELENGTH)
MONITOR_SIZE = (1.4 * WAVELENGTH, 1.4 * WAVELENGTH, 0.0)

VALUE_RTOL = 5e-3
VALUE_ATOL = 1e-6
GRAD_RTOL = 5e-2
GRAD_ATOL = 1e-5


class FluxMonitorCaseIdentity(BaseModel):
    """Semantic identity for the FluxMonitor online adjoint equivalence case."""

    wavelength: float
    permittivity: float


def _make_simulation(permittivity: float, *, include_flux_monitor: bool) -> td.Simulation:
    source = td.PlaneWave(
        center=(0.0, 0.0, -0.95 * WAVELENGTH),
        size=(1.8 * WAVELENGTH, 1.8 * WAVELENGTH, 0.0),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        direction="+",
    )
    structure = td.Structure(
        geometry=td.Box(size=BOX_SIZE, center=(0.0, 0.0, 0.0)),
        medium=td.Medium(permittivity=permittivity),
    )
    field_monitor = td.FieldMonitor(
        center=MONITOR_CENTER,
        size=MONITOR_SIZE,
        freqs=[FREQ0],
        name="field",
        colocate=True,
        use_colocated_integration=True,
    )

    monitors: list[td.Monitor] = [field_monitor]
    if include_flux_monitor:
        monitors.insert(
            0,
            td.FluxMonitor(
                center=MONITOR_CENTER,
                size=MONITOR_SIZE,
                freqs=[FREQ0],
                name="flux",
                enable_adjoint=True,
            ),
        )

    return td.Simulation(
        size=SIM_SIZE,
        center=(0.0, 0.0, 0.0),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=24, wavelength=WAVELENGTH),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[source],
        structures=[structure],
        monitors=monitors,
        run_time=RUN_TIME,
    )


FLUX_MONITOR_CASE = FluxMonitorCaseIdentity(
    wavelength=WAVELENGTH,
    permittivity=float(PARAMS0[0]),
)


def _collect_flux_monitor_evaluation_data(numerical_case_dir: Path) -> EvaluationData:
    """Collect raw values and gradients needed to compare FluxMonitor and FieldMonitor flux."""
    forward_values = {}

    def flux_objective(params):
        sim = _make_simulation(params[0], include_flux_monitor=True)
        data = web.run(
            sim,
            task_name="flux_monitor_native_equivalence",
            path=str(numerical_case_dir / "flux_monitor_native_equivalence.hdf5"),
            local_gradient=True,
            verbose=False,
        )
        flux_value = data["flux"].flux.values.item()
        field_flux_value = data["field"].flux.values.item()
        forward_values["flux_run"] = (
            float(get_static(flux_value)),
            float(get_static(field_flux_value)),
        )
        return flux_value

    def field_objective(params):
        sim = _make_simulation(params[0], include_flux_monitor=False)
        data = web.run(
            sim,
            task_name="flux_monitor_field_equivalence",
            path=str(numerical_case_dir / "flux_monitor_field_equivalence.hdf5"),
            local_gradient=True,
            verbose=False,
        )
        return data["field"].flux.values.item()

    flux_value, flux_grad = value_and_grad(flux_objective)(PARAMS0)
    field_value, field_grad = value_and_grad(field_objective)(PARAMS0)

    flux_run_value, flux_run_field_value = forward_values["flux_run"]

    return {
        "flux_value": float(flux_value),
        "field_value": float(field_value),
        "flux_grad": np.asarray(flux_grad, dtype=float),
        "field_grad": np.asarray(field_grad, dtype=float),
        "flux_run_value": flux_run_value,
        "flux_run_field_value": flux_run_field_value,
    }


def _evaluate_flux_monitor_evaluation_data(
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    """Evaluate saved-or-fresh FluxMonitor equivalence data into regression metrics."""
    flux_value = float(np.asarray(evaluation_data["flux_value"], dtype=float))
    field_value = float(np.asarray(evaluation_data["field_value"], dtype=float))
    flux_run_value = float(np.asarray(evaluation_data["flux_run_value"], dtype=float))
    flux_run_field_value = float(np.asarray(evaluation_data["flux_run_field_value"], dtype=float))
    flux_grad = np.asarray(evaluation_data["flux_grad"], dtype=float)
    field_grad = np.asarray(evaluation_data["field_grad"], dtype=float)

    regression_metrics = [
        condition_metric("flux_gradient_finite", bool(np.all(np.isfinite(flux_grad)))),
        condition_metric("field_gradient_finite", bool(np.all(np.isfinite(field_grad)))),
        condition_metric("flux_gradient_nonzero", not np.allclose(flux_grad, 0.0)),
        condition_metric("field_gradient_nonzero", not np.allclose(field_grad, 0.0)),
    ]

    for metrics in (
        evaluate_allclose_agreement(
            flux_run_value,
            flux_run_field_value,
            rtol=VALUE_RTOL,
            atol=VALUE_ATOL,
            metric_name="same_run_flux_value_scaled_error",
        ),
        evaluate_allclose_agreement(
            flux_value,
            field_value,
            rtol=VALUE_RTOL,
            atol=VALUE_ATOL,
            metric_name="separate_run_flux_value_scaled_error",
        ),
        evaluate_allclose_agreement(
            flux_grad,
            field_grad,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            metric_name="flux_gradient_scaled_error",
        ),
    ):
        agreement_regression_metrics, _observation_metrics, _diagnostics = metrics
        regression_metrics.extend(agreement_regression_metrics)

    observation_metrics = [
        Metric(
            name="flux_gradient_norm",
            observed=float(np.linalg.norm(flux_grad)),
            expected=0.0,
            comparator="gte",
        ),
        Metric(
            name="field_gradient_norm",
            observed=float(np.linalg.norm(field_grad)),
            expected=0.0,
            comparator="gte",
        ),
    ]
    diagnostics = {
        "flux_value": flux_value,
        "field_value": field_value,
        "flux_run_field_value": flux_run_field_value,
        "flux_grad_norm": float(np.linalg.norm(flux_grad)),
        "field_grad_norm": float(np.linalg.norm(field_grad)),
    }
    return regression_metrics, observation_metrics, diagnostics


def _print_flux_monitor_summary(evaluation_data: EvaluationData) -> None:
    """Print the original FluxMonitor equivalence summary."""
    flux_value = float(np.asarray(evaluation_data["flux_value"], dtype=float))
    field_value = float(np.asarray(evaluation_data["field_value"], dtype=float))
    flux_run_field_value = float(np.asarray(evaluation_data["flux_run_field_value"], dtype=float))
    flux_grad = np.asarray(evaluation_data["flux_grad"], dtype=float)
    field_grad = np.asarray(evaluation_data["field_grad"], dtype=float)

    print(
        "[flux-monitor-equivalence] "
        f"native_value={flux_value:.6e}, field_value={field_value:.6e}, "
        f"same_run_field_value={flux_run_field_value:.6e}, "
        f"native_grad={flux_grad}, field_grad={field_grad}",
        file=sys.stderr,
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case_identity",
    [FLUX_MONITOR_CASE],
    ids=lambda case_identity: case_identity_id(case_identity, prefix="flux-monitor"),
)
def test_flux_monitor_adjoint_matches_field_monitor_flux_online(
    request: pytest.FixtureRequest,
    case_identity: FluxMonitorCaseIdentity,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
):
    """Compare native FluxMonitor value and gradient against FieldMonitor.flux online."""
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_flux_monitor_evaluation_data(numerical_case_dir),
    )
    _print_flux_monitor_summary(evaluation_data)
    regression_metrics, observation_metrics, _diagnostics = _evaluate_flux_monitor_evaluation_data(
        evaluation_data
    )
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "FluxMonitor online adjoint equivalence failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
