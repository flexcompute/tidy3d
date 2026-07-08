from __future__ import annotations

from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from autograd.tracer import getval
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web

from .numerical_test_helpers import (
    EvaluationData,
    case_identity_id,
    condition_metric,
    evaluate_allclose_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)
from .result_models import Metric

THETAS_DEG = [-10, -5, 0, 5, 10]
AXES = [0, 1, 2]
REF_PLANES = ["bottom", "middle", "top"]
LOCAL_GRADIENT = True
VERBOSE = False


class PolySlabSidewallCaseIdentity(BaseModel):
    """Semantic identity for one PolySlab sidewall finite-difference comparison."""

    theta0_deg: float
    axis: int
    reference_plane: str
    fd_step_deg: float
    rtol: float
    atol: float
    domain: tuple[float, float, float]
    min_steps_per_wvl: int
    uniform_dx: float | None


def _build_sim(
    theta_deg: float,
    case: PolySlabSidewallCaseIdentity,
) -> td.Simulation:
    """Construct simulation matching the runner's setup (Gaussian beam + field monitor)."""
    wvl = 1.0
    freq0 = td.C_0 / wvl

    half_x = case.domain[0] / 2.0
    src_x = -(half_x - 0.1)
    mnt_x = half_x - 0.1

    src = td.GaussianBeam(
        center=(src_x, 0, 0),
        size=(0, td.inf, td.inf),
        direction="+",
        waist_radius=0.5,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
    )
    mnt = td.FieldMonitor(
        size=(0, 3, 3),
        center=(mnt_x, 0, 0),
        freqs=[freq0],
        name="field",
    )

    theta = anp.deg2rad(theta_deg)
    verts = anp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    geom = td.PolySlab(
        vertices=verts,
        slab_bounds=(-1, 1),
        axis=case.axis,
        sidewall_angle=theta,
        dilation=0.0,
        reference_plane=case.reference_plane,
    )
    struct = td.Structure(geometry=geom, medium=td.Medium(permittivity=2))

    grid_spec = (
        td.GridSpec.uniform(dl=case.uniform_dx)
        if case.uniform_dx is not None
        else td.GridSpec.auto(min_steps_per_wvl=case.min_steps_per_wvl)
    )

    return td.Simulation(
        size=case.domain,
        grid_spec=grid_spec,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[src],
        structures=[struct],
        monitors=[mnt],
        run_time=1e-12,
    )


def _objective(
    theta_deg: float,
    case: PolySlabSidewallCaseIdentity,
    case_dir,
    verbose: bool,
) -> float:
    sim = _build_sim(theta_deg, case)
    task_name = f"obj_axis{case.axis}_ref{case.reference_plane}_t{float(getval(theta_deg)):+0.3f}"
    out_path = case_dir / f"{task_name}.hdf5"
    data = web.run(
        sim,
        task_name=task_name,
        local_gradient=LOCAL_GRADIENT,
        verbose=verbose,
        path=str(out_path),
    )
    return data["field"].flux.item()


POLYSLAB_SIDEWALL_CASES = [
    PolySlabSidewallCaseIdentity(
        theta0_deg=theta0_deg,
        axis=axis,
        reference_plane=reference_plane,
        fd_step_deg=2e-2,
        rtol=5e-2,
        atol=1e-3,
        domain=(4.0, 4.0, 4.0),
        min_steps_per_wvl=30,
        uniform_dx=None,
    )
    for axis in AXES
    for reference_plane in REF_PLANES
    for theta0_deg in THETAS_DEG
]


def _collect_polyslab_sidewall_evaluation_data(
    case: PolySlabSidewallCaseIdentity,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect adjoint and finite-difference gradients for one sidewall-angle case."""
    objective_dir = numerical_case_dir / "objective"
    objective_dir.mkdir(parents=True, exist_ok=True)

    obj_fun = lambda tdeg: _objective(
        tdeg,
        case,
        objective_dir,
        VERBOSE,
    )
    obj, grad_adj = value_and_grad(obj_fun)(anp.array(case.theta0_deg))

    uid = f"axis{case.axis}_ref{case.reference_plane}_t{case.theta0_deg:+0.3f}"
    sims = {
        f"plus_{uid}": _build_sim(
            case.theta0_deg + case.fd_step_deg,
            case,
        ),
        f"minus_{uid}": _build_sim(
            case.theta0_deg - case.fd_step_deg,
            case,
        ),
    }
    fd_dir = numerical_case_dir / "finite_difference"
    fd_dir.mkdir(parents=True, exist_ok=True)

    datas = web.run_async(
        sims,
        path_dir=str(fd_dir),
        local_gradient=LOCAL_GRADIENT,
        verbose=VERBOSE,
    )

    obj_plus = float(datas[f"plus_{uid}"]["field"].flux.item())
    obj_minus = float(datas[f"minus_{uid}"]["field"].flux.item())

    return {
        "objective_value": float(obj),
        "grad_adj": float(grad_adj),
        "obj_plus": obj_plus,
        "obj_minus": obj_minus,
    }


def _evaluate_polyslab_sidewall_evaluation_data(
    case: PolySlabSidewallCaseIdentity,
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    """Evaluate saved-or-fresh sidewall-angle data into RFC-style metrics."""
    grad_adj = float(np.asarray(evaluation_data["grad_adj"], dtype=float))
    obj_plus = float(np.asarray(evaluation_data["obj_plus"], dtype=float))
    obj_minus = float(np.asarray(evaluation_data["obj_minus"], dtype=float))
    grad_fd = (obj_plus - obj_minus) / (2 * case.fd_step_deg)
    regression_metrics = [
        condition_metric("adjoint_gradient_finite", np.isfinite(grad_adj)),
    ]
    allclose_metrics, observation_metrics, diagnostics = evaluate_allclose_agreement(
        grad_adj,
        grad_fd,
        rtol=case.rtol,
        atol=case.atol,
        metric_name="sidewall_gradient_scaled_error",
    )
    regression_metrics.extend(allclose_metrics)
    diagnostics.update(
        {
            "grad_adj": grad_adj,
            "grad_fd": grad_fd,
        }
    )
    return regression_metrics, observation_metrics, diagnostics


def _print_polyslab_sidewall_summary(
    case: PolySlabSidewallCaseIdentity,
    diagnostics: dict[str, float],
) -> None:
    """Print the original PolySlab sidewall diagnostic summary."""
    print("\n" * 3)
    print("-" * 20)
    print(f"axis: {case.axis}")
    print(f"reference_plane: {case.reference_plane}")
    print(f"theta0_deg: {case.theta0_deg}")
    print(f"Grad (adjoint): {diagnostics['grad_adj']}")
    print(f"Grad (finite difference): {diagnostics['grad_fd']}")
    print("-" * 20)
    print("\n" * 3)


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    POLYSLAB_SIDEWALL_CASES,
    ids=lambda case: case_identity_id(case, prefix="sidewall"),
)
def test_autograd_polyslab_sidewall_vs_fd(
    request: pytest.FixtureRequest,
    case: PolySlabSidewallCaseIdentity,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Adjoint dJ/dtheta matches centered FD for PolySlab.sidewall_angle across axes/ref planes."""
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case,
        collect_evaluation_data=lambda: _collect_polyslab_sidewall_evaluation_data(
            case, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_polyslab_sidewall_evaluation_data(case, evaluation_data)
    )
    _print_polyslab_sidewall_summary(case, diagnostics)
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "PolySlab sidewall finite-difference comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
