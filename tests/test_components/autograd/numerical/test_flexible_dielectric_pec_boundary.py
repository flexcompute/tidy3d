# test autograd for dielectric-dielectric and dielectric-PEC hybrid shape gradients and gradient consistency
# between different structures
from __future__ import annotations

from pathlib import Path

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static
from tidy3d.config import config

from .numerical_test_helpers import (
    EvaluationData,
    GradientComparisonDiagnostics,
    MetricGroups,
    case_identity_id,
    finalize_result,
    load_or_collect_evaluation_data,
)
from .result_models import Metric

ADJ_WVL_UM = 1.5
ADJ_FREQ0 = td.C_0 / ADJ_WVL_UM
DIELECTRIC_LEFT_PERMITTIVITY = 1.5**2
DIELECTRIC_CENTER_PERMITTIVITY = 2.5**2
LOCAL_GRADIENT = True
PLOT_FD_ADJ_COMPARISON = False
RMS_NORMALIZED_THRESHOLD = 1e-12

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


class FlexibleBoundaryCaseIdentity(BaseModel):
    """Semantic identity for one flexible dielectric/PEC boundary case."""

    is_2d: bool
    simulation_background_permittivity: float


flexible_boundary_cases = [
    FlexibleBoundaryCaseIdentity(
        is_2d=is_2d,
        simulation_background_permittivity=simulation_background_permittivity,
    )
    for simulation_background_permittivity in (1.0**2, 1.25**2)
    for is_2d in (True, False)
]


def _collect_flexible_boundary_evaluation_data(
    case_identity: FlexibleBoundaryCaseIdentity,
) -> EvaluationData:
    """Test that we can integrate dielectric-dielectric and dielectric-PEC boundaries on the same structure and that
    we are not sensitive to the structure we choose for autograd when the boundary is shared."""
    is_2d = case_identity.is_2d
    simulation_background_permittivity = case_identity.simulation_background_permittivity
    thickness = 0.0 if is_2d else 0.3 * ADJ_WVL_UM
    pec_material = td.PEC2D if is_2d else td.PECMedium()
    box_2d_dielectric_material_left = td.Medium2D.from_medium(
        td.Medium(permittivity=DIELECTRIC_LEFT_PERMITTIVITY), 0.0
    )
    box_2d_dielectric_material_center = td.Medium2D.from_medium(
        td.Medium(permittivity=DIELECTRIC_CENTER_PERMITTIVITY), 0.0
    )

    box_material_left = (
        box_2d_dielectric_material_left
        if is_2d
        else td.Medium(permittivity=DIELECTRIC_LEFT_PERMITTIVITY)
    )
    box_material_center = (
        box_2d_dielectric_material_center
        if is_2d
        else td.Medium(permittivity=DIELECTRIC_CENTER_PERMITTIVITY)
    )

    def make_structures(left_center_shared_edge, center_right_shared_edge, grad_edge_on_center):
        left_box_edge = (
            get_static(left_center_shared_edge) if grad_edge_on_center else left_center_shared_edge
        )
        center_box_left_edge = (
            left_center_shared_edge if grad_edge_on_center else get_static(left_center_shared_edge)
        )

        right_box_edge = (
            get_static(center_right_shared_edge)
            if grad_edge_on_center
            else center_right_shared_edge
        )
        center_box_right_edge = (
            center_right_shared_edge
            if grad_edge_on_center
            else get_static(center_right_shared_edge)
        )

        left_box_dielectric = td.Box.from_bounds(
            (-ADJ_WVL_UM, -ADJ_WVL_UM, -0.5 * thickness),
            (left_box_edge, ADJ_WVL_UM, 0.5 * thickness),
        )

        center_box_dielectric = td.Box.from_bounds(
            (center_box_left_edge, -ADJ_WVL_UM, -0.5 * thickness),
            (center_box_right_edge, ADJ_WVL_UM, 0.5 * thickness),
        )

        right_box_pec = td.Box.from_bounds(
            (right_box_edge, -ADJ_WVL_UM, -0.5 * thickness),
            (ADJ_WVL_UM, ADJ_WVL_UM, 0.5 * thickness),
        )

        left_box = td.Structure(geometry=left_box_dielectric, medium=box_material_left)

        center_box = td.Structure(
            geometry=center_box_dielectric,
            medium=box_material_center,
            background_medium=td.PECMedium(),
        )

        right_box = td.Structure(geometry=right_box_pec, medium=pec_material)

        return [left_box, center_box, right_box]

    def make_base_sim():
        source = td.PlaneWave(
            center=(0.0, 0.0, -2 * ADJ_WVL_UM),
            size=(td.inf, td.inf, 0.0),
            source_time=td.GaussianPulse(freq0=ADJ_FREQ0, fwidth=0.2 * ADJ_FREQ0),
            direction="+",
        )

        monitor = td.FieldMonitor(
            center=(0.5 * ADJ_WVL_UM, 0.0, 2 * ADJ_WVL_UM),
            size=(0.5 * ADJ_WVL_UM, 0.5 * ADJ_WVL_UM, 0.0),
            freqs=[ADJ_FREQ0],
            name="fields",
        )

        layer_spec = td.LayerRefinementSpec.from_layer_bounds(
            axis=2,
            bounds=(-thickness / 2, thickness / 2),
        )

        grid_spec = td.GridSpec.auto(
            wavelength=ADJ_WVL_UM,
            min_steps_per_wvl=20,
            layer_refinement_specs=[layer_spec],
        )

        boundary_spec = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        )

        return td.Simulation(
            center=(0.0, 0.0, 0.0),
            size=(4.0 * ADJ_WVL_UM, 4 * ADJ_WVL_UM, 6 * ADJ_WVL_UM),
            grid_spec=grid_spec,
            boundary_spec=boundary_spec,
            run_time=1e-11,
            structures=[],
            sources=[source],
            monitors=[monitor],
            medium=td.Medium(permittivity=simulation_background_permittivity),
        )

    def make_obj_fn(grad_edge_on_center):
        def obj_fn(edges):
            left_center_shared_edge, center_right_shared_edge = edges

            base_sim = make_base_sim()

            structures = make_structures(
                left_center_shared_edge, center_right_shared_edge, grad_edge_on_center
            )

            sim_with_structures = base_sim.updated_copy(structures=structures)

            sim_data = web.run(sim_with_structures, local_gradient=LOCAL_GRADIENT)

            fields = sim_data["fields"]

            return np.sum(np.abs(fields.Ex.data) ** 2)

        return obj_fn

    obj_fn_edge_on_center = make_obj_fn(True)
    obj_fn_edge_on_outside = make_obj_fn(False)

    grad_fn_edge_on_center = ag.value_and_grad(obj_fn_edge_on_center)
    grad_fn_edge_on_outside = ag.value_and_grad(obj_fn_edge_on_outside)

    edges = [-0.5 * ADJ_WVL_UM, 0.5 * ADJ_WVL_UM]

    f_edge_on_center, g_edge_on_center = grad_fn_edge_on_center(edges)
    f_edge_on_outside, g_edge_on_outside = grad_fn_edge_on_outside(edges)

    g_edge_on_center = np.array(g_edge_on_center)
    g_edge_on_outside = np.array(g_edge_on_outside)

    return {
        "f_edge_on_center": float(f_edge_on_center),
        "f_edge_on_outside": float(f_edge_on_outside),
        "g_edge_on_center": g_edge_on_center,
        "g_edge_on_outside": g_edge_on_outside,
    }


def _evaluate_flexible_boundary_evaluation_data(evaluation_data: EvaluationData) -> MetricGroups:
    """Evaluate saved-or-fresh flexible-boundary data into RFC-style metrics."""
    g_edge_on_center = np.asarray(evaluation_data["g_edge_on_center"], dtype=float)
    g_edge_on_outside = np.asarray(evaluation_data["g_edge_on_outside"], dtype=float)
    rms_error_normalized = np.sqrt(
        np.mean((g_edge_on_center - g_edge_on_outside) ** 2)
    ) / np.linalg.norm(g_edge_on_outside)
    regression_metrics = [
        Metric(
            name="rms_error_normalized",
            observed=float(rms_error_normalized),
            expected=RMS_NORMALIZED_THRESHOLD,
            comparator="lt",
        )
    ]
    diagnostics = {"rms_error_normalized": float(rms_error_normalized)}
    return regression_metrics, [], diagnostics


def _print_flexible_boundary_summary(
    case_identity: FlexibleBoundaryCaseIdentity,
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    """Print the existing flexible-boundary comparison summary."""
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    g_edge_on_center = np.asarray(evaluation_data["g_edge_on_center"], dtype=float)
    g_edge_on_outside = np.asarray(evaluation_data["g_edge_on_outside"], dtype=float)

    print("\n" * 3)
    print("-" * 20)
    print("Results")
    print(f"Evaluation mode: {mode_label}")
    print(f"is 2d: {case_identity.is_2d}")
    print(f"simulation background permittivity: {case_identity.simulation_background_permittivity}")
    print(
        "function vals (edge on center, edge on outside): "
        f"{evaluation_data['f_edge_on_center']}, {evaluation_data['f_edge_on_outside']}"
    )
    print(f"gradients (edge on center, edge on outside): {g_edge_on_center}, {g_edge_on_outside}")
    print(f"rms error (normalized): {diagnostics['rms_error_normalized']}")
    print("\n" * 3)
    print("-" * 20)


def _plot_flexible_boundary_comparison(
    case_identity: FlexibleBoundaryCaseIdentity,
    evaluation_data: EvaluationData,
) -> None:
    """Plot the existing flexible-boundary gradient comparison."""
    g_edge_on_center = np.asarray(evaluation_data["g_edge_on_center"], dtype=float)
    g_edge_on_outside = np.asarray(evaluation_data["g_edge_on_outside"], dtype=float)
    if PLOT_FD_ADJ_COMPARISON:
        plt.scatter([0, 1], g_edge_on_center, color="b", marker="o", facecolors="none")
        plt.scatter([0, 1], g_edge_on_outside, color="g", marker="x")
        plt.legend(["edges on center box", "edges on outside boxes"])
        plt.title(
            "Test: "
            f"is_2d: {case_identity.is_2d}\n"
            "simulation background n: "
            f"{np.sqrt(case_identity.simulation_background_permittivity)}"
        )
        plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case_identity",
    flexible_boundary_cases,
    ids=lambda params: case_identity_id(params, prefix="flexible-boundary"),
)
def test_flexible_boundary_integration(
    request: pytest.FixtureRequest,
    case_identity: FlexibleBoundaryCaseIdentity,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    monkeypatch: pytest.MonkeyPatch,
    redirect_stdout_to_stderr: None,
) -> None:
    """Test shared-boundary consistency for dielectric-dielectric and dielectric-PEC gradients."""
    monkeypatch.setattr(config.adjoint, "default_wavelength_fraction", 0.01)
    monkeypatch.setattr(config.adjoint, "minimum_spacing_fraction", 0.01)

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_flexible_boundary_evaluation_data(case_identity),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_flexible_boundary_evaluation_data(evaluation_data)
    )
    _print_flexible_boundary_summary(
        case_identity,
        evaluation_data,
        diagnostics,
        eval_only=numerical_eval_only,
    )
    _plot_flexible_boundary_comparison(case_identity, evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "RMS error too large; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
