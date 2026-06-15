"""Numerical consistency checks for clip-operation gradient formulations."""

from __future__ import annotations

from pathlib import Path

import autograd as ag
import numpy as np
import pytest
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tidy3d.config import config

from .numerical_test_helpers import (
    EvaluationData,
    GradientComparisonDiagnostics,
    MetricGroups,
    case_identity_id,
    evaluate_allclose_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)

pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

LOCAL_GRADIENT = True
ADJ_WVL_UM = 1.5
ADJ_FREQ0 = td.C_0 / ADJ_WVL_UM

DIELECTRIC_PERMITTIVITY = 2.6**2
RMS_NORMALIZED_THRESHOLD = 1e-3
RMS_NORMALIZED_THRESHOLD_PEC = 1e-2
RMS_ABSOLUTE_THRESHOLD = 1e-8

Y_HALF = 0.6 * ADJ_WVL_UM
Z_HALF = 0.18 * ADJ_WVL_UM

UNION_SPLIT_EDGE = 0.0
LEFT_FIXED_EDGE = -0.95 * ADJ_WVL_UM
PRIMARY_RIGHT_FIXED_EDGE = 0.65 * ADJ_WVL_UM
SECONDARY_RIGHT_FIXED_EDGE = 1.15 * ADJ_WVL_UM
EDGE_INTERFACE_WIDTH = 0.35 * ADJ_WVL_UM
EDGE_INTERFACE_PERMITTIVITY = 4.2**2
DIFFERENCE_STRESS_OVERLAP_LEFT = -0.15 * ADJ_WVL_UM


class ClipOperationConsistencyCaseIdentity(BaseModel):
    """Semantic identity for one clip-operation consistency case."""

    material_kind: str
    clip_operation: str


class DifferenceClipEdgeInterfaceCaseIdentity(BaseModel):
    """Semantic identity for one difference edge-interface stress case."""

    material_kind: str


clip_operation_consistency_cases = [
    ClipOperationConsistencyCaseIdentity(
        material_kind=material_kind,
        clip_operation=clip_operation,
    )
    for material_kind in ("dielectric", "pec")
    for clip_operation in ("difference", "intersection", "symmetric_difference", "union")
]

difference_clip_edge_interface_cases = [
    DifferenceClipEdgeInterfaceCaseIdentity(material_kind=material_kind)
    for material_kind in ("dielectric", "pec")
]


def _make_box(xmin: float, xmax: float) -> td.Box:
    """Create a 3D box used by all test geometries."""
    return td.Box.from_bounds((xmin, -Y_HALF, -Z_HALF), (xmax, Y_HALF, Z_HALF))


def _make_base_simulation(material_kind: str) -> td.Simulation:
    """Create a compact 3D simulation used for all consistency checks."""
    source = td.PlaneWave(
        center=(0.0, 0.0, -2.0 * ADJ_WVL_UM),
        size=(td.inf, td.inf, 0.0),
        source_time=td.GaussianPulse(freq0=ADJ_FREQ0, fwidth=0.2 * ADJ_FREQ0),
        direction="+",
    )

    monitor = td.FieldMonitor(
        center=(0.0, 0.0, 2.0 * ADJ_WVL_UM),
        size=(1.2 * ADJ_WVL_UM, 1.2 * ADJ_WVL_UM, 0.0),
        freqs=[ADJ_FREQ0],
        name="fields",
    )

    if material_kind == "pec":
        layer_spec = td.LayerRefinementSpec.from_layer_bounds(axis=2, bounds=(-Z_HALF, Z_HALF))
        grid_spec = td.GridSpec.auto(
            wavelength=ADJ_WVL_UM,
            min_steps_per_wvl=30,
            layer_refinement_specs=[layer_spec],
        )
    elif material_kind == "dielectric":
        grid_spec = td.GridSpec.auto(
            wavelength=ADJ_WVL_UM,
            min_steps_per_wvl=30,
        )
    else:
        raise ValueError(f"Unknown material_kind='{material_kind}'.")

    return td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.5 * ADJ_WVL_UM, 4.0 * ADJ_WVL_UM, 6.0 * ADJ_WVL_UM),
        grid_spec=grid_spec,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        run_time=1e-11,
        medium=td.Medium(permittivity=1.0),
        structures=[],
        sources=[source],
        monitors=[monitor],
    )


def _material_for_case(material_kind: str):
    if material_kind == "dielectric":
        return td.Medium(permittivity=DIELECTRIC_PERMITTIVITY)
    if material_kind == "pec":
        return td.PECMedium()
    raise ValueError(f"Unknown material_kind='{material_kind}'.")


def _initial_edges(clip_operation: str) -> np.ndarray:
    """Initial edge parameters for each clip operation."""
    if clip_operation == "union":
        return np.array([-0.9 * ADJ_WVL_UM, 0.9 * ADJ_WVL_UM])
    if clip_operation == "difference":
        return np.array([-0.9 * ADJ_WVL_UM, -0.15 * ADJ_WVL_UM])
    if clip_operation in {"intersection", "symmetric_difference"}:
        return np.array([-0.15 * ADJ_WVL_UM, 0.65 * ADJ_WVL_UM])
    raise ValueError(f"Unknown clip_operation='{clip_operation}'.")


def _clip_structures(clip_operation: str, edges, medium) -> list[td.Structure]:
    """Construct structure(s) using a ClipOperation."""
    if clip_operation == "union":
        left_edge, right_edge = edges
        box_a = _make_box(left_edge, UNION_SPLIT_EDGE)
        box_b = _make_box(UNION_SPLIT_EDGE, right_edge)
    elif clip_operation == "difference":
        left_edge, overlap_left = edges
        box_a = _make_box(left_edge, PRIMARY_RIGHT_FIXED_EDGE)
        box_b = _make_box(overlap_left, SECONDARY_RIGHT_FIXED_EDGE)
    elif clip_operation in {"intersection", "symmetric_difference"}:
        overlap_left, right_edge = edges
        box_a = _make_box(LEFT_FIXED_EDGE, right_edge)
        box_b = _make_box(overlap_left, SECONDARY_RIGHT_FIXED_EDGE)
    else:
        raise ValueError(f"Unknown clip_operation='{clip_operation}'.")

    clipped_geometry = td.ClipOperation(
        operation=clip_operation,
        geometry_a=box_a,
        geometry_b=box_b,
    )
    return [td.Structure(geometry=clipped_geometry, medium=medium)]


def _equivalent_structures(clip_operation: str, edges, medium) -> list[td.Structure]:
    """Construct structure(s) with equivalent non-clip geometry."""
    if clip_operation == "union":
        left_edge, right_edge = edges
        equivalent_boxes = [
            _make_box(left_edge, UNION_SPLIT_EDGE),
            _make_box(UNION_SPLIT_EDGE, right_edge),
        ]
    elif clip_operation == "difference":
        left_edge, overlap_left = edges
        equivalent_boxes = [_make_box(left_edge, overlap_left)]
    elif clip_operation == "intersection":
        overlap_left, right_edge = edges
        equivalent_boxes = [_make_box(overlap_left, right_edge)]
    elif clip_operation == "symmetric_difference":
        overlap_left, right_edge = edges
        equivalent_boxes = [
            _make_box(LEFT_FIXED_EDGE, overlap_left),
            _make_box(right_edge, SECONDARY_RIGHT_FIXED_EDGE),
        ]
    else:
        raise ValueError(f"Unknown clip_operation='{clip_operation}'.")

    return [td.Structure(geometry=box, medium=medium) for box in equivalent_boxes]


def _difference_edge_interface_structures() -> list[td.Structure]:
    """Extra structures that place a material interface on the overestimated difference bound."""
    edge_interface_box = td.Box.from_bounds(
        (PRIMARY_RIGHT_FIXED_EDGE, -Y_HALF, -Z_HALF),
        (PRIMARY_RIGHT_FIXED_EDGE + EDGE_INTERFACE_WIDTH, Y_HALF, Z_HALF),
    )
    edge_interface_medium = td.Medium(permittivity=EDGE_INTERFACE_PERMITTIVITY)
    return [td.Structure(geometry=edge_interface_box, medium=edge_interface_medium)]


def _difference_stress_clip_structures(edges, medium) -> list[td.Structure]:
    """Difference clip structures tracing box_a right boundary."""
    left_edge, right_edge_a = edges
    box_a = _make_box(left_edge, right_edge_a)
    box_b = _make_box(DIFFERENCE_STRESS_OVERLAP_LEFT, SECONDARY_RIGHT_FIXED_EDGE)
    clipped_geometry = td.ClipOperation(
        operation="difference",
        geometry_a=box_a,
        geometry_b=box_b,
    )
    return [td.Structure(geometry=clipped_geometry, medium=medium)]


def _difference_stress_equivalent_structures(edges, medium) -> list[td.Structure]:
    """Equivalent non-clip structures for the stress configuration.

    The objective is intentionally independent of ``right_edge_a`` so any non-zero
    ``dJ/d(right_edge_a)`` in the clip branch indicates fake edge contributions.
    """
    left_edge, _right_edge_a = edges
    equivalent_box = _make_box(left_edge, DIFFERENCE_STRESS_OVERLAP_LEFT)
    return [td.Structure(geometry=equivalent_box, medium=medium)]


def _evaluate_objective(
    sim_base: td.Simulation,
    structures: list[td.Structure],
    task_name: str,
    grid_fixed: td.Grid,
) -> float:
    sim = sim_base.updated_copy(
        structures=structures,
        grid_spec=td.GridSpec.from_grid(grid_fixed),
    )

    sim_data = web.run(
        sim,
        task_name=task_name,
        local_gradient=LOCAL_GRADIENT,
    )
    fields = sim_data["fields"]

    return np.sum(np.abs(fields.Ex.data) ** 2) + 0.25 * np.sum(np.abs(fields.Ey.data) ** 2)


def _gradient_rtol(material_kind: str) -> float:
    """Return the existing relative gradient tolerance for a material case."""
    if material_kind == "pec":
        return RMS_NORMALIZED_THRESHOLD_PEC
    return RMS_NORMALIZED_THRESHOLD


def _collect_clip_operation_consistency_evaluation_data(
    case_identity: ClipOperationConsistencyCaseIdentity,
) -> EvaluationData:
    """Compare clip-operation gradients against equivalent explicit-geometry formulations."""
    material_kind = case_identity.material_kind
    clip_operation = case_identity.clip_operation
    sim_base = _make_base_simulation(material_kind=material_kind)
    medium = _material_for_case(material_kind)
    edges0 = _initial_edges(clip_operation)

    def fixed_grid_from_equivalent(edges) -> td.Grid:
        structures_equivalent = _equivalent_structures(
            clip_operation=clip_operation,
            edges=edges,
            medium=medium,
        )
        sim_equivalent = sim_base.updated_copy(structures=structures_equivalent)
        return sim_equivalent.grid

    def objective_clip(edges):
        grid_fixed = fixed_grid_from_equivalent(edges)
        return _evaluate_objective(
            sim_base,
            _clip_structures(clip_operation=clip_operation, edges=edges, medium=medium),
            task_name=f"clip_consistency_{clip_operation}_{material_kind}_clipped",
            grid_fixed=grid_fixed,
        )

    def objective_equivalent(edges):
        grid_fixed = fixed_grid_from_equivalent(edges)
        return _evaluate_objective(
            sim_base,
            _equivalent_structures(clip_operation=clip_operation, edges=edges, medium=medium),
            task_name=f"clip_consistency_{clip_operation}_{material_kind}_comparison",
            grid_fixed=grid_fixed,
        )

    clip_value, clip_grad = ag.value_and_grad(objective_clip)(edges0)
    equiv_value, equiv_grad = ag.value_and_grad(objective_equivalent)(edges0)

    clip_grad = np.asarray(clip_grad, dtype=float)
    equiv_grad = np.asarray(equiv_grad, dtype=float)
    return {
        "clip_value": float(clip_value),
        "equiv_value": float(equiv_value),
        "clip_grad": clip_grad,
        "equiv_grad": equiv_grad,
    }


def _evaluate_clip_gradient_agreement(
    evaluation_data: EvaluationData,
    *,
    material_kind: str,
) -> MetricGroups:
    """Evaluate clip/equivalent gradient agreement using the old allclose tolerances."""
    clip_grad = np.asarray(evaluation_data["clip_grad"], dtype=float)
    equiv_grad = np.asarray(evaluation_data["equiv_grad"], dtype=float)
    regression_metrics, observation_metrics, diagnostics = evaluate_allclose_agreement(
        actual=clip_grad,
        desired=equiv_grad,
        rtol=_gradient_rtol(material_kind),
        atol=RMS_ABSOLUTE_THRESHOLD,
        metric_name="gradient_allclose_scaled_error",
    )

    rms_error = float(np.sqrt(np.mean((clip_grad - equiv_grad) ** 2)))
    grad_norm = float(np.linalg.norm(equiv_grad))
    diagnostics["rms_error"] = rms_error
    diagnostics["grad_norm"] = grad_norm
    diagnostics["rms_error_normalized"] = rms_error / max(grad_norm, 1e-12)
    return regression_metrics, observation_metrics, diagnostics


def _print_clip_operation_consistency_summary(
    case_identity: ClipOperationConsistencyCaseIdentity,
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    """Print the existing clip-operation consistency summary."""
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    clip_grad = np.asarray(evaluation_data["clip_grad"], dtype=float)
    equiv_grad = np.asarray(evaluation_data["equiv_grad"], dtype=float)

    print("\n" + "-" * 20)
    print(
        f"material_kind: {case_identity.material_kind}, "
        f"clip_operation: {case_identity.clip_operation}, mode: {mode_label}"
    )
    print(
        "objective values (clip, equivalent): "
        f"{evaluation_data['clip_value']}, {evaluation_data['equiv_value']}"
    )
    print(f"gradients (clip): {clip_grad}")
    print(f"gradients (equivalent): {equiv_grad}")
    print(f"rms_error: {diagnostics['rms_error']}")
    print(f"rms_error_normalized: {diagnostics['rms_error_normalized']}")
    print("-" * 20 + "\n")


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case_identity",
    clip_operation_consistency_cases,
    ids=lambda params: case_identity_id(params, prefix="clip-consistency"),
)
def test_clip_operation_consistency(
    request: pytest.FixtureRequest,
    case_identity: ClipOperationConsistencyCaseIdentity,
    monkeypatch: pytest.MonkeyPatch,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr: None,
) -> None:
    """Compare clip-operation gradients against equivalent explicit-geometry formulations."""
    monkeypatch.setattr(config.adjoint, "default_wavelength_fraction", 0.01)

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_clip_operation_consistency_evaluation_data(
            case_identity
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_clip_gradient_agreement(
        evaluation_data,
        material_kind=case_identity.material_kind,
    )
    _print_clip_operation_consistency_summary(
        case_identity,
        evaluation_data,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Clip-operation gradients do not match equivalent explicit geometry; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


def _collect_difference_clip_edge_interface_evaluation_data(
    case_identity: DifferenceClipEdgeInterfaceCaseIdentity,
) -> EvaluationData:
    """Stress-test difference gradients with an interface at the overestimated clip bound."""
    material_kind = case_identity.material_kind
    sim_base = _make_base_simulation(material_kind=material_kind)
    medium = _material_for_case(material_kind)
    edges0 = np.array([-0.9 * ADJ_WVL_UM, PRIMARY_RIGHT_FIXED_EDGE])
    edge_structures = _difference_edge_interface_structures()

    def fixed_grid_from_equivalent(edges) -> td.Grid:
        structures_equivalent = _difference_stress_equivalent_structures(edges=edges, medium=medium)
        sim_equivalent = sim_base.updated_copy(
            structures=[*structures_equivalent, *edge_structures]
        )
        return sim_equivalent.grid

    def objective_clip(edges):
        if edges[1] <= DIFFERENCE_STRESS_OVERLAP_LEFT:
            raise ValueError("Stress-test parameterization requires right_edge_a > overlap_left.")
        grid_fixed = fixed_grid_from_equivalent(edges)
        return _evaluate_objective(
            sim_base,
            [
                *_difference_stress_clip_structures(edges=edges, medium=medium),
                *edge_structures,
            ],
            task_name=f"clip_consistency_difference_{material_kind}_edge_interface_clipped",
            grid_fixed=grid_fixed,
        )

    def objective_equivalent(edges):
        grid_fixed = fixed_grid_from_equivalent(edges)
        return _evaluate_objective(
            sim_base,
            [
                *_difference_stress_equivalent_structures(edges=edges, medium=medium),
                *edge_structures,
            ],
            task_name=f"clip_consistency_difference_{material_kind}_edge_interface_comparison",
            grid_fixed=grid_fixed,
        )

    clip_value, clip_grad = ag.value_and_grad(objective_clip)(edges0)
    equiv_value, equiv_grad = ag.value_and_grad(objective_equivalent)(edges0)

    clip_grad = np.asarray(clip_grad, dtype=float)
    equiv_grad = np.asarray(equiv_grad, dtype=float)
    return {
        "clip_value": float(clip_value),
        "equiv_value": float(equiv_value),
        "clip_grad": clip_grad,
        "equiv_grad": equiv_grad,
    }


def _print_difference_clip_edge_interface_summary(
    case_identity: DifferenceClipEdgeInterfaceCaseIdentity,
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    """Print the existing edge-interface stress summary."""
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    clip_grad = np.asarray(evaluation_data["clip_grad"], dtype=float)
    equiv_grad = np.asarray(evaluation_data["equiv_grad"], dtype=float)

    print("\n" + "-" * 20)
    print(
        f"material_kind: {case_identity.material_kind}, "
        f"clip_operation: difference (edge-interface stress), mode: {mode_label}"
    )
    print(
        "objective values (clip, equivalent): "
        f"{evaluation_data['clip_value']}, {evaluation_data['equiv_value']}"
    )
    print(f"gradients (clip): {clip_grad}")
    print(f"gradients (equivalent): {equiv_grad}")
    print(f"right-edge component (clip, equivalent): {clip_grad[1]}, {equiv_grad[1]}")
    print(f"rms_error: {diagnostics['rms_error']}")
    print(f"rms_error_normalized: {diagnostics['rms_error_normalized']}")
    print("-" * 20 + "\n")


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case_identity",
    difference_clip_edge_interface_cases,
    ids=lambda params: case_identity_id(params, prefix="difference-edge-interface"),
)
def test_difference_clip_edge_interface_masking(
    request: pytest.FixtureRequest,
    case_identity: DifferenceClipEdgeInterfaceCaseIdentity,
    monkeypatch: pytest.MonkeyPatch,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr: None,
) -> None:
    """Stress-test difference gradients with an interface at the overestimated clip bound."""
    monkeypatch.setattr(config.adjoint, "default_wavelength_fraction", 0.01)

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_difference_clip_edge_interface_evaluation_data(
            case_identity
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_clip_gradient_agreement(
        evaluation_data,
        material_kind=case_identity.material_kind,
    )
    _print_difference_clip_edge_interface_summary(
        case_identity,
        evaluation_data,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Difference clip edge-interface gradients do not match equivalent explicit geometry; "
            f"inspect {numerical_case_dir / 'evaluation_data.npz'} and "
            f"{numerical_case_dir / 'result.json'}"
        ),
    )
