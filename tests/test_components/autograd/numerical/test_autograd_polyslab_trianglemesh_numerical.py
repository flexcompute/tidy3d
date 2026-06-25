# Test autograd and compare to numerically computed finite difference gradients for
# PolySlab and TriangleMesh geometries representing the same rectangular slab.
from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from pydantic import BaseModel

import tidy3d as td
from tests.test_components.autograd.numerical.test_autograd_box_polyslab_numerical import (
    dimension_permutation,
    finite_difference,
    make_base_simulation,
    run_parameter_simulations,
    squeeze_dimension,
)
from tidy3d import config

from .numerical_test_helpers import (
    EvaluationData,
    case_identity_id,
    finalize_result,
    gradient_angle_deg,
    load_or_collect_evaluation_data,
)
from .result_models import Metric

config.local_cache.enabled = True
WL_UM = 0.65
FREQ0 = td.C_0 / WL_UM
PERIODS_UM = (3 * WL_UM, 4 * WL_UM)
INFINITE_DIM_SIZE_UM = 0.1
SRC_OFFSET = -2.5
MONITOR_OFFSET = 2.5
PERMITTIVITY = 2.5**2
MESH_SPACING_UM = WL_UM / 40.0
FINITE_DIFFERENCE_STEP = MESH_SPACING_UM
LOCAL_GRADIENT = True
VERBOSE = False
PLOT_FD_ADJ_COMPARISON = False
COMPARE_TO_FINITE_DIFFERENCE = True
COMPARE_TO_POLYSLAB = True

ANGLE_OVERLAP_THRESH_DEG = 10.0
ANGLE_OVERLAP_FD_ADJ_THRESH_DEG = 10.0

VERTEX_SIGNS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ]
)

TRIANGLE_FACE_VERTEX_IDS = np.array(
    [
        (1, 3, 0),
        (4, 1, 0),
        (0, 3, 2),
        (2, 4, 0),
        (1, 7, 3),
        (5, 1, 4),
        (5, 7, 1),
        (3, 7, 2),
        (6, 4, 2),
        (2, 7, 6),
        (6, 5, 4),
        (7, 5, 6),
    ],
    dtype=int,
)

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


def _triangles_from_params(params, box_center):
    params_arr = anp.array(params)
    center_arr = anp.array(box_center)
    half_size = 0.5 * params_arr
    vertices = center_arr + anp.array(VERTEX_SIGNS) * half_size
    return vertices[anp.array(TRIANGLE_FACE_VERTEX_IDS)]


def make_trianglemesh_geometry(params, box_center):
    triangles = _triangles_from_params(params, box_center)
    mesh = td.TriangleMesh.from_triangles(triangles)
    return mesh


def make_polyslab_geometry(params, box_center, axis: int) -> td.PolySlab:
    half_size = 0.5 * params
    slab_bounds = (
        box_center[axis] - half_size[axis],
        box_center[axis] + half_size[axis],
    )
    plane_axes = [idx for idx in range(3) if idx != axis]

    vertices = []
    for sign_0, sign_1 in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        coord_0 = box_center[plane_axes[0]] + sign_0 * half_size[plane_axes[0]]
        coord_1 = box_center[plane_axes[1]] + sign_1 * half_size[plane_axes[1]]
        vertices.append((coord_0, coord_1))

    return td.PolySlab(vertices=tuple(vertices), slab_bounds=slab_bounds, axis=axis)


def make_objective(
    make_geometry,
    box_center,
    tag: str,
    base_sim: td.Simulation,
    fom,
    tmp_path,
    *,
    local_gradient: bool,
    fixed_grid_spec: td.GridSpec | None = None,
):
    objective_base_sim = base_sim
    if fixed_grid_spec is not None:
        objective_base_sim = base_sim.updated_copy(grid_spec=fixed_grid_spec, validate=True)

    def objective(parameters):
        results = run_parameter_simulations(
            parameters,
            make_geometry,
            box_center,
            tag,
            objective_base_sim,
            fom,
            tmp_path,
            local_gradient=local_gradient,
        )

        return results

    return objective


def fixed_grid_spec_for_parameters(
    make_geometry,
    box_center,
    params: anp.ndarray,
    base_sim: td.Simulation,
) -> td.GridSpec:
    """Resolve the unperturbed simulation grid and reuse it for finite differences."""
    geometry = make_geometry(params, box_center)
    structure = td.Structure(
        geometry=geometry,
        medium=td.Medium(permittivity=PERMITTIVITY),
    )
    sim = base_sim.updated_copy(structures=[structure], validate=True)
    return td.GridSpec.from_grid(sim.grid)


class PolySlabTriangleMeshCaseIdentity(BaseModel):
    """Semantic identity for one PolySlab/TriangleMesh comparison."""

    is_3d: bool
    infinite_dim_2d: int
    shift_box_center: bool
    compare_to_finite_difference: bool
    compare_to_polyslab: bool


POLYSLAB_TRIANGLEMESH_CASES = [
    PolySlabTriangleMeshCaseIdentity(
        is_3d=is_3d,
        infinite_dim_2d=infinite_dim_2d,
        shift_box_center=shift_box_center,
        compare_to_finite_difference=COMPARE_TO_FINITE_DIFFERENCE,
        compare_to_polyslab=COMPARE_TO_POLYSLAB,
    )
    for is_3d, infinite_dim_2d in [
        (True, 2),
        (False, 0),
        (False, 1),
        (False, 2),
    ]
    for shift_box_center in (True, False)
]


def _box_center_for_case(case: PolySlabTriangleMeshCaseIdentity) -> list[float]:
    box_center = [0.0, 0.0, 0.0]
    if case.shift_box_center:
        # test what happens when part of the structure falls outside the simulation domain
        # but don't shift along source axis
        if case.is_3d:
            box_center[0:2] = [0.5 * p for p in PERIODS_UM]
        else:
            _, final_dim_2d = dimension_permutation(case.infinite_dim_2d)
            box_center[case.infinite_dim_2d] = 0.5 * INFINITE_DIM_SIZE_UM
            box_center[final_dim_2d] = 0.5 * PERIODS_UM[0]
    return box_center


def _initial_params_for_case(case: PolySlabTriangleMeshCaseIdentity) -> anp.ndarray:
    if case.shift_box_center:
        slab_init_size = [2.0 * WL_UM, 2.5 * WL_UM, 0.75 * WL_UM]
    else:
        slab_init_size = [1.0 * WL_UM, 1.25 * WL_UM, 0.75 * WL_UM]
    return anp.array(slab_init_size)


def _collect_polyslab_trianglemesh_evaluation_data(
    case: PolySlabTriangleMeshCaseIdentity,
    numerical_case_dir,
) -> EvaluationData:
    base_sim, fom = make_base_simulation(
        case.is_3d,
        case.infinite_dim_2d if not case.is_3d else None,
    )
    initial_params = _initial_params_for_case(case)
    polyslab_axis = 2 if case.is_3d else case.infinite_dim_2d
    box_center = _box_center_for_case(case)
    sim_path_dir = numerical_case_dir / "simulations"
    sim_path_dir.mkdir(parents=True, exist_ok=True)
    fixed_grid_spec = fixed_grid_spec_for_parameters(
        make_trianglemesh_geometry,
        box_center,
        initial_params,
        base_sim,
    )

    triangle_objective = make_objective(
        make_trianglemesh_geometry,
        box_center,
        "trianglemesh",
        base_sim,
        fom,
        sim_path_dir,
        local_gradient=LOCAL_GRADIENT,
        fixed_grid_spec=fixed_grid_spec,
    )

    polyslab_objective = make_objective(
        lambda p, box_center: make_polyslab_geometry(p, box_center, polyslab_axis),
        box_center,
        "polyslab",
        base_sim,
        fom,
        sim_path_dir,
        local_gradient=LOCAL_GRADIENT,
        fixed_grid_spec=fixed_grid_spec,
    )

    triangle_objective_fd = make_objective(
        make_trianglemesh_geometry,
        box_center,
        "trianglemesh_fd",
        base_sim,
        fom,
        sim_path_dir,
        local_gradient=False,
        fixed_grid_spec=fixed_grid_spec,
    )

    _triangle_value, triangle_grad = value_and_grad(triangle_objective)([initial_params])
    triangle_grad_filtered = squeeze_dimension(triangle_grad, case.is_3d, case.infinite_dim_2d)

    _polyslab_value, polyslab_grad = value_and_grad(polyslab_objective)([initial_params])
    polyslab_grad_filtered = squeeze_dimension(polyslab_grad, case.is_3d, case.infinite_dim_2d)

    fd_triangle = squeeze_dimension(
        finite_difference(triangle_objective_fd, initial_params, case.is_3d, case.infinite_dim_2d),
        case.is_3d,
        case.infinite_dim_2d,
    )

    return {
        "fd_triangle": np.asarray(fd_triangle, dtype=float),
        "triangle_grad": np.asarray(triangle_grad_filtered, dtype=float),
        "triangle_grad_full": np.asarray(triangle_grad, dtype=float),
        "polyslab_grad": np.asarray(polyslab_grad_filtered, dtype=float),
    }


def _evaluate_polyslab_trianglemesh_evaluation_data(
    case: PolySlabTriangleMeshCaseIdentity,
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    triangle_grad = np.asarray(evaluation_data["triangle_grad"], dtype=float)
    triangle_grad_full = np.asarray(evaluation_data["triangle_grad_full"], dtype=float)
    polyslab_grad = np.asarray(evaluation_data["polyslab_grad"], dtype=float)
    fd_triangle = np.asarray(evaluation_data["fd_triangle"], dtype=float)
    triangle_grad_norm = float(np.linalg.norm(triangle_grad_full))

    regression_metrics: list[Metric] = []
    observation_metrics: list[Metric] = [
        Metric(
            name="triangle_grad_norm",
            observed=triangle_grad_norm,
            expected=0.0,
            comparator="gte",
        )
    ]

    if case.is_3d or case.infinite_dim_2d not in [1, 2]:
        regression_metrics.append(
            Metric(
                name="triangle_grad_norm_nonzero",
                observed=triangle_grad_norm,
                expected=1e-6,
                comparator="gt",
            )
        )

    diagnostics: dict[str, float] = {}
    if case.compare_to_polyslab:
        triangle_polyslab_overlap_deg = gradient_angle_deg(triangle_grad, polyslab_grad)
        diagnostics["triangle_polyslab_overlap_deg"] = triangle_polyslab_overlap_deg
        regression_metrics.append(
            Metric(
                name="triangle_polyslab_overlap_deg",
                observed=triangle_polyslab_overlap_deg,
                expected=ANGLE_OVERLAP_THRESH_DEG,
                comparator="lt",
            )
        )

    if case.compare_to_finite_difference:
        triangle_fd_adj_overlap_deg = gradient_angle_deg(triangle_grad, fd_triangle)
        diagnostics["triangle_fd_adj_overlap_deg"] = triangle_fd_adj_overlap_deg
        regression_metrics.append(
            Metric(
                name="triangle_fd_adj_overlap_deg",
                observed=triangle_fd_adj_overlap_deg,
                expected=ANGLE_OVERLAP_FD_ADJ_THRESH_DEG,
                comparator="lt",
            )
        )

    return regression_metrics, observation_metrics, diagnostics


def _print_polyslab_trianglemesh_summary(
    case: PolySlabTriangleMeshCaseIdentity,
    diagnostics: dict[str, float],
    *,
    eval_only: bool,
) -> None:
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    print(f"Evaluation mode: {mode_label}")
    print(
        "PolySlab/TriangleMesh case: "
        f"is_3d={case.is_3d}, infinite_dim_2d={case.infinite_dim_2d}, "
        f"shift_box_center={case.shift_box_center}"
    )
    if "triangle_polyslab_overlap_deg" in diagnostics:
        print(
            "TriangleMesh FD vs. polyslab overlap: "
            f"{diagnostics['triangle_polyslab_overlap_deg']:.3f} deg"
        )
    if "triangle_fd_adj_overlap_deg" in diagnostics:
        print(
            "TriangleMesh FD vs. Adjoint angle overlap: "
            f"{diagnostics['triangle_fd_adj_overlap_deg']:.3f} deg"
        )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    POLYSLAB_TRIANGLEMESH_CASES,
    ids=lambda case: case_identity_id(case, prefix="polyslab-trianglemesh"),
)
def test_polyslab_and_trianglemesh_gradients_match(
    request: pytest.FixtureRequest,
    case: PolySlabTriangleMeshCaseIdentity,
    numerical_case_dir,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Compare TriangleMesh gradients to equivalent PolySlab and finite-difference values."""
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case,
        collect_evaluation_data=lambda: _collect_polyslab_trianglemesh_evaluation_data(
            case,
            numerical_case_dir,
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_polyslab_trianglemesh_evaluation_data(case, evaluation_data)
    )
    _print_polyslab_trianglemesh_summary(
        case,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "TriangleMesh gradient comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
