# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import autograd as ag
import numpy as np
import pytest
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.types.base import Size

from .numerical_test_helpers import (
    EvaluationData,
    case_identity_from_parameters,
    case_identity_id,
    evaluate_fd_adjoint_alignment_group,
    finalize_result,
    load_or_collect_evaluation_data,
    select_fd_gradient_from_objective_values,
)
from .result_models import Metric

td.config.logging.level = "ERROR"

PLOT_FD_ADJ_COMPARISON = False
NUM_VERTICES_PER_FD_TESTS = 10
RUN_WITH_FD_CONVERGENCE = True
FD_CONVERGENCE_THRESHOLD = 0.05
MIN_FD_COMPARISON = 2
ALIGNMENT_OVERLAP_THRESHOLD_DEG = 45.0
FIXED_MESH_REFINEMENT_FACTOR = 200.0
LOCAL_GRADIENT = True
VERBOSE = False

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


def get_sim_geometry(mesh_wvl_um):
    return td.Box(size=(5 * mesh_wvl_um, 5 * mesh_wvl_um, 7 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    monitor_size_wvl,
    box_for_override,
    mesh_refinement_factor,
    run_time=1e-8,
):
    sim_geometry = get_sim_geometry(mesh_wvl_um)

    sim_size_um = sim_geometry.size
    sim_center_um = sim_geometry.center

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    dl_design = mesh_wvl_um / mesh_refinement_factor

    mesh_overrides = []
    mesh_overrides.extend(
        [
            td.MeshOverrideStructure(
                geometry=box_for_override,
                dl=[dl_design, dl_design, dl_design],
            ),
        ]
    )

    src_size = (*sim_size_um[0:2], 0)

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um

    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)
    src = td.PlaneWave(
        center=(0, 0, -2 * mesh_wvl_um),
        size=src_size,
        source_time=pulse,
        direction="+",
        pol_angle=np.pi / 4.0,
    )

    field_monitor = td.FieldMonitor(
        # shifted center
        center=(0, 0, 0.25 * sim_size_um[2]),
        size=tuple(dim * mesh_wvl_um for dim in monitor_size_wvl),
        name="monitor_fields",
        freqs=[freq0],
    )

    sim_base = td.Simulation(
        center=sim_center_um,
        size=sim_size_um,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=30,
            wavelength=mesh_wvl_um,
            override_structures=mesh_overrides,
        ),
        structures=[],
        sources=[src],
        monitors=[field_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_objective_function_2D(create_sim_base, eval_fn, polyslab_z_value, sim_path_dir):
    # 2D PEC objective will take in an array of parameters that can be reshaped into
    # an Nx2 array where N is the number of vertices in a polyslab and the  parameters are
    # center (x,y) and lateral size (x,y) assuming the thickness in z is 0
    #
    # box_param_arrays should be a list of such parameters for cases where we
    # are running finite difference simulations and want to put together a
    # whole batch
    def objective(polyslab_param_arrays):
        sim_base = create_sim_base()

        layer_refinement_specs = []

        simulation_dict = {}
        for idx in range(len(polyslab_param_arrays)):
            get_polyslab_params = polyslab_param_arrays[idx]
            reshape_parameters = np.reshape(get_polyslab_params, (NUM_VERTICES_PER_FD_TESTS, 2))

            polyslab_structures = [
                td.Structure(
                    geometry=td.PolySlab(
                        vertices=tuple(map(tuple, reshape_parameters)),
                        slab_bounds=(polyslab_z_value, polyslab_z_value),
                    ),
                    medium=td.PEC2D,
                )
            ]

            layer_refinement_specs.append(
                td.LayerRefinementSpec.from_layer_bounds(
                    axis=2,
                    bounds=(polyslab_z_value, polyslab_z_value),
                )
            )

            sim_with_block = sim_base.updated_copy(
                structures=tuple(list(sim_base.structures) + polyslab_structures),
                grid_spec=sim_base.grid_spec.updated_copy(
                    layer_refinement_specs=layer_refinement_specs
                ),
            )

            simulation_dict[f"numerical_rf_polyslab_2d_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(polyslab_param_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_rf_polyslab_2d_testing_{idx}"]))

        if len(polyslab_param_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def create_objective_function_3D(
    create_sim_base, eval_fn, polyslab_z_value, polyslab_z_thickness, sim_path_dir
):
    # 2D PEC objective will take in an array of parameters that can be reshaped into
    # an Nx2 array where N is the number of vertices in a polyslab and the  parameters are
    # center (x,y) and lateral size (x,y) assuming the thickness in z is 0
    #
    # box_param_arrays should be a list of such parameters for cases where we
    # are running finite difference simulations and want to put together a
    # whole batch
    def objective(polyslab_param_arrays):
        sim_base = create_sim_base()

        layer_refinement_specs = []

        simulation_dict = {}
        for idx in range(len(polyslab_param_arrays)):
            get_polyslab_params = polyslab_param_arrays[idx]
            reshape_parameters = np.reshape(get_polyslab_params, (NUM_VERTICES_PER_FD_TESTS, 2))

            polyslab_structures = [
                td.Structure(
                    geometry=td.PolySlab(
                        vertices=tuple(map(tuple, reshape_parameters)),
                        slab_bounds=(
                            polyslab_z_value - 0.5 * polyslab_z_thickness,
                            polyslab_z_value + 0.5 * polyslab_z_thickness,
                        ),
                    ),
                    medium=td.PECMedium(),
                )
            ]

            layer_refinement_specs.append(
                td.LayerRefinementSpec.from_layer_bounds(
                    axis=2,
                    bounds=(
                        polyslab_z_value - 0.5 * polyslab_z_thickness,
                        polyslab_z_value + 0.5 * polyslab_z_thickness,
                    ),
                )
            )

            sim_with_block = sim_base.updated_copy(
                structures=tuple(list(sim_base.structures) + polyslab_structures),
                grid_spec=sim_base.grid_spec.updated_copy(
                    layer_refinement_specs=layer_refinement_specs
                ),
            )

            simulation_dict[f"numerical_rf_polyslab_3d_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(polyslab_param_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_rf_polyslab_3d_testing_{idx}"]))

        if len(polyslab_param_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def make_eval_fns(monitor_size_wvl):
    def intensity(sim_data):
        field_data = sim_data["monitor_fields"]
        _shape_x, _shape_y, _shape_z, *_ = field_data.Ex.values.shape

        proj_pol_rotation = field_data.Ex.values - field_data.Ey.values
        return np.sum(np.abs(proj_pol_rotation) ** 2)

    eval_fns = [intensity]
    eval_fn_names = ["intensity"]

    return eval_fns, eval_fn_names


def generate_polyslab(min_dim_lateral, max_dim_lateral, rng):
    x_radius = 0.5 * rng.uniform(min_dim_lateral, max_dim_lateral)
    y_radius = 0.5 * rng.uniform(min_dim_lateral, max_dim_lateral)
    rotation = rng.uniform(0, 2 * np.pi)

    theta = np.linspace(0, 2 * np.pi, NUM_VERTICES_PER_FD_TESTS, endpoint=False)

    x_values = x_radius * np.cos(theta)
    y_values = y_radius * np.sin(theta)

    def rotate(x, y, phi):
        x_rotate = x * np.cos(phi) + y * np.sin(phi)
        y_rotate = -x * np.sin(phi) + y * np.cos(phi)

        return x_rotate, y_rotate

    x_rotated, y_rotated = rotate(x_values, y_values, rotation)

    return np.array(list(zip(x_rotated, y_rotated))).flatten()


def run_and_process_fd(polyslab_parameters, fd_step, objective):
    objective_values_by_step = []
    for fd_step_idx in range(len(fd_step)):
        fd_polyslab_parameters = []

        for param_idx in range(len(polyslab_parameters)):
            copy_params_up = polyslab_parameters.copy()
            copy_params_down = polyslab_parameters.copy()

            copy_params_up[param_idx] += fd_step[fd_step_idx]
            copy_params_down[param_idx] -= fd_step[fd_step_idx]

            fd_polyslab_parameters.append(copy_params_up)
            fd_polyslab_parameters.append(copy_params_down)

        objective_values_by_step.append(np.asarray(objective(fd_polyslab_parameters), dtype=float))

    return np.asarray(objective_values_by_step, dtype=float)


mm = 1e3

background_indices = [1.0, 1.5]
mesh_wvls_mm = [15.0, 15.0]
adj_wvls_mm = [15.0, 20.0]

mesh_refinement_factors = np.linspace(200.0, 300.0, 4)

polyslab_z_thickneses_3d_wvl = np.linspace(0.2, 0.4, 4)

mesh_wvls_um = [mesh_wvl_mm * mm for mesh_wvl_mm in mesh_wvls_mm]
adj_wvls_um = [adj_wvl_mm * mm for adj_wvl_mm in adj_wvls_mm]

monitor_size_3d_wvl = (1.0, 1.0, 0)


class RFPolySlab2DCaseIdentity(BaseModel):
    """Semantic identity for one 2D RF PEC PolySlab finite-difference case."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_size_wvl: Size
    num_vertices_per_fd_tests: int
    mesh_refinement_factor: float
    eval_fn_name: str
    run_with_fd_convergence: bool
    fd_convergence_threshold: float
    min_fd_comparison: int
    alignment_overlap_threshold_deg: float


class RFPolySlab2DTestParameters(RFPolySlab2DCaseIdentity):
    """Full parameter bundle for one 2D RF PolySlab test invocation."""

    eval_fn: Callable
    test_number: int


class RFPolySlab3DCaseIdentity(BaseModel):
    """Semantic identity for one 3D RF PEC PolySlab finite-difference case."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_size_wvl: Size
    num_vertices_per_fd_tests: int
    polyslab_z_thickness_wvl: float
    fixed_mesh_refinement_factor: float
    eval_fn_name: str
    run_with_fd_convergence: bool
    fd_convergence_threshold: float
    min_fd_comparison: int
    alignment_overlap_threshold_deg: float


class RFPolySlab3DTestParameters(RFPolySlab3DCaseIdentity):
    """Full parameter bundle for one 3D RF PolySlab test invocation."""

    eval_fn: Callable
    test_number: int


rf_2d_test_parameters: list[RFPolySlab2DTestParameters] = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_3d_wvl)

    for mesh_refinement_factor in mesh_refinement_factors:
        for eval_fn_idx, eval_fn in enumerate(eval_fns):
            rf_2d_test_parameters.append(
                RFPolySlab2DTestParameters(
                    mesh_wvl_um=mesh_wvl_um,
                    adj_wvl_um=adj_wvl_um,
                    monitor_size_wvl=monitor_size_3d_wvl,
                    num_vertices_per_fd_tests=NUM_VERTICES_PER_FD_TESTS,
                    mesh_refinement_factor=mesh_refinement_factor,
                    eval_fn=eval_fn,
                    eval_fn_name=eval_fn_names[eval_fn_idx],
                    run_with_fd_convergence=RUN_WITH_FD_CONVERGENCE,
                    fd_convergence_threshold=FD_CONVERGENCE_THRESHOLD,
                    min_fd_comparison=MIN_FD_COMPARISON,
                    alignment_overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
                    test_number=test_number,
                )
            )

            test_number += 1


rf_3d_test_parameters: list[RFPolySlab3DTestParameters] = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_3d_wvl)

    for polyslab_z_thickness_wvl in polyslab_z_thickneses_3d_wvl:
        for eval_fn_idx, eval_fn in enumerate(eval_fns):
            rf_3d_test_parameters.append(
                RFPolySlab3DTestParameters(
                    mesh_wvl_um=mesh_wvl_um,
                    adj_wvl_um=adj_wvl_um,
                    monitor_size_wvl=monitor_size_3d_wvl,
                    num_vertices_per_fd_tests=NUM_VERTICES_PER_FD_TESTS,
                    polyslab_z_thickness_wvl=polyslab_z_thickness_wvl,
                    fixed_mesh_refinement_factor=FIXED_MESH_REFINEMENT_FACTOR,
                    eval_fn=eval_fn,
                    eval_fn_name=eval_fn_names[eval_fn_idx],
                    run_with_fd_convergence=RUN_WITH_FD_CONVERGENCE,
                    fd_convergence_threshold=FD_CONVERGENCE_THRESHOLD,
                    min_fd_comparison=MIN_FD_COMPARISON,
                    alignment_overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
                    test_number=test_number,
                )
            )

            test_number += 1


def _case_identity_2d(parameters: RFPolySlab2DTestParameters) -> RFPolySlab2DCaseIdentity:
    """Build the semantic case identity for a 2D RF PolySlab case."""
    return case_identity_from_parameters(RFPolySlab2DCaseIdentity, parameters)


def _case_identity_3d(parameters: RFPolySlab3DTestParameters) -> RFPolySlab3DCaseIdentity:
    """Build the semantic case identity for a 3D RF PolySlab case."""
    return case_identity_from_parameters(RFPolySlab3DCaseIdentity, parameters)


def _collect_rf_polyslab_2d_evaluation_data(
    rf_2d_test_parameters: RFPolySlab2DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect raw FD and adjoint data for one 2D RF PEC PolySlab case."""
    mesh_wvl_um = rf_2d_test_parameters.mesh_wvl_um
    adj_wvl_um = rf_2d_test_parameters.adj_wvl_um
    monitor_size_wvl = rf_2d_test_parameters.monitor_size_wvl
    mesh_refinement_factor = rf_2d_test_parameters.mesh_refinement_factor
    eval_fn = rf_2d_test_parameters.eval_fn
    test_number = rf_2d_test_parameters.test_number

    dim_um = 1.5 * mesh_wvl_um
    thickness_box_placement_um = 1.2 * mesh_wvl_um
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(*sim_geometry.size[0:2], thickness_box_placement_um + 0.3 * mesh_wvl_um),
    )
    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    mesh_cell_override_size = mesh_wvl_um / mesh_refinement_factor
    if RUN_WITH_FD_CONVERGENCE:
        fd_step = np.linspace(2 * mesh_cell_override_size, mesh_cell_override_size, 2)
    else:
        fd_step = np.array([mesh_cell_override_size])

    polyslab = generate_polyslab(0.5 * dim_um, dim_um, rng)
    polyslab_z_value = 0

    objective = create_objective_function_2D(
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        monitor_size_wvl=monitor_size_wvl,
        box_for_override=box_for_override,
        mesh_refinement_factor=mesh_refinement_factor: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            monitor_size_wvl=monitor_size_wvl,
            box_for_override=box_for_override,
            mesh_refinement_factor=mesh_refinement_factor,
        ),
        eval_fn,
        polyslab_z_value,
        sim_path_dir=str(sim_path_dir),
    )

    _obj, adj_grad = ag.value_and_grad(objective)([polyslab])
    fd_objective_values_by_step = run_and_process_fd(
        polyslab_parameters=polyslab, fd_step=fd_step, objective=objective
    )

    return {
        "fd_objective_values_by_step": np.asarray(fd_objective_values_by_step, dtype=float),
        "adj_grad": np.squeeze(np.asarray(adj_grad, dtype=float)),
        "fd_step": np.asarray(fd_step, dtype=float),
        "polyslab_parameters": np.asarray(polyslab, dtype=float),
        "polyslab_z_value": float(polyslab_z_value),
    }


def _rf_polyslab_vertex_vectors(evaluation_data: EvaluationData) -> tuple[np.ndarray, np.ndarray]:
    """Return valid vertex FD/adjoint vectors for one RF PolySlab case."""
    if "fd_objective_values_by_step" in evaluation_data:
        fd_grad, valid_mask = select_fd_gradient_from_objective_values(
            evaluation_data["fd_objective_values_by_step"],
            evaluation_data["fd_step"],
            run_with_fd_convergence=RUN_WITH_FD_CONVERGENCE,
            fd_convergence_threshold=FD_CONVERGENCE_THRESHOLD,
        )
    else:
        fd_grad = np.asarray(evaluation_data["fd_grad"], dtype=float)
        valid_mask = np.asarray(evaluation_data["valid_mask"], dtype=bool)
    adj_grad = np.asarray(evaluation_data["adj_grad"], dtype=float)
    return fd_grad[valid_mask], adj_grad[valid_mask]


def _evaluate_rf_polyslab_evaluation_data(
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    """Evaluate saved-or-fresh RF PolySlab data into RFC-style metrics."""
    vertex_fd, vertex_adj = _rf_polyslab_vertex_vectors(evaluation_data)
    regression_metrics, observation_metrics, diagnostics = evaluate_fd_adjoint_alignment_group(
        prefix="vertex",
        fd_data=vertex_fd,
        adj_data=vertex_adj,
        min_count=MIN_FD_COMPARISON,
        overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
    )
    return (
        regression_metrics,
        observation_metrics,
        {f"vertex_{k}": v for k, v in diagnostics.items()},
    )


def _print_rf_polyslab_2d_summary(
    rf_2d_test_parameters: RFPolySlab2DTestParameters,
    diagnostics: dict[str, float],
) -> None:
    """Print the historical 2D RF PolySlab diagnostic summary."""
    print(f"\n2D PEC PolySlab Test {rf_2d_test_parameters.test_number} Summary:")
    print(f"Mesh wavelength (um): {rf_2d_test_parameters.mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {rf_2d_test_parameters.adj_wvl_um}")
    print(f"Monitor size (wavelengths): {rf_2d_test_parameters.monitor_size_wvl}")
    print(f"Mesh refinement factor: {rf_2d_test_parameters.mesh_refinement_factor}")
    print(f"Eval function: {rf_2d_test_parameters.eval_fn_name}")
    print(
        f"Vertex mean (std): {diagnostics['vertex_error_mean']} ({diagnostics['vertex_error_std']})"
    )
    print(
        "Vertex norm mean (std): "
        f"{diagnostics['vertex_error_norm_mean']} ({diagnostics['vertex_error_norm_std']})"
    )
    print(f"Vertex overlap deg: {diagnostics['vertex_overlap_deg']}")
    print("\n")


def _collect_rf_polyslab_3d_evaluation_data(
    rf_3d_test_parameters: RFPolySlab3DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect raw FD and adjoint data for one 3D RF PEC PolySlab case."""
    mesh_wvl_um = rf_3d_test_parameters.mesh_wvl_um
    adj_wvl_um = rf_3d_test_parameters.adj_wvl_um
    monitor_size_wvl = rf_3d_test_parameters.monitor_size_wvl
    polyslab_z_thickness_wvl = rf_3d_test_parameters.polyslab_z_thickness_wvl
    eval_fn = rf_3d_test_parameters.eval_fn
    test_number = rf_3d_test_parameters.test_number

    dim_um = 1.5 * mesh_wvl_um
    thickness_box_placement_um = 1.0 * mesh_wvl_um
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(*sim_geometry.size[0:2], thickness_box_placement_um),
    )
    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    mesh_cell_override_size = mesh_wvl_um / FIXED_MESH_REFINEMENT_FACTOR
    if RUN_WITH_FD_CONVERGENCE:
        fd_step = np.linspace(2 * mesh_cell_override_size, mesh_cell_override_size, 2)
    else:
        fd_step = np.array([mesh_cell_override_size])

    polyslab = generate_polyslab(0.5 * dim_um, dim_um, rng)
    polyslab_z_value = 0
    polyslab_z_thickness = polyslab_z_thickness_wvl * adj_wvl_um

    objective = create_objective_function_3D(
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        monitor_size_wvl=monitor_size_wvl,
        box_for_override=box_for_override,
        mesh_refinement_factor=FIXED_MESH_REFINEMENT_FACTOR: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            monitor_size_wvl=monitor_size_wvl,
            box_for_override=box_for_override,
            mesh_refinement_factor=mesh_refinement_factor,
        ),
        eval_fn,
        polyslab_z_value,
        polyslab_z_thickness,
        sim_path_dir=str(sim_path_dir),
    )

    _obj, adj_grad = ag.value_and_grad(objective)([polyslab])
    fd_objective_values_by_step = run_and_process_fd(
        polyslab_parameters=polyslab, fd_step=fd_step, objective=objective
    )

    return {
        "fd_objective_values_by_step": np.asarray(fd_objective_values_by_step, dtype=float),
        "adj_grad": np.squeeze(np.asarray(adj_grad, dtype=float)),
        "fd_step": np.asarray(fd_step, dtype=float),
        "polyslab_parameters": np.asarray(polyslab, dtype=float),
        "polyslab_z_value": float(polyslab_z_value),
        "polyslab_z_thickness": float(polyslab_z_thickness),
    }


def _print_rf_polyslab_3d_summary(
    rf_3d_test_parameters: RFPolySlab3DTestParameters,
    diagnostics: dict[str, float],
) -> None:
    """Print the historical 3D RF PolySlab diagnostic summary."""
    print(f"\n3D PEC PolySlab Test {rf_3d_test_parameters.test_number} Summary:")
    print(f"Mesh wavelength (um): {rf_3d_test_parameters.mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {rf_3d_test_parameters.adj_wvl_um}")
    print(f"Monitor size (wavelengths): {rf_3d_test_parameters.monitor_size_wvl}")
    print(f"Polyslab z thickness (wavelengths): {rf_3d_test_parameters.polyslab_z_thickness_wvl}")
    print(f"Eval function: {rf_3d_test_parameters.eval_fn_name}")
    print(
        f"Vertex mean (std): {diagnostics['vertex_error_mean']} ({diagnostics['vertex_error_std']})"
    )
    print(
        "Vertex norm mean (std): "
        f"{diagnostics['vertex_error_norm_mean']} ({diagnostics['vertex_error_norm_std']})"
    )
    print(f"Vertex overlap deg: {diagnostics['vertex_overlap_deg']}")
    print("\n")


@pytest.mark.numerical
@pytest.mark.parametrize(
    "rf_2d_test_parameters",
    rf_2d_test_parameters,
    ids=lambda params: case_identity_id(_case_identity_2d(params), prefix="rf-polyslab-2d"),
)
def test_finite_difference_2d_polyslab_pec(
    request: pytest.FixtureRequest,
    rf_2d_test_parameters: RFPolySlab2DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Compare 2D RF PEC PolySlab adjoint gradients to finite differences."""
    case_identity = _case_identity_2d(rf_2d_test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_rf_polyslab_2d_evaluation_data(
            rf_2d_test_parameters, rng, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_rf_polyslab_evaluation_data(
        evaluation_data
    )
    _print_rf_polyslab_2d_summary(rf_2d_test_parameters, diagnostics)
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "2D RF PEC PolySlab finite-difference comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "rf_3d_test_parameters",
    rf_3d_test_parameters,
    ids=lambda params: case_identity_id(_case_identity_3d(params), prefix="rf-polyslab-3d"),
)
def test_finite_difference_3d_polyslab_pec(
    request: pytest.FixtureRequest,
    rf_3d_test_parameters: RFPolySlab3DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Compare 3D RF PEC PolySlab adjoint gradients to finite differences."""
    case_identity = _case_identity_3d(rf_3d_test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_rf_polyslab_3d_evaluation_data(
            rf_3d_test_parameters, rng, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_rf_polyslab_evaluation_data(
        evaluation_data
    )
    _print_rf_polyslab_3d_summary(rf_3d_test_parameters, diagnostics)
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "3D RF PEC PolySlab finite-difference comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
