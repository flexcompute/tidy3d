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
NUM_BOXES_PER_FD_TESTS = 1
RUN_WITH_FD_CONVERGENCE = True
FD_CONVERGENCE_THRESHOLD = 0.075
MIN_FD_COMPARISON = 2
ALIGNMENT_OVERLAP_THRESHOLD_DEG = 45.0
FIXED_MESH_REFINEMENT_FACTOR = 200.0
LOCAL_GRADIENT = True
VERBOSE = False
USE_POLYSLAB_FOR_BOX = False

assert NUM_BOXES_PER_FD_TESTS == 1, "Currently only supporting a single box in the numerical test!"

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


def get_sim_geometry(mesh_wvl_um):
    return td.Box(size=(4 * mesh_wvl_um, 4 * mesh_wvl_um, 7 * mesh_wvl_um), center=(0, 0, 0))


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
        center=(0, 0, 0.125 * sim_size_um[2]),
        # center=(0, 0, -0.75 * mesh_wvl_um),
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


def create_objective_function_2D(create_sim_base, eval_fn, box_z_values, sim_path_dir):
    # 2D PEC objective will take in an array of parameters that can be reshaped into
    # a Nx4 array where N is the number of boxes and the four parameters are
    # center (x,y) and lateral size (x,y) assuming the thickness in z is 0
    #
    # box_param_arrays should be a list of such parameters for cases where we
    # are running finite difference simulations and want to put together a
    # whole batch
    def objective(box_param_arrays):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(box_param_arrays)):
            get_boxes_params = box_param_arrays[idx]
            num_boxes = len(get_boxes_params) // 2
            reshape_parameters = np.reshape(get_boxes_params, (num_boxes, 2))

            box_structures = []

            for box_idx in range(num_boxes):
                box_params = reshape_parameters[box_idx]

                if USE_POLYSLAB_FOR_BOX:
                    min_x = -0.5 * box_params[0]
                    min_y = -0.5 * box_params[1]

                    max_x = 0.5 * box_params[0]
                    max_y = 0.5 * box_params[1]

                    vertices = ((min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y))

                    polyslab = td.PolySlab(
                        vertices=vertices,
                        slab_bounds=(box_z_values[box_idx], box_z_values[box_idx]),
                    )
                    box_structures.append(td.Structure(geometry=polyslab, medium=td.PEC2D))
                else:
                    box_structures.append(
                        td.Structure(
                            geometry=td.Box(
                                center=(0.0, 0.0, box_z_values[box_idx]),
                                size=(box_params[0], box_params[1], 0),
                            ),
                            medium=td.PEC2D,
                        )
                    )

            sim_with_block = sim_base.updated_copy(
                structures=tuple(list(sim_base.structures) + box_structures)
            )

            simulation_dict[f"numerical_rf_box_2d_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(box_param_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_rf_box_2d_testing_{idx}"]))

        if len(box_param_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def create_objective_function_3D(create_sim_base, eval_fn, box_z_thickneses, sim_path_dir):
    # 3D PEC objective will take in an array of parameters that can be reshaped into
    # a Nx5 array where N is the number of boxes and the four parameters are
    # center (x,y) and lateral size (x,y) and the z position of the box. The box thicknesses
    # are set in box_z_thicknesses
    #
    # box_param_arrays should be a list of such parameters for cases where we
    # are running finite difference simulations and want to put together a
    # whole batch
    def objective(box_param_arrays):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(box_param_arrays)):
            get_boxes_params = box_param_arrays[idx]
            num_boxes = len(get_boxes_params) // 3
            reshape_parameters = np.reshape(get_boxes_params, (num_boxes, 3))

            box_structures = []

            for box_idx in range(num_boxes):
                box_params = reshape_parameters[box_idx]

                if USE_POLYSLAB_FOR_BOX:
                    min_x = -0.5 * box_params[0]
                    min_y = -0.5 * box_params[1]

                    max_x = 0.5 * box_params[0]
                    max_y = 0.5 * box_params[1]

                    vertices = ((min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y))
                    polyslab = td.PolySlab(
                        vertices=vertices,
                        slab_bounds=(
                            box_params[2] - 0.5 * box_z_thickneses[box_idx],
                            box_params[2] + 0.5 * box_z_thickneses[box_idx],
                        ),
                    )
                    box_structures.append(td.Structure(geometry=polyslab, medium=td.PECMedium()))
                else:
                    box_structures.append(
                        td.Structure(
                            geometry=td.Box(
                                center=(0.0, 0.0, box_params[2]),
                                size=(box_params[0], box_params[1], box_z_thickneses[box_idx]),
                            ),
                            medium=td.PECMedium(),
                        )
                    )

            sim_with_block = sim_base.updated_copy(
                structures=tuple(list(sim_base.structures) + box_structures)
            )

            simulation_dict[f"numerical_rf_box_3d_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(box_param_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_rf_box_3d_testing_{idx}"]))

        if len(box_param_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def make_eval_fns(monitor_size_wvl):
    num_nonzero_spatial_dims = 3 - np.sum(np.isclose(monitor_size_wvl, 0))

    def intensity(sim_data):
        field_data = sim_data["monitor_fields"]
        _shape_x, _shape_y, _shape_z, *_ = field_data.Ex.values.shape

        proj_pol_rotation = field_data.Ex.values - field_data.Ey.values
        return np.sum(np.abs(proj_pol_rotation) ** 2)

    eval_fns = [intensity]
    eval_fn_names = ["intensity"]

    return eval_fns, eval_fn_names


def generate_boxes(lateral_dim_bounds, mesh_cell_override_size, is_2d, rng, mesh_wvl_um):
    all_box_parameters = []
    box_z_values = []

    for _box in range(NUM_BOXES_PER_FD_TESTS):
        x_size = rng.uniform(0.6 * mesh_wvl_um, 0.7 * mesh_wvl_um)
        y_size = rng.uniform(0.2 * mesh_wvl_um, 0.3 * mesh_wvl_um)

        if is_2d:
            all_box_parameters += [x_size, y_size]
            box_z_values.append(0.0)
        else:
            all_box_parameters += [
                x_size,
                y_size,
                0.0,
            ]

    if is_2d:
        return all_box_parameters, box_z_values
    else:
        return all_box_parameters


def run_and_process_fd(all_box_parameters, fd_step, objective):
    objective_values_by_step = []
    for fd_step_idx in range(len(fd_step)):
        fd_box_parameters = []

        for param_idx in range(len(all_box_parameters)):
            copy_params_up = all_box_parameters.copy()
            copy_params_down = all_box_parameters.copy()

            copy_params_up[param_idx] += fd_step[fd_step_idx]
            copy_params_down[param_idx] -= fd_step[fd_step_idx]

            fd_box_parameters.append(copy_params_up)
            fd_box_parameters.append(copy_params_down)

        objective_values_by_step.append(np.asarray(objective(fd_box_parameters), dtype=float))

    return np.asarray(objective_values_by_step, dtype=float)


mm = 1e3

background_indices = [1.0]
mesh_wvls_mm = [15.0, 30.0]
adj_wvls_mm = [15.0, 30.0]

mesh_refinement_factors = np.linspace(200.0, 300.0, 4)
box_z_thickneses_3d_wvl = np.linspace(0.2, 0.4, 4)

mesh_wvls_um = [mesh_wvl_mm * mm for mesh_wvl_mm in mesh_wvls_mm]
adj_wvls_um = [adj_wvl_mm * mm for adj_wvl_mm in adj_wvls_mm]

monitor_size_3d_wvl = (1.0, 1.0, 0)


class RFBox2DCaseIdentity(BaseModel):
    """Semantic identity for one 2D RF PEC box finite-difference case."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_size_wvl: Size
    num_boxes_per_fd_tests: int
    mesh_refinement_factor: float
    eval_fn_name: str
    run_with_fd_convergence: bool
    fd_convergence_threshold: float
    min_fd_comparison: int
    alignment_overlap_threshold_deg: float
    use_polyslab_for_box: bool


class RFBox2DTestParameters(RFBox2DCaseIdentity):
    """Full parameter bundle for one 2D RF box test invocation."""

    eval_fn: Callable
    test_number: int


class RFBox3DCaseIdentity(BaseModel):
    """Semantic identity for one 3D RF PEC box finite-difference case."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_size_wvl: Size
    num_boxes_per_fd_tests: int
    box_z_thickness_wvl: float
    fixed_mesh_refinement_factor: float
    eval_fn_name: str
    run_with_fd_convergence: bool
    fd_convergence_threshold: float
    min_fd_comparison: int
    alignment_overlap_threshold_deg: float
    use_polyslab_for_box: bool


class RFBox3DTestParameters(RFBox3DCaseIdentity):
    """Full parameter bundle for one 3D RF box test invocation."""

    eval_fn: Callable
    test_number: int


rf_2d_test_parameters: list[RFBox2DTestParameters] = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_3d_wvl)

    for mesh_refinement_factor in mesh_refinement_factors:
        for eval_fn_idx, eval_fn in enumerate(eval_fns):
            rf_2d_test_parameters.append(
                RFBox2DTestParameters(
                    mesh_wvl_um=mesh_wvl_um,
                    adj_wvl_um=adj_wvl_um,
                    monitor_size_wvl=monitor_size_3d_wvl,
                    num_boxes_per_fd_tests=NUM_BOXES_PER_FD_TESTS,
                    mesh_refinement_factor=mesh_refinement_factor,
                    eval_fn=eval_fn,
                    eval_fn_name=eval_fn_names[eval_fn_idx],
                    run_with_fd_convergence=RUN_WITH_FD_CONVERGENCE,
                    fd_convergence_threshold=FD_CONVERGENCE_THRESHOLD,
                    min_fd_comparison=MIN_FD_COMPARISON,
                    alignment_overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
                    use_polyslab_for_box=USE_POLYSLAB_FOR_BOX,
                    test_number=test_number,
                )
            )

            test_number += 1


rf_3d_test_parameters: list[RFBox3DTestParameters] = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_3d_wvl)

    for box_z_thickness_wvl in box_z_thickneses_3d_wvl:
        for eval_fn_idx, eval_fn in enumerate(eval_fns):
            rf_3d_test_parameters.append(
                RFBox3DTestParameters(
                    mesh_wvl_um=mesh_wvl_um,
                    adj_wvl_um=adj_wvl_um,
                    monitor_size_wvl=monitor_size_3d_wvl,
                    num_boxes_per_fd_tests=NUM_BOXES_PER_FD_TESTS,
                    box_z_thickness_wvl=box_z_thickness_wvl,
                    fixed_mesh_refinement_factor=FIXED_MESH_REFINEMENT_FACTOR,
                    eval_fn=eval_fn,
                    eval_fn_name=eval_fn_names[eval_fn_idx],
                    run_with_fd_convergence=RUN_WITH_FD_CONVERGENCE,
                    fd_convergence_threshold=FD_CONVERGENCE_THRESHOLD,
                    min_fd_comparison=MIN_FD_COMPARISON,
                    alignment_overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
                    use_polyslab_for_box=USE_POLYSLAB_FOR_BOX,
                    test_number=test_number,
                )
            )

            test_number += 1


def _case_identity_2d(parameters: RFBox2DTestParameters) -> RFBox2DCaseIdentity:
    """Build the semantic case identity for a 2D RF box case."""
    return case_identity_from_parameters(RFBox2DCaseIdentity, parameters)


def _case_identity_3d(parameters: RFBox3DTestParameters) -> RFBox3DCaseIdentity:
    """Build the semantic case identity for a 3D RF box case."""
    return case_identity_from_parameters(RFBox3DCaseIdentity, parameters)


def _collect_rf_box_2d_evaluation_data(
    rf_2d_test_parameters: RFBox2DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect raw FD and adjoint data for one 2D RF PEC box case."""
    mesh_wvl_um = rf_2d_test_parameters.mesh_wvl_um
    adj_wvl_um = rf_2d_test_parameters.adj_wvl_um
    monitor_size_wvl = rf_2d_test_parameters.monitor_size_wvl
    mesh_refinement_factor = rf_2d_test_parameters.mesh_refinement_factor
    eval_fn = rf_2d_test_parameters.eval_fn
    test_number = rf_2d_test_parameters.test_number

    dim_um = 3 * mesh_wvl_um
    thickness_box_placement_um = 0.6 * mesh_wvl_um
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

    all_box_parameters, box_z_values = generate_boxes(
        lateral_dim_bounds=[-0.5 * dim_um, 0.5 * dim_um],
        mesh_cell_override_size=mesh_cell_override_size,
        is_2d=True,
        rng=rng,
        mesh_wvl_um=mesh_wvl_um,
    )

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
        box_z_values,
        sim_path_dir=str(sim_path_dir),
    )

    _obj, adj_grad = ag.value_and_grad(objective)([all_box_parameters])
    fd_objective_values_by_step = run_and_process_fd(
        all_box_parameters=all_box_parameters, fd_step=fd_step, objective=objective
    )

    return {
        "fd_objective_values_by_step": np.asarray(fd_objective_values_by_step, dtype=float),
        "adj_grad": np.asarray(adj_grad, dtype=float),
        "fd_step": np.asarray(fd_step, dtype=float),
        "box_parameters": np.asarray(all_box_parameters, dtype=float),
        "box_z_values": np.asarray(box_z_values, dtype=float),
    }


def _rf_box_2d_width_vectors(evaluation_data: EvaluationData) -> tuple[np.ndarray, np.ndarray]:
    """Return valid width FD/adjoint vectors for a 2D RF box case."""
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
    reshape_fd = np.reshape(fd_grad, (NUM_BOXES_PER_FD_TESTS, 2))
    reshape_adj = np.reshape(adj_grad, (NUM_BOXES_PER_FD_TESTS, 2))
    reshape_valid = np.reshape(valid_mask, (NUM_BOXES_PER_FD_TESTS, 2))
    return reshape_fd[reshape_valid], reshape_adj[reshape_valid]


def _evaluate_rf_box_2d_evaluation_data(
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    """Evaluate saved-or-fresh 2D RF box data into RFC-style metrics."""
    width_fd, width_adj = _rf_box_2d_width_vectors(evaluation_data)
    regression_metrics, observation_metrics, diagnostics = evaluate_fd_adjoint_alignment_group(
        prefix="width",
        fd_data=width_fd,
        adj_data=width_adj,
        min_count=MIN_FD_COMPARISON,
        overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
    )
    return (
        regression_metrics,
        observation_metrics,
        {f"width_{k}": v for k, v in diagnostics.items()},
    )


def _print_rf_box_2d_summary(
    rf_2d_test_parameters: RFBox2DTestParameters,
    diagnostics: dict[str, float],
) -> None:
    """Print the historical 2D RF box diagnostic summary."""
    print(f"\n2D PEC Box Test {rf_2d_test_parameters.test_number} Summary:")
    print(f"Mesh wavelength (um): {rf_2d_test_parameters.mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {rf_2d_test_parameters.adj_wvl_um}")
    print(f"Monitor size (wavelengths): {rf_2d_test_parameters.monitor_size_wvl}")
    print(f"Mesh refinement factor: {rf_2d_test_parameters.mesh_refinement_factor}")
    print(f"Eval function: {rf_2d_test_parameters.eval_fn_name}")
    print(f"Width mean (std): {diagnostics['width_error_mean']} ({diagnostics['width_error_std']})")
    print(
        "Width norm mean (std): "
        f"{diagnostics['width_error_norm_mean']} ({diagnostics['width_error_norm_std']})"
    )
    print(f"Width overlap deg: {diagnostics['width_overlap_deg']}")
    print("\n")


def _collect_rf_box_3d_evaluation_data(
    rf_3d_test_parameters: RFBox3DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect raw FD and adjoint data for one 3D RF PEC box case."""
    mesh_wvl_um = rf_3d_test_parameters.mesh_wvl_um
    adj_wvl_um = rf_3d_test_parameters.adj_wvl_um
    monitor_size_wvl = rf_3d_test_parameters.monitor_size_wvl
    box_z_thickness_wvl = rf_3d_test_parameters.box_z_thickness_wvl
    eval_fn = rf_3d_test_parameters.eval_fn
    test_number = rf_3d_test_parameters.test_number

    dim_um = 3 * mesh_wvl_um
    thickness_box_placement_um = 1.2 * mesh_wvl_um
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(*sim_geometry.size[0:2], thickness_box_placement_um + 0.3 * mesh_wvl_um),
    )
    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    mesh_cell_override_size = mesh_wvl_um / FIXED_MESH_REFINEMENT_FACTOR
    if RUN_WITH_FD_CONVERGENCE:
        fd_step = np.linspace(2 * mesh_cell_override_size, mesh_cell_override_size, 2)
    else:
        fd_step = np.array([mesh_cell_override_size])

    all_box_parameters = generate_boxes(
        lateral_dim_bounds=[-0.5 * dim_um, 0.5 * dim_um],
        mesh_cell_override_size=mesh_cell_override_size,
        is_2d=False,
        rng=rng,
        mesh_wvl_um=mesh_wvl_um,
    )
    box_z_thicknesses = box_z_thickness_wvl * adj_wvl_um * np.ones(NUM_BOXES_PER_FD_TESTS)

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
        box_z_thicknesses,
        sim_path_dir=str(sim_path_dir),
    )

    _obj, adj_grad = ag.value_and_grad(objective)([all_box_parameters])
    fd_objective_values_by_step = run_and_process_fd(
        all_box_parameters=all_box_parameters, fd_step=fd_step, objective=objective
    )

    return {
        "fd_objective_values_by_step": np.asarray(fd_objective_values_by_step, dtype=float),
        "adj_grad": np.asarray(adj_grad, dtype=float),
        "fd_step": np.asarray(fd_step, dtype=float),
        "box_parameters": np.asarray(all_box_parameters, dtype=float),
        "box_z_thicknesses": np.asarray(box_z_thicknesses, dtype=float),
    }


def _rf_box_3d_vectors(
    evaluation_data: EvaluationData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return valid width and z-coordinate FD/adjoint vectors for a 3D RF box case."""
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
    reshape_fd = np.reshape(fd_grad, (NUM_BOXES_PER_FD_TESTS, 3))
    reshape_adj = np.reshape(adj_grad, (NUM_BOXES_PER_FD_TESTS, 3))
    reshape_valid = np.reshape(valid_mask, (NUM_BOXES_PER_FD_TESTS, 3))
    width_valid = reshape_valid[:, 0:2]
    z_valid = reshape_valid[:, 2]
    return (
        reshape_fd[:, 0:2][width_valid],
        reshape_adj[:, 0:2][width_valid],
        reshape_fd[:, 2][z_valid],
        reshape_adj[:, 2][z_valid],
    )


def _evaluate_rf_box_3d_evaluation_data(
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    """Evaluate saved-or-fresh 3D RF box data into RFC-style metrics."""
    width_fd, width_adj, z_coord_fd, z_coord_adj = _rf_box_3d_vectors(evaluation_data)
    width_regression, width_observation, width_diagnostics = evaluate_fd_adjoint_alignment_group(
        prefix="width",
        fd_data=width_fd,
        adj_data=width_adj,
        min_count=MIN_FD_COMPARISON,
        overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
    )
    z_regression, z_observation, z_diagnostics = evaluate_fd_adjoint_alignment_group(
        prefix="z_coord",
        fd_data=z_coord_fd,
        adj_data=z_coord_adj,
        min_count=1,
        overlap_threshold_deg=ALIGNMENT_OVERLAP_THRESHOLD_DEG,
    )
    diagnostics = {f"width_{k}": v for k, v in width_diagnostics.items()}
    diagnostics.update({f"z_coord_{k}": v for k, v in z_diagnostics.items()})
    return width_regression + z_regression, width_observation + z_observation, diagnostics


def _print_rf_box_3d_summary(
    rf_3d_test_parameters: RFBox3DTestParameters,
    diagnostics: dict[str, float],
) -> None:
    """Print the historical 3D RF box diagnostic summary."""
    print(f"\n3D PEC Box Test {rf_3d_test_parameters.test_number} Summary:")
    print(f"Mesh wavelength (um): {rf_3d_test_parameters.mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {rf_3d_test_parameters.adj_wvl_um}")
    print(f"Monitor size (wavelengths): {rf_3d_test_parameters.monitor_size_wvl}")
    print(f"Box z thickness (wavelengths): {rf_3d_test_parameters.box_z_thickness_wvl}")
    print(f"Eval function: {rf_3d_test_parameters.eval_fn_name}")
    print(f"Width mean (std): {diagnostics['width_error_mean']} ({diagnostics['width_error_std']})")
    print(
        "Width norm mean (std): "
        f"{diagnostics['width_error_norm_mean']} ({diagnostics['width_error_norm_std']})"
    )
    print(f"Width overlap deg: {diagnostics['width_overlap_deg']}")
    print(f"Z mean (std): {diagnostics['z_coord_error_mean']} ({diagnostics['z_coord_error_std']})")
    print(
        "Z norm mean (std): "
        f"{diagnostics['z_coord_error_norm_mean']} ({diagnostics['z_coord_error_norm_std']})"
    )
    print(f"Z overlap deg: {diagnostics['z_coord_overlap_deg']}")
    print("\n")


@pytest.mark.numerical
@pytest.mark.parametrize(
    "rf_2d_test_parameters",
    rf_2d_test_parameters,
    ids=lambda params: case_identity_id(_case_identity_2d(params), prefix="rf-box-2d"),
)
def test_finite_difference_2d_box_pec(
    request: pytest.FixtureRequest,
    rf_2d_test_parameters: RFBox2DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Compare 2D RF PEC box adjoint gradients to finite differences."""
    case_identity = _case_identity_2d(rf_2d_test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_rf_box_2d_evaluation_data(
            rf_2d_test_parameters, rng, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_rf_box_2d_evaluation_data(
        evaluation_data
    )
    _print_rf_box_2d_summary(rf_2d_test_parameters, diagnostics)
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "2D RF PEC box finite-difference comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "rf_3d_test_parameters",
    rf_3d_test_parameters,
    ids=lambda params: case_identity_id(_case_identity_3d(params), prefix="rf-box-3d"),
)
def test_finite_difference_3d_box_pec(
    request: pytest.FixtureRequest,
    rf_3d_test_parameters: RFBox3DTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Compare 3D RF PEC box adjoint gradients to finite differences."""
    case_identity = _case_identity_3d(rf_3d_test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_rf_box_3d_evaluation_data(
            rf_3d_test_parameters, rng, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_rf_box_3d_evaluation_data(
        evaluation_data
    )
    _print_rf_box_3d_summary(rf_3d_test_parameters, diagnostics)
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "3D RF PEC box finite-difference comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
