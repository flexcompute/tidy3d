# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

import operator

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web

td.config.logging.level = "ERROR"

PLOT_FD_ADJ_COMPARISON = False
NUM_BOXES_PER_FD_TESTS = 1
RUN_WITH_FD_CONVERGENCE = True
FD_CONVERGENCE_THRESHOLD = 0.075
MIN_FD_COMPARISON = 2
FIXED_MESH_REFINEMENT_FACTOR = 200.0
SAVE_FD_ADJ_DATA = True
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
USE_POLYSLAB_FOR_BOX = False
NUMERICAL_RESULTS_DATA_DIR = (
    "./numerical_rf_box_polyslab_test/" if USE_POLYSLAB_FOR_BOX else "./numerical_rf_box_box_test/"
)

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

    src_size = sim_size_um[0:2] + (0,)

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
        shape_x, shape_y, shape_z, *_ = field_data.Ex.values.shape

        proj_pol_rotation = field_data.Ex.values - field_data.Ey.values
        return np.sum(np.abs(proj_pol_rotation) ** 2)

    eval_fns = [intensity]
    eval_fn_names = ["intensity"]

    return eval_fns, eval_fn_names


def compare_fd_adj(fd_data, adj_data):
    norm_fd = fd_data / np.linalg.norm(fd_data)
    norm_adj = adj_data / np.linalg.norm(adj_data)

    dot_fd_adj = np.sum(norm_fd * norm_adj)

    directional_overlap_deg = np.arccos(dot_fd_adj) * 180.0 / np.pi

    def compute_error_statistics(a, b):
        relative_error = np.abs(a - b) / np.abs(b)
        return np.mean(relative_error), np.std(relative_error)

    error_mean, error_std = compute_error_statistics(fd_data, adj_data)
    error_norm_mean, error_norm_std = compute_error_statistics(norm_fd, norm_adj)

    return directional_overlap_deg, error_mean, error_std, error_norm_mean, error_norm_std


def generate_boxes(lateral_dim_bounds, mesh_cell_override_size, is_2d, rng):
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
    all_fd_grad_parameters = []
    for fd_step_idx in range(len(fd_step)):
        fd_box_parameters = []

        for param_idx in range(len(all_box_parameters)):
            copy_params_up = all_box_parameters.copy()
            copy_params_down = all_box_parameters.copy()

            copy_params_up[param_idx] += fd_step[fd_step_idx]
            copy_params_down[param_idx] -= fd_step[fd_step_idx]

            fd_box_parameters.append(copy_params_up)
            fd_box_parameters.append(copy_params_down)

        all_obj = objective(fd_box_parameters)

        fd_grad_parameters = []

        for param_idx in range(len(all_box_parameters)):
            fd_grad = (all_obj[2 * param_idx] - all_obj[2 * param_idx + 1]) / (
                2 * fd_step[fd_step_idx]
            )

            fd_grad_parameters.append(fd_grad)

        all_fd_grad_parameters.append(fd_grad_parameters)

    all_fd_grad_parameters = np.array(all_fd_grad_parameters)

    if RUN_WITH_FD_CONVERGENCE:
        assert len(fd_step) == 2, "Currently only convergence testing with 2 points"

        argmin_convergence_test = np.argmin(fd_step)
        argmax_convergence_test = np.argmax(fd_step)

        relative_diff = (
            all_fd_grad_parameters[argmax_convergence_test]
            - all_fd_grad_parameters[argmin_convergence_test]
        ) / all_fd_grad_parameters[argmin_convergence_test]

        valid_mask = np.abs(relative_diff) < FD_CONVERGENCE_THRESHOLD
        fd_grad = np.squeeze(all_fd_grad_parameters[argmin_convergence_test])
    else:
        fd_grad = np.squeeze(all_fd_grad_parameters)
        valid_mask = np.ones(fd_grad.shape, dtype=bool)

    return fd_grad, valid_mask


mm = 1e3

background_indices = [1.0]
mesh_wvls_mm = [15.0, 30.0]
adj_wvls_mm = [15.0, 30.0]

mesh_refinement_factors = np.linspace(200.0, 300.0, 4)
box_z_thickneses_3d_wvl = np.linspace(0.2, 0.4, 4)

mesh_wvls_um = [mesh_wvl_mm * mm for mesh_wvl_mm in mesh_wvls_mm]
adj_wvls_um = [adj_wvl_mm * mm for adj_wvl_mm in adj_wvls_mm]

monitor_size_3d_wvl = (1.0, 1.0, 0)

rf_2d_test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_3d_wvl)

    for mesh_refinement_factor in mesh_refinement_factors:
        for eval_fn_idx, eval_fn in enumerate(eval_fns):
            rf_2d_test_parameters.append(
                {
                    "mesh_wvl_um": mesh_wvl_um,
                    "adj_wvl_um": adj_wvl_um,
                    "monitor_size_wvl": monitor_size_3d_wvl,
                    "mesh_refinement_factor": mesh_refinement_factor,
                    "eval_fn": eval_fn,
                    "eval_fn_name": eval_fn_names[eval_fn_idx],
                    "test_number": test_number,
                }
            )

            test_number += 1


rf_3d_test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_3d_wvl)

    for box_z_thickness_wvl in box_z_thickneses_3d_wvl:
        for eval_fn_idx, eval_fn in enumerate(eval_fns):
            rf_3d_test_parameters.append(
                {
                    "mesh_wvl_um": mesh_wvl_um,
                    "adj_wvl_um": adj_wvl_um,
                    "monitor_size_wvl": monitor_size_3d_wvl,
                    "box_z_thickness_wvl": box_z_thickness_wvl,
                    "eval_fn": eval_fn,
                    "eval_fn_name": eval_fn_names[eval_fn_idx],
                    "test_number": test_number,
                }
            )

            test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize(
    "rf_2d_test_parameters, dir_name",
    zip(
        rf_2d_test_parameters,
        ([NUMERICAL_RESULTS_DATA_DIR] if SAVE_FD_ADJ_DATA else [None]) * len(rf_2d_test_parameters),
    ),
    indirect=["dir_name"],
)
def test_finite_difference_2d_box_pec(
    rf_2d_test_parameters, rng, tmp_path, create_directory, redirect_stdout_to_stderr
):
    """Test a variety of autograd permittivity gradients for 2D PEC boxes by"""
    """comparing them to numerical finite difference."""

    test_number = rf_2d_test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_size_wvl,
        mesh_refinement_factor,
        eval_fn,
        eval_fn_name,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_size_wvl",
        "mesh_refinement_factor",
        "eval_fn",
        "eval_fn_name",
        "test_number",
    )(rf_2d_test_parameters)

    dim_um = 3 * mesh_wvl_um

    thickness_box_placement_um = 0.6 * mesh_wvl_um  # 1.2 * mesh_wvl_um

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0),
        size=sim_geometry.size[0:2] + (thickness_box_placement_um + 0.3 * mesh_wvl_um,),
    )

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_wvl)

    sim_path_dir = tmp_path / f"test{test_number}"
    sim_path_dir.mkdir()

    mesh_cell_override_size = mesh_wvl_um / mesh_refinement_factor

    if RUN_WITH_FD_CONVERGENCE:
        fd_step = np.linspace(2 * mesh_cell_override_size, mesh_cell_override_size, 2)
    else:
        fd_step = np.array([mesh_cell_override_size])

    z_planes = np.array([0.0])

    all_box_parameters, box_z_values = generate_boxes(
        lateral_dim_bounds=[-0.5 * dim_um, 0.5 * dim_um],
        mesh_cell_override_size=mesh_cell_override_size,
        is_2d=True,
        rng=rng,
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

    obj_val_and_grad = ag.value_and_grad(objective)

    obj, adj_grad = obj_val_and_grad([all_box_parameters])

    fd_grad, valid_mask = run_and_process_fd(
        all_box_parameters=all_box_parameters, fd_step=fd_step, objective=objective
    )

    fd_grad = np.array(fd_grad)
    adj_grad = np.array(adj_grad)
    valid_mask = np.array(valid_mask)

    reshape_fd = np.reshape(fd_grad, (NUM_BOXES_PER_FD_TESTS, 2))
    reshape_adj = np.reshape(adj_grad, (NUM_BOXES_PER_FD_TESTS, 2))
    reshape_valid = np.reshape(valid_mask, (NUM_BOXES_PER_FD_TESTS, 2))

    all_center_fd = []
    all_center_adj = []

    all_width_fd = []
    all_width_adj = []

    for box_idx in range(NUM_BOXES_PER_FD_TESTS):
        width_fd = reshape_fd[box_idx]
        width_adj = reshape_adj[box_idx]

        width_valid = reshape_valid[box_idx]

        add_widths_fd = [width_fd[idx] for idx in range(len(width_fd)) if width_valid[idx]]
        add_widths_adj = [width_adj[idx] for idx in range(len(width_adj)) if width_valid[idx]]

        all_width_fd += add_widths_fd
        all_width_adj += add_widths_adj

    all_width_fd = np.array(all_width_fd)
    all_width_adj = np.array(all_width_adj)

    assert len(all_width_fd) >= MIN_FD_COMPARISON, "Too many widths were trimmed"

    (
        width_overlap_deg,
        width_error_mean,
        width_error_std,
        width_error_norm_mean,
        width_error_norm_std,
    ) = compare_fd_adj(all_width_fd, all_width_adj)

    width_data = np.zeros((2, len(all_width_fd)))

    width_data[SAVE_FD_LOC, :] = all_width_fd
    width_data[SAVE_ADJ_LOC, :] = all_width_adj

    print(f"\n2D PEC Box Test {test_number} Summary:")
    print(f"Mesh wavelength (um): {mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {adj_wvl_um}")
    print(f"Monitor size (wavelengths): {monitor_size_wvl}")
    print(f"Mesh refinement factor: {mesh_refinement_factor}")
    print(f"Eval function: {eval_fn_name}")
    print(f"Width mean (std): {width_error_mean} ({width_error_std})")
    print(f"Width norm mean (std): {width_error_norm_mean} ({width_error_norm_std})")
    print(f"Width overlap deg: {width_overlap_deg}")
    print("\n")

    if SAVE_FD_ADJ_DATA:
        np.save(
            f"{NUMERICAL_RESULTS_DATA_DIR}/rf_box_2d_width_fd_adj_data_test_{test_number}.npy",
            width_data,
        )

        yvals = [width_error_mean, width_error_norm_mean]
        yerrs = [width_error_std, width_error_norm_std]
        bars = plt.bar(
            ["width\nerror", "width\nerror\nnorm"],
            yvals,
            yerr=yerrs,
            color="skyblue",
            edgecolor="black",
            capsize=5,
        )

        plt.title(f"Vector Alignment (deg):\n{width_overlap_deg:.2f}")
        plt.xticks(rotation=45)

        for idx, bar in enumerate(bars):
            yval = bar.get_height()
            yerr = yerrs[idx]
            plt.text(
                bar.get_x() + 0.5 * bar.get_width(),
                yval + yerr + 0.1,
                round(yval, 3),
                ha="center",
                va="bottom",
            )

        plt.savefig(f"{NUMERICAL_RESULTS_DATA_DIR}/rf_2d_box_summary_plot_test_{test_number}.png")


@pytest.mark.numerical
@pytest.mark.parametrize(
    "rf_3d_test_parameters, dir_name",
    zip(
        rf_3d_test_parameters,
        ([NUMERICAL_RESULTS_DATA_DIR] if SAVE_FD_ADJ_DATA else [None]) * len(rf_3d_test_parameters),
    ),
    indirect=["dir_name"],
)
def test_finite_difference_3d_box_pec(
    rf_3d_test_parameters, rng, tmp_path, create_directory, redirect_stdout_to_stderr
):
    """Test a variety of autograd permittivity gradients for 3D PEC boxes by"""
    """comparing them to numerical finite difference."""

    test_number = rf_3d_test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_size_wvl,
        box_z_thickness_wvl,
        eval_fn,
        eval_fn_name,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_size_wvl",
        "box_z_thickness_wvl",
        "eval_fn",
        "eval_fn_name",
        "test_number",
    )(rf_3d_test_parameters)

    dim_um = 3 * mesh_wvl_um

    thickness_box_placement_um = 1.2 * mesh_wvl_um

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0),
        size=sim_geometry.size[0:2] + (thickness_box_placement_um + 0.3 * mesh_wvl_um,),
    )

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_wvl)

    sim_path_dir = tmp_path / f"test{test_number}"
    sim_path_dir.mkdir()

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

    obj_val_and_grad = ag.value_and_grad(objective)

    obj, adj_grad = obj_val_and_grad([all_box_parameters])

    fd_grad, valid_mask = run_and_process_fd(
        all_box_parameters=all_box_parameters, fd_step=fd_step, objective=objective
    )

    fd_grad = np.array(fd_grad)
    adj_grad = np.array(adj_grad)
    valid_mask = np.array(valid_mask)

    reshape_fd = np.reshape(fd_grad, (NUM_BOXES_PER_FD_TESTS, 3))
    reshape_adj = np.reshape(adj_grad, (NUM_BOXES_PER_FD_TESTS, 3))
    reshape_valid = np.reshape(valid_mask, (NUM_BOXES_PER_FD_TESTS, 3))

    all_center_fd = []
    all_center_adj = []

    all_width_fd = []
    all_width_adj = []

    all_z_coord_fd = []
    all_z_coord_adj = []

    for box_idx in range(NUM_BOXES_PER_FD_TESTS):
        width_fd = reshape_fd[box_idx, 0:2]
        width_adj = reshape_adj[box_idx, 0:2]

        width_valid = reshape_valid[box_idx, 0:2]

        add_widths_fd = [width_fd[idx] for idx in range(len(width_fd)) if width_valid[idx]]
        add_widths_adj = [width_adj[idx] for idx in range(len(width_adj)) if width_valid[idx]]

        all_width_fd += add_widths_fd
        all_width_adj += add_widths_adj

        if reshape_valid[box_idx, 2]:
            all_z_coord_fd.append(reshape_fd[box_idx, 2])
            all_z_coord_adj.append(reshape_adj[box_idx, 2])

    all_width_fd = np.array(all_width_fd)
    all_width_adj = np.array(all_width_adj)

    all_z_coord_fd = np.array(all_z_coord_fd)
    all_z_coord_adj = np.array(all_z_coord_adj)

    assert len(all_width_fd) >= MIN_FD_COMPARISON, "Too many widths were trimmed"
    assert len(all_z_coord_fd) >= 1, "Too many z coordinates were trimmed"

    (
        width_overlap_deg,
        width_error_mean,
        width_error_std,
        width_error_norm_mean,
        width_error_norm_std,
    ) = compare_fd_adj(all_width_fd, all_width_adj)
    (
        z_coord_overlap_deg,
        z_coord_error_mean,
        z_coord_error_std,
        z_coord_error_norm_mean,
        z_coord_error_norm_std,
    ) = compare_fd_adj(all_z_coord_fd, all_z_coord_adj)

    width_data = np.zeros((2, len(all_width_fd)))
    z_coord_data = np.zeros((2, len(all_z_coord_fd)))

    width_data[SAVE_FD_LOC, :] = all_width_fd
    width_data[SAVE_ADJ_LOC, :] = all_width_adj

    z_coord_data[SAVE_FD_LOC, :] = all_z_coord_fd
    z_coord_data[SAVE_ADJ_LOC, :] = all_z_coord_adj

    print(f"\n3D PEC Box Test {test_number} Summary:")
    print(f"Mesh wavelength (um): {mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {adj_wvl_um}")
    print(f"Monitor size (wavelengths): {monitor_size_wvl}")
    print(f"Box z thickness (wavelengths): {box_z_thickness_wvl}")
    print(f"Eval function: {eval_fn_name}")
    print(f"Width mean (std): {width_error_mean} ({width_error_std})")
    print(f"Width norm mean (std): {width_error_norm_mean} ({width_error_norm_std})")
    print(f"Width overlap deg: {width_overlap_deg}")
    print(f"Z mean (std): {z_coord_error_mean} ({z_coord_error_std})")
    print(f"Z norm mean (std): {z_coord_error_norm_mean} ({z_coord_error_norm_std})")
    print(f"Z overlap deg: {z_coord_overlap_deg}")
    print("\n")

    if SAVE_FD_ADJ_DATA:
        np.save(
            f"{NUMERICAL_RESULTS_DATA_DIR}/rf_box_3d_width_fd_adj_data_test_{test_number}.npy",
            width_data,
        )
        np.save(
            f"{NUMERICAL_RESULTS_DATA_DIR}/rf_box_3d_z_coord_fd_adj_data_test_{test_number}.npy",
            z_coord_data,
        )

        yvals = [
            width_error_mean,
            width_error_norm_mean,
            z_coord_error_mean,
            z_coord_error_norm_mean,
        ]
        yerrs = [
            width_error_std,
            width_error_norm_std,
            z_coord_error_std,
            z_coord_error_norm_std,
        ]
        bars = plt.bar(
            [
                "width\nerror",
                "width\nerror\nnorm",
                "z\ncoord\nerror",
                "z\ncoord\nerror\nnorm",
            ],
            yvals,
            yerr=yerrs,
            color="skyblue",
            edgecolor="black",
            capsize=5,
        )

        plt.title(f"Vector Alignment (deg):\n{width_overlap_deg:.2f}, {z_coord_overlap_deg:.2f}")
        plt.ylabel("Relative Error")

        for idx, bar in enumerate(bars):
            yval = bar.get_height()
            yerr = yerrs[idx]
            plt.text(
                bar.get_x() + 0.5 * bar.get_width(),
                yval + yerr + 0.1,
                round(yval, 3),
                ha="center",
                va="bottom",
            )

        plt.savefig(f"{NUMERICAL_RESULTS_DATA_DIR}/rf_3d_box_summary_plot_test_{test_number}.png")
