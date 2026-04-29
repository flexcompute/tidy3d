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
NUM_VERTICES_PER_FD_TESTS = 10
RUN_WITH_FD_CONVERGENCE = True
FD_CONVERGENCE_THRESHOLD = 0.05
MIN_FD_COMPARISON = 2
FIXED_MESH_REFINEMENT_FACTOR = 200.0
SAVE_FD_ADJ_DATA = True
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
NUMERICAL_RESULTS_DATA_DIR = "./numerical_rf_polyslab_test/"

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
    all_fd_grad_parameters = []
    for fd_step_idx in range(len(fd_step)):
        fd_polyslab_parameters = []

        for param_idx in range(len(polyslab_parameters)):
            copy_params_up = polyslab_parameters.copy()
            copy_params_down = polyslab_parameters.copy()

            copy_params_up[param_idx] += fd_step[fd_step_idx]
            copy_params_down[param_idx] -= fd_step[fd_step_idx]

            fd_polyslab_parameters.append(copy_params_up)
            fd_polyslab_parameters.append(copy_params_down)

        all_obj = objective(fd_polyslab_parameters)

        fd_grad_parameters = []

        for param_idx in range(len(polyslab_parameters)):
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

background_indices = [1.0, 1.5]
mesh_wvls_mm = [15.0, 15.0]
adj_wvls_mm = [15.0, 20.0]

mesh_refinement_factors = np.linspace(200.0, 300.0, 4)

polyslab_z_thickneses_3d_wvl = np.linspace(0.2, 0.4, 4)

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

    for polyslab_z_thickness_wvl in polyslab_z_thickneses_3d_wvl:
        for eval_fn_idx, eval_fn in enumerate(eval_fns):
            rf_3d_test_parameters.append(
                {
                    "mesh_wvl_um": mesh_wvl_um,
                    "adj_wvl_um": adj_wvl_um,
                    "monitor_size_wvl": monitor_size_3d_wvl,
                    "polyslab_z_thickness_wvl": polyslab_z_thickness_wvl,
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
def test_finite_difference_2d_polyslab_pec(
    rf_2d_test_parameters, rng, tmp_path, create_directory, redirect_stdout_to_stderr
):
    """Test a variety of autograd permittivity gradients for 2D `PolySlab` PEC by"""
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

    dim_um = 1.5 * mesh_wvl_um

    thickness_box_placement_um = 1.2 * mesh_wvl_um

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(*sim_geometry.size[0:2], thickness_box_placement_um + 0.3 * mesh_wvl_um),
    )

    _eval_fns, _eval_fn_names = make_eval_fns(monitor_size_wvl)

    sim_path_dir = tmp_path / f"test{test_number}"
    sim_path_dir.mkdir()

    mesh_cell_override_size = mesh_wvl_um / mesh_refinement_factor

    if RUN_WITH_FD_CONVERGENCE:
        fd_step = np.linspace(2 * mesh_cell_override_size, mesh_cell_override_size, 2)
    else:
        fd_step = np.array([mesh_cell_override_size])

    z_planes = np.linspace(-0.5 * thickness_box_placement_um, 0.5 * thickness_box_placement_um, 3)

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

    obj_val_and_grad = ag.value_and_grad(objective)

    _obj, adj_grad = obj_val_and_grad([polyslab])

    fd_grad, valid_mask = run_and_process_fd(
        polyslab_parameters=polyslab, fd_step=fd_step, objective=objective
    )

    fd_grad = np.array(fd_grad)
    adj_grad = np.squeeze(np.array(adj_grad))
    valid_mask = np.array(valid_mask)

    vertex_fd = fd_grad[valid_mask]
    vertex_adj = adj_grad[valid_mask]

    assert len(vertex_fd) >= MIN_FD_COMPARISON, "Too many vertices were trimmed"

    (
        vertex_overlap_deg,
        vertex_error_mean,
        vertex_error_std,
        vertex_error_norm_mean,
        vertex_error_norm_std,
    ) = compare_fd_adj(vertex_fd, vertex_adj)

    vertex_data = np.zeros((2, len(vertex_fd)))

    vertex_data[SAVE_FD_LOC, :] = vertex_fd
    vertex_data[SAVE_ADJ_LOC, :] = vertex_adj

    print(f"\n2D PEC PolySlab Test {test_number} Summary:")
    print(f"Mesh wavelength (um): {mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {adj_wvl_um}")
    print(f"Monitor size (wavelengths): {monitor_size_wvl}")
    print(f"Mesh refinement factor: {mesh_refinement_factor}")
    print(f"Eval function: {eval_fn_name}")
    print(f"Vertex mean (std): {vertex_error_mean} ({vertex_error_std})")
    print(f"Vertex norm mean (std): {vertex_error_norm_mean} ({vertex_error_norm_std})")
    print(f"Vertex overlap deg: {vertex_overlap_deg}")
    print("\n")

    if SAVE_FD_ADJ_DATA:
        np.save(
            f"{NUMERICAL_RESULTS_DATA_DIR}/rf_polyslab_2d_vertex_fd_adj_data_test_{test_number}.npy",
            vertex_data,
        )

        yvals = [vertex_error_mean, vertex_error_norm_mean]
        yerrs = [vertex_error_std, vertex_error_norm_std]
        bars = plt.bar(
            ["vertex\nerror", "vertex\nerror\nnorm"],
            yvals,
            yerr=yerrs,
            color="skyblue",
            edgecolor="black",
            capsize=5,
        )

        plt.title(f"Vector Alignment (deg):\n{vertex_overlap_deg:.2f}")
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

        plt.savefig(
            f"{NUMERICAL_RESULTS_DATA_DIR}/rf_2d_polyslab_summary_plot_test_{test_number}.png"
        )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "rf_3d_test_parameters, dir_name",
    zip(
        rf_3d_test_parameters,
        ([NUMERICAL_RESULTS_DATA_DIR] if SAVE_FD_ADJ_DATA else [None]) * len(rf_3d_test_parameters),
    ),
    indirect=["dir_name"],
)
def test_finite_difference_3d_polyslab_pec(
    rf_3d_test_parameters, rng, tmp_path, create_directory, redirect_stdout_to_stderr
):
    """Test a variety of autograd permittivity gradients for 3D PEC `PolySlab` by"""
    """comparing them to numerical finite difference."""

    test_number = rf_3d_test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_size_wvl,
        polyslab_z_thickness_wvl,
        eval_fn,
        eval_fn_name,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_size_wvl",
        "polyslab_z_thickness_wvl",
        "eval_fn",
        "eval_fn_name",
        "test_number",
    )(rf_3d_test_parameters)

    dim_um = 1.5 * mesh_wvl_um

    thickness_box_placement_um = 1.0 * mesh_wvl_um

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(*sim_geometry.size[0:2], thickness_box_placement_um),
    )

    _eval_fns, _eval_fn_names = make_eval_fns(monitor_size_wvl)

    sim_path_dir = tmp_path / f"test{test_number}"
    sim_path_dir.mkdir()

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

    obj_val_and_grad = ag.value_and_grad(objective)

    _obj, adj_grad = obj_val_and_grad([polyslab])

    fd_grad, valid_mask = run_and_process_fd(
        polyslab_parameters=polyslab, fd_step=fd_step, objective=objective
    )

    fd_grad = np.array(fd_grad)
    adj_grad = np.squeeze(np.array(adj_grad))
    valid_mask = np.array(valid_mask)

    vertex_fd = fd_grad[valid_mask]
    vertex_adj = adj_grad[valid_mask]

    assert len(vertex_fd) >= MIN_FD_COMPARISON, "Too many vertices were trimmed"

    (
        vertex_overlap_deg,
        vertex_error_mean,
        vertex_error_std,
        vertex_error_norm_mean,
        vertex_error_norm_std,
    ) = compare_fd_adj(vertex_fd, vertex_adj)

    vertex_data = np.zeros((2, len(vertex_fd)))

    vertex_data[SAVE_FD_LOC, :] = vertex_fd
    vertex_data[SAVE_ADJ_LOC, :] = vertex_adj

    print(f"\n3D PEC PolySlab Test {test_number} Summary:")
    print(f"Mesh wavelength (um): {mesh_wvl_um}")
    print(f"Adjoint wavelength (um): {adj_wvl_um}")
    print(f"Monitor size (wavelengths): {monitor_size_wvl}")
    print(f"Polyslab z thickness (wavelengths): {polyslab_z_thickness_wvl}")
    print(f"Eval function: {eval_fn_name}")
    print(f"Vertex mean (std): {vertex_error_mean} ({vertex_error_std})")
    print(f"Vertex norm mean (std): {vertex_error_norm_mean} ({vertex_error_norm_std})")
    print(f"Vertex overlap deg: {vertex_overlap_deg}")
    print("\n")

    if SAVE_FD_ADJ_DATA:
        np.save(
            f"{NUMERICAL_RESULTS_DATA_DIR}/rf_polyslab_3d_vertex_fd_adj_data_test_{test_number}.npy",
            vertex_data,
        )

        yvals = [vertex_error_mean, vertex_error_norm_mean]
        yerrs = [vertex_error_std, vertex_error_norm_std]
        bars = plt.bar(
            ["vertex\nerror", "vertex\nerror\nnorm"],
            yvals,
            yerr=yerrs,
            color="skyblue",
            edgecolor="black",
            capsize=5,
        )

        plt.title(f"Vector Alignment (deg):\n{vertex_overlap_deg:.2f}")
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

        plt.savefig(
            f"{NUMERICAL_RESULTS_DATA_DIR}/rf_3d_polyslab_summary_plot_test_{test_number}.png"
        )
