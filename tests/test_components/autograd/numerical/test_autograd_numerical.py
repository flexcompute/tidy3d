# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

import operator

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import tidy3d as td
import tidy3d.web as web

PLOT_FD_ADJ_COMPARISON = False
NUM_FINITE_DIFFERENCE = 10
SAVE_FD_ADJ_DATA = False
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
NUMERICAL_RESULTS_SUBDIR = "numerical_field_test"

RMS_THRESHOLD = 0.25

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


FINITE_DIFF_PERM_SEED = 1.5**2
MESH_FACTOR_DESIGN = 30.0


def get_sim_geometry(mesh_wvl_um):
    return td.Box(size=(5 * mesh_wvl_um, 5 * mesh_wvl_um, 7 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    monitor_size_wvl,
    box_for_override,
    monitor_bg_index=1.0,
    run_time=1e-11,
):
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    sim_size_um = sim_geometry.size
    sim_center_um = sim_geometry.center

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    dl_design = mesh_wvl_um / MESH_FACTOR_DESIGN

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
    )

    field_monitor = td.FieldMonitor(
        center=(0, 0, 0.25 * sim_size_um[2]),
        size=tuple(dim * mesh_wvl_um for dim in monitor_size_wvl),
        name="monitor_fields",
        freqs=[freq0],
    )

    monitor_index_block = td.Box(
        center=(0, 0, 0.25 * sim_size_um[2] + mesh_wvl_um),
        size=(*tuple(2 * size for size in sim_size_um[0:2]), mesh_wvl_um + 0.5 * sim_size_um[2]),
    )
    monitor_index_block_structure = td.Structure(
        geometry=monitor_index_block, medium=td.Medium(permittivity=monitor_bg_index**2)
    )

    sim_base = td.Simulation(
        center=sim_center_um,
        size=sim_size_um,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=30,
            wavelength=mesh_wvl_um,
            override_structures=mesh_overrides,
        ),
        structures=[monitor_index_block_structure],
        sources=[src],
        monitors=[field_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_objective_function(
    geometry, create_sim_base, eval_fn, sim_path_dir, perm_init, cm_interp_method
):
    block_structure = td.Structure.from_permittivity_array(
        eps_data=perm_init,
        geometry=geometry,
    )

    sim_base = create_sim_base()

    sim_with_block = sim_base.updated_copy(structures=(*sim_base.structures, block_structure))

    # use a fixed grid for all forward and finite difference simulations
    grid_fixed = sim_with_block.grid

    def objective(perm_arrays):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(perm_arrays)):
            block_structure = td.Structure.from_permittivity_array(
                eps_data=perm_arrays[idx],
                geometry=geometry,
            )

            block_structure = block_structure.updated_copy(
                medium=block_structure.medium.updated_copy(interp_method=cm_interp_method)
            )

            sim_with_block = sim_base.updated_copy(
                structures=(*sim_base.structures, block_structure),
                grid_spec=td.GridSpec.from_grid(grid_fixed),
            )

            simulation_dict[f"numerical_field_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(perm_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_field_testing_{idx}"]))

        if len(perm_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def make_eval_fns(monitor_size_wvl):
    num_nonzero_spatial_dims = 3 - np.sum(np.isclose(monitor_size_wvl, 0))

    def intensity(sim_data):
        field_data = sim_data["monitor_fields"]
        shape_x, shape_y, shape_z, *_ = field_data.Ex.values.shape

        total = 0.0
        return np.sum(
            np.abs(field_data.Ex.values[shape_x // 2, shape_y // 2, shape_z // 2]) ** 2
            + np.abs(field_data.Ey.values[shape_x // 2, shape_y // 2, shape_z // 2]) ** 2
            + np.abs(field_data.Ez.values[shape_x // 2, shape_y // 2, shape_z // 2]) ** 2
        )

    eval_fns = [intensity]
    eval_fn_names = ["intensity"]

    if num_nonzero_spatial_dims == 2:

        def flux(sim_data):
            field_data = sim_data["monitor_fields"]

            return np.sum(field_data.flux.values)

        eval_fns.append(flux)
        eval_fn_names.append("flux")

    return eval_fns, eval_fn_names


background_indices = [1.0, 1.5]
mesh_wvls_um = [1.55, 1.55, 10 * 1.55, 10 * 1.55]
adj_wvls_um = [1.55, 2.2, 10 * 1.55, 10 * 2.2]
monitor_sizes_3d_wvl = [(0.5, 0.5, 0), (0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0)]
cm_interp_methods = ["nearest", "linear"]

field_data_test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for monitor_size_wvl in monitor_sizes_3d_wvl:
        eval_fns, eval_fn_names = make_eval_fns(monitor_size_wvl)

        for monitor_bg_index in background_indices:
            for eval_fn_idx, eval_fn in enumerate(eval_fns):
                for cm_interp_method in cm_interp_methods:
                    field_data_test_parameters.append(
                        {
                            "mesh_wvl_um": mesh_wvl_um,
                            "adj_wvl_um": adj_wvl_um,
                            "monitor_size_wvl": monitor_size_wvl,
                            "monitor_bg_index": monitor_bg_index,
                            "eval_fn": eval_fn,
                            "eval_fn_name": eval_fn_names[eval_fn_idx],
                            "cm_interp_method": cm_interp_method,
                            "test_number": test_number,
                        }
                    )

                    test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize("field_data_test_parameters", field_data_test_parameters)
def test_finite_difference_field_data(
    field_data_test_parameters, rng, numerical_case_dir, redirect_stdout_to_stderr
):
    """Test a variety of autograd permittivity gradients for FieldData by"""
    """comparing them to numerical finite difference."""

    num_tests = 0
    for monitor_size_wvl in monitor_sizes_3d_wvl:
        eval_fns, _ = make_eval_fns(monitor_size_wvl)
        num_tests += len(eval_fns) * len(background_indices) * len(mesh_wvls_um)

    test_results = np.zeros((2, NUM_FINITE_DIFFERENCE))

    test_number = field_data_test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_size_wvl,
        monitor_bg_index,
        eval_fn,
        eval_fn_name,
        cm_interp_method,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_size_wvl",
        "monitor_bg_index",
        "eval_fn",
        "eval_fn_name",
        "cm_interp_method",
        "test_number",
    )(field_data_test_parameters)

    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(center=(0, 0, 0), size=(dim_um, dim_um, thickness_um))

    dim = 1 + int(dim_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    Nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0), size=sim_geometry.size[0:2] + (thickness_um + mesh_wvl_um,)
    )

    eval_fns, eval_fn_names = make_eval_fns(monitor_size_wvl)

    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, Nz))

    objective = create_objective_function(
        block,
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        monitor_size_wvl=monitor_size_wvl,
        box_for_override=box_for_override,
        monitor_bg_index=monitor_bg_index: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            monitor_size_wvl=monitor_size_wvl,
            box_for_override=box_for_override,
            monitor_bg_index=monitor_bg_index,
        ),
        eval_fn,
        sim_path_dir=str(sim_path_dir),
        perm_init=perm_init,
        cm_interp_method=cm_interp_method,
    )

    obj_val_and_grad = ag.value_and_grad(objective)

    obj, adj_grad = obj_val_and_grad([perm_init])

    # empirical step size from running other finite difference tests for field
    # cases with permittivity
    fd_step = 0.1

    all_perm = []
    pattern_dot_adj_gradient = np.zeros(NUM_FINITE_DIFFERENCE)

    for fd_idx in range(NUM_FINITE_DIFFERENCE):
        random_pattern = rng.random((dim, dim, Nz)) - 0.5
        random_pattern = gaussian_filter(random_pattern, sigma=3)
        random_pattern /= np.linalg.norm(random_pattern)

        pattern_dot_adj_gradient[fd_idx] = np.sum(random_pattern * adj_grad)

        perm_up = perm_init.copy() + fd_step * random_pattern
        perm_down = perm_init.copy() - fd_step * random_pattern

        all_perm.append(perm_up)
        all_perm.append(perm_down)

    all_obj = objective(all_perm)

    fd_grad = np.zeros(NUM_FINITE_DIFFERENCE)
    for fd_idx in range(NUM_FINITE_DIFFERENCE):
        obj_up_location = 2 * fd_idx
        obj_down_location = 2 * fd_idx + 1

        fd_grad[fd_idx] = (all_obj[obj_up_location] - all_obj[obj_down_location]) / (2 * fd_step)

    rms_error = np.linalg.norm(fd_grad - pattern_dot_adj_gradient)
    fd_mag = np.linalg.norm(fd_grad)
    adj_mag = np.linalg.norm(pattern_dot_adj_gradient)
    percentage_error = 100.0 * np.mean(
        np.abs(fd_grad - pattern_dot_adj_gradient) / (np.abs(fd_grad) + np.finfo(np.float64).eps)
    )

    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{test_number}")
    print(f"Mesh and adjoint wavelengths: {mesh_wvl_um}, {adj_wvl_um}")
    print(f"Monitor size: {monitor_size_wvl}")
    print(f"Background index for monitor: {monitor_bg_index}")
    print(f"Eval function: {eval_fn_name}")
    print(f"Custom medium interpolation method: {cm_interp_method}")
    print(f"RMS Error: {rms_error}")
    print(f"FD, Adj magnitudes: {fd_mag}, {adj_mag}")
    print(f"Percentage Error: {percentage_error}")
    print("-" * 20)
    print("\n" * 3)

    if PLOT_FD_ADJ_COMPARISON:
        plt.plot(pattern_dot_adj_gradient, color="g", linewidth=2.0, label="Adjoint")
        plt.plot(fd_grad, color="b", linewidth=1.5, linestyle="--", label="Finite difference")
        plt.title(f"Gradient for objective: {eval_fn_name}")
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.legend()
        plt.show()

    test_results[SAVE_FD_LOC, :] = fd_grad
    test_results[SAVE_ADJ_LOC, :] = pattern_dot_adj_gradient

    save_idx = test_number + 1
    save_path = None
    if SAVE_FD_ADJ_DATA:
        results_dir = numerical_case_dir / NUMERICAL_RESULTS_SUBDIR
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"results_{save_idx}.npy"

    try:
        assert rms_error < RMS_THRESHOLD * fd_mag, "RMS error magnitude too large"
    finally:
        if save_path is not None:
            np.save(save_path, test_results)

    test_number += 1
