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
SAVE_FD_ADJ_DATA = True
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
NUMERICAL_RESULTS_SUBDIR = "numerical_periodic_test"

RMS_THRESHOLD = 0.25

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


FINITE_DIFF_PERM_SEED = 1.5**2
MESH_FACTOR_DESIGN = 30.0


def get_sim_geometry(mesh_wvl_um):
    return td.Box(
        size=(3.5 * mesh_wvl_um, 3.5 * mesh_wvl_um, 7 * mesh_wvl_um),
        center=(3.5 * mesh_wvl_um / 4.0, 0, 0),
    )


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    box_for_override,
    pw_angle_deg,
    grating_mode,
    monitor_bg_index=1.0,
    run_time=1e-11,
):
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    sim_size_um = sim_geometry.size
    sim_center_um = sim_geometry.center

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

    src_size = (*sim_size_um[0:2], 0)

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um

    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)

    src = td.PlaneWave(
        center=(1.0, 0, -0.25 * sim_size_um[2]),
        size=[td.inf, td.inf, 0],
        source_time=pulse,
        direction="+",
        angle_theta=(pw_angle_deg * np.pi / 180.0),
    )

    bloch_x = td.Boundary.bloch_from_source(
        source=src,
        domain_size=sim_size_um[0],
        axis=0,
    )
    bloch_y = td.Boundary.bloch_from_source(
        source=src,
        domain_size=sim_size_um[1],
        axis=1,
    )

    boundary_spec = td.BoundarySpec(
        x=bloch_x,
        y=bloch_y,
        z=td.Boundary.pml(num_layers=48),
    )

    assert (grating_mode == "transmission") or (grating_mode == "reflection"), (
        "Unknown grating mode specified!"
    )
    if grating_mode == "transmission":
        diffraction_monitor = td.DiffractionMonitor(
            center=(
                0,
                sim_center_um[1],
                0.25 * sim_size_um[2],
            ),
            size=(np.inf, np.inf, 0),
            name="monitor_diffraction",
            freqs=[freq0],
            normal_dir="+",
        )
    else:
        diffraction_monitor = td.DiffractionMonitor(
            center=(sim_center_um[0], sim_center_um[1], -0.35 * sim_size_um[2]),
            size=(np.inf, np.inf, 0),
            name="monitor_diffraction",
            freqs=[freq0],
            normal_dir="-",
        )

    monitor_index_block = td.Box(
        center=(sim_center_um[0], sim_center_um[1], 0.25 * sim_size_um[2] + mesh_wvl_um),
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
        monitors=[diffraction_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_objective_function(geometry, create_sim_base, eval_fn, sim_path_dir):
    def objective(perm_arrays):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(perm_arrays)):
            block_structure = td.Structure.from_permittivity_array(
                eps_data=perm_arrays[idx],
                geometry=geometry,
            )

            sim_with_block = sim_base.updated_copy(
                structures=(*sim_base.structures, block_structure)
            )

            simulation_dict[f"numerical_periodic_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(perm_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_periodic_testing_{idx}"]))

        if len(perm_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def make_eval_fns(orders_x, orders_y, polarization):
    def transmission_order_pol_amp_sq(sim_data):
        total = 0.0

        for order_x_val in orders_x:
            for order_y_val in orders_y:
                total += np.sum(
                    np.abs(
                        sim_data["monitor_diffraction"]
                        .amps.sel(
                            polarization=polarization, orders_x=order_x_val, orders_y=order_y_val
                        )
                        .data
                    )
                    ** 2
                )

        return total

    eval_fns = [transmission_order_pol_amp_sq]
    eval_fn_names = [f"transmission_order_pol_amp_sq_{orders_x}_{orders_y}_{polarization}"]

    return eval_fns, eval_fn_names


background_indices = [1.0, 1.5]
mesh_wvls_um = [1.55]
adj_wvls_um = [1.55]

orders_x = [(0,), (1,), (2,), (1, 2)]
orders_y = [(0,), (0,), (0,), (1,)]
polarizations = ["p", "p", "p", "s"]

grating_modes = ["transmission", "reflection"]

pw_angles_deg = [0.0, 10.0]

periodic_test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for grating_mode in grating_modes:
        for order_idx in range(len(orders_x)):
            eval_fns, eval_fn_names = make_eval_fns(
                orders_x=orders_x[order_idx],
                orders_y=orders_y[order_idx],
                polarization=polarizations[order_idx],
            )

            for pw_angle_deg in pw_angles_deg:
                for monitor_bg_index in background_indices:
                    for eval_fn_idx, eval_fn in enumerate(eval_fns):
                        periodic_test_parameters.append(
                            {
                                "mesh_wvl_um": mesh_wvl_um,
                                "adj_wvl_um": adj_wvl_um,
                                "monitor_bg_index": monitor_bg_index,
                                "pw_angle_deg": pw_angle_deg,
                                "order_x": orders_x[order_idx],
                                "order_y": orders_y[order_idx],
                                "polarization": polarizations[order_idx],
                                "grating_mode": grating_mode,
                                "eval_fn": eval_fn,
                                "eval_fn_name": eval_fn_names[eval_fn_idx],
                                "test_number": test_number,
                            }
                        )

                        test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize("periodic_test_parameters", periodic_test_parameters)
def test_finite_difference_diffraction_data(
    periodic_test_parameters, rng, numerical_case_dir, redirect_stdout_to_stderr
):
    """Test a variety of autograd permittivity gradients for DiffractionData by"""
    """comparing them to numerical finite difference."""

    test_results = np.zeros((2, NUM_FINITE_DIFFERENCE))

    test_number = periodic_test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_bg_index,
        pw_angle_deg,
        order_x,
        order_y,
        polarization,
        grating_mode,
        eval_fn,
        eval_fn_name,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_bg_index",
        "pw_angle_deg",
        "order_x",
        "order_y",
        "polarization",
        "grating_mode",
        "eval_fn",
        "eval_fn_name",
        "test_number",
    )(periodic_test_parameters)

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(
        center=(sim_geometry.center[0], sim_geometry.center[1], 0),
        size=(dim_um, dim_um, thickness_um),
    )

    dim = 1 + int(dim_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    Nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

    box_for_override = td.Box(
        center=(sim_geometry.center[0], sim_geometry.center[1], 0),
        size=(*sim_geometry.size[0:2], thickness_um + mesh_wvl_um),
    )

    _eval_fns, _eval_fn_names = make_eval_fns(
        orders_x=order_x, orders_y=order_y, polarization=polarization
    )

    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    objective = create_objective_function(
        block,
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        box_for_override=box_for_override,
        pw_angle_deg=pw_angle_deg,
        grating_mode=grating_mode,
        monitor_bg_index=monitor_bg_index: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            box_for_override=box_for_override,
            pw_angle_deg=pw_angle_deg,
            grating_mode=grating_mode,
            monitor_bg_index=monitor_bg_index,
        ),
        eval_fn,
        sim_path_dir=str(sim_path_dir),
    )

    obj_val_and_grad = ag.value_and_grad(objective)

    perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, Nz))

    _obj, adj_grad = obj_val_and_grad([perm_init])

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
    print(f"Input plane wave angle (deg): {pw_angle_deg}")
    print(f"(X, Y) order, polarization: ({order_x}, {order_y}), {polarization}")
    print(f"Background index for monitor: {monitor_bg_index}")
    print(f"Eval function: {eval_fn_name}")
    print(f"RMS Error: {rms_error}")
    print(f"FD, Adj magnitudes: {fd_mag}, {adj_mag}")
    print(f"Percentage Error: {percentage_error}")
    print("-" * 20)
    print("\n" * 3)

    test_results[SAVE_FD_LOC, :] = fd_grad
    test_results[SAVE_ADJ_LOC, :] = pattern_dot_adj_gradient

    save_idx = test_number + 1
    save_path = None
    if SAVE_FD_ADJ_DATA:
        results_dir = numerical_case_dir / NUMERICAL_RESULTS_SUBDIR
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"results_{save_idx}.npy"

    if PLOT_FD_ADJ_COMPARISON:
        plt.plot(pattern_dot_adj_gradient, color="g", linewidth=2.0)
        plt.plot(fd_grad, color="b", linewidth=1.5, linestyle="--")
        plt.title(f"Gradient for objective: {eval_fn_name}")
        plt.legend(["Adjoint", "Finite difference"])
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.show()

    try:
        assert rms_error < RMS_THRESHOLD * fd_mag, "RMS error magnitude too large"
    finally:
        if save_path is not None:
            np.save(save_path, test_results)
