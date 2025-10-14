# tests numerical_structures autograd hook for run_custom and run_async_custom and compares to numerically computed finite difference gradients
from __future__ import annotations

import operator
import sys

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
import trimesh
import xarray as xr

import tidy3d as td
from tidy3d.web.api.autograd.autograd import run_async_custom, run_custom

PLOT_FD_ADJ_COMPARISON = True
NUM_FINITE_DIFFERENCE = 10
SAVE_FD_ADJ_DATA = True
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
NUMERICAL_RESULTS_DATA_DIR = "./numerical_numerical_structures_test/"
SHOW_PRINT_STATEMENTS = True

OVERLAP_ERROR_THRESHOLD_DEG = 15.0

ADJOINT_SPHERE_PERMITTIVITY = 1.5**2

RMS_THRESHOLD = 0.25

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

if SHOW_PRINT_STATEMENTS:
    sys.stdout = sys.stderr


SIMULATION_SIZE_MESH_WVL_FACTOR = 3.5
SIMULATION_HEIGHT_WVL_FACTOR = 5

RING_OFFSET_MAX_MESH_WVL_FACTOR = 0.25
RING_MIN_RADIUS_MESH_WVL_FACTOR = 0.3
RING_MAX_RADIUS_MESH_WVL_FACOTOR = 0.4

FD_STEP_MESH_WVL_FACTOR = 1.0 / 75.0


def get_sim_geometry(mesh_wvl_um):
    return td.Box(
        size=(
            SIMULATION_SIZE_MESH_WVL_FACTOR * mesh_wvl_um,
            SIMULATION_SIZE_MESH_WVL_FACTOR * mesh_wvl_um,
            SIMULATION_HEIGHT_WVL_FACTOR * mesh_wvl_um,
        ),
        center=(0, 0, 0),
    )


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    pw_angle_deg,
    monitor_bg_index=1.0,
    run_time=2e-11,
):
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    sim_size_um = sim_geometry.size
    sim_center_um = sim_geometry.center

    src_size = sim_size_um[0:2] + (0,)

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um

    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)

    src = td.PlaneWave(
        center=(sim_center_um[0], sim_center_um[1], -2.0),
        size=[td.inf, td.inf, 0],
        source_time=pulse,
        direction="+",
        angle_theta=(pw_angle_deg * np.pi / 180.0),
    )

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    field_monitor = td.FieldMonitor(
        center=(
            sim_center_um[0],
            sim_center_um[1],
            mesh_wvl_um / 1.5,
        ),
        size=(mesh_wvl_um, mesh_wvl_um, 0),
        name="monitor_fields",
        freqs=[freq0],
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
        ),
        structures=[monitor_index_block_structure],
        sources=[src],
        monitors=[field_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_ring(params):
    ring_mesh = trimesh.creation.annulus(
        r_min=params[0], r_max=params[1], height=params[2], sections=100
    )

    ring_geo = td.TriangleMesh.from_trimesh(ring_mesh)

    return td.Structure(geometry=ring_geo, medium=td.Medium(permittivity=1.5**2))


def vjp_ring(parameters, derivative_info):
    max_frequency = np.max(derivative_info.frequencies)
    min_wvl = td.C_0 / max_frequency

    step_size = min_wvl / 20.0

    ps_paths = set()
    ps_paths.update({("permittivity",)})

    # pass interpolators to PolySlab if available to avoid redundant conversions
    update_kwargs = {
        "paths": list(ps_paths),
        "deep": False,
    }
    derivative_info_custom_medium = derivative_info.updated_copy(**update_kwargs)

    params_np = np.array(parameters)

    vjps = {}
    for path in derivative_info.paths:
        param_idx = path[0]

        params_up = params_np.copy()
        params_down = params_np.copy()

        params_up[param_idx] += step_size
        params_down[param_idx] -= step_size

        rin_up = create_ring(params_up)
        ring_down = create_ring(params_down)

        eps_up = derivative_info.updated_epsilon(rin_up.geometry)
        eps_down = derivative_info.updated_epsilon(ring_down.geometry)

        eps_grad = (eps_up - eps_down) / (2 * step_size)

        custom_medium = td.CustomMedium(permittivity=xr.ones_like(eps_grad.isel(f=0, drop=True)))
        vjps_custom_medium = custom_medium._compute_derivatives(derivative_info_custom_medium)

        total_grad = np.real(np.sum(eps_grad.sum("f").data * vjps_custom_medium[("permittivity",)]))

        vjps[path] = total_grad

    return vjps


def create_objective_function(geometry, create_sim_base, eval_fn, run_fn, sim_path_dir):
    def objective(ring_parameters_lists):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(ring_parameters_lists)):
            simulation_dict[f"numerical_numerical_structures_testing_{idx}"] = sim_base.copy()

        assert (run_fn == "run_custom") or (run_fn == "run_async_custom"), (
            "Unrecognized run function!"
        )
        if run_fn == "run_custom":
            sim_data = {}
            idx = 0
            for key, sim_val in simulation_dict.items():
                ring_generator = {
                    0: {
                        "function": create_ring,
                        "parameters": ring_parameters_lists[idx],
                        "vjp": vjp_ring,
                    }
                }
                sim_data[key] = run_custom(
                    sim_val,
                    local_gradient=LOCAL_GRADIENT,
                    verbose=VERBOSE,
                    numerical_structures=ring_generator,
                )

                idx += 1
        elif run_fn == "run_async_custom":
            user_vjp_dict = {}
            numerical_structures_dict = {}

            for idx, key in enumerate(simulation_dict):
                user_vjp_dict[key] = ((1, "radius", vjp_ring), (1, "center", vjp_ring))

                ring_generator = {
                    0: {
                        "function": create_ring,
                        "parameters": ring_parameters_lists[idx],
                        "vjp": vjp_ring,
                    }
                }
                numerical_structures_dict[key] = ring_generator

            sim_data = run_async_custom(
                simulation_dict,
                path_dir=sim_path_dir,
                local_gradient=LOCAL_GRADIENT,
                verbose=VERBOSE,
                numerical_structures=numerical_structures_dict,
            )

        objective_vals = []
        for idx in range(len(ring_parameters_lists)):
            objective_vals.append(
                eval_fn(sim_data[f"numerical_numerical_structures_testing_{idx}"])
            )

        if len(ring_parameters_lists) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def make_eval_fns():
    def transmission(sim_data):
        total = 0.0

        return np.sum(np.abs(sim_data["monitor_fields"].flux.data) ** 2)

    eval_fns = [transmission]
    eval_fn_names = ["transmission"]

    return eval_fns, eval_fn_names


background_indices = [1.0]
mesh_wvls_um = [1.5]
adj_wvls_um = [1.5]

orders_x = [(1,)]
orders_y = [(0,)]
polarizations = ["p"]


pw_angles_deg = [0.0]

run_functions = ["run_custom", "run_async_custom"]

test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    eval_fns, eval_fn_names = make_eval_fns()

    for pw_angle_deg in pw_angles_deg:
        for monitor_bg_index in background_indices:
            for eval_fn_idx, eval_fn in enumerate(eval_fns):
                for run_fn in run_functions:
                    test_parameters.append(
                        {
                            "mesh_wvl_um": mesh_wvl_um,
                            "adj_wvl_um": adj_wvl_um,
                            "monitor_bg_index": monitor_bg_index,
                            "pw_angle_deg": pw_angle_deg,
                            "eval_fn": eval_fn,
                            "eval_fn_name": eval_fn_names[eval_fn_idx],
                            "run_fn": run_fn,
                            "test_number": test_number,
                        }
                    )

                    test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize(
    "test_parameters, dir_name",
    zip(
        test_parameters,
        ([NUMERICAL_RESULTS_DATA_DIR] if SAVE_FD_ADJ_DATA else [None]) * len(test_parameters),
    ),
    indirect=["dir_name"],
)
def test_finite_difference_numerical_structures(test_parameters, rng, tmp_path, create_directory):
    """Test a variety of autograd permittivity gradients for DiffractionData by"""
    """comparing them to numerical finite difference."""

    test_number = test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_bg_index,
        pw_angle_deg,
        eval_fn,
        eval_fn_name,
        run_fn,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_bg_index",
        "pw_angle_deg",
        "eval_fn",
        "eval_fn_name",
        "run_fn",
        "test_number",
    )(test_parameters)

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(
        center=(sim_geometry.center[0], sim_geometry.center[1], 0),
        size=(dim_um, dim_um, thickness_um),
    )

    eval_fns, eval_fn_names = make_eval_fns()

    sim_path_dir = tmp_path / f"test{test_number}"
    sim_path_dir.mkdir()

    objective = create_objective_function(
        block,
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        pw_angle_deg=pw_angle_deg,
        monitor_bg_index=monitor_bg_index: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            pw_angle_deg=pw_angle_deg,
            monitor_bg_index=monitor_bg_index,
        ),
        eval_fn,
        run_fn,
        sim_path_dir=str(sim_path_dir),
    )

    obj_val_and_grad = ag.value_and_grad(objective)

    ring_init_mesh_wvl_factor = [0.15, 0.30, 0.2]
    ring_init = [r * mesh_wvl_um for r in ring_init_mesh_wvl_factor]

    test_results = np.zeros((2, len(ring_init)))

    obj, adj_grad = obj_val_and_grad([ring_init])
    adj_grad = np.squeeze(np.array(adj_grad))

    # empirical step size from running other finite difference tests for field
    # cases with permittivity
    fd_step = FD_STEP_MESH_WVL_FACTOR * mesh_wvl_um

    all_rings = []
    for fd_idx in range(len(ring_init)):
        rin_up = ring_init.copy()
        ring_down = ring_init.copy()

        rin_up[fd_idx] += fd_step
        ring_down[fd_idx] -= fd_step

        all_rings.append(rin_up)
        all_rings.append(ring_down)

    all_obj = objective(all_rings)

    fd_grad = np.zeros(len(ring_init))
    for fd_idx in range(len(ring_init)):
        obj_up_location = 2 * fd_idx
        obj_down_location = 2 * fd_idx + 1

        fd_grad[fd_idx] = (all_obj[obj_up_location] - all_obj[obj_down_location]) / (2 * fd_step)

    rms_error = np.linalg.norm(fd_grad - adj_grad)
    fd_mag = np.linalg.norm(fd_grad)
    adj_mag = np.linalg.norm(adj_grad)

    dot = np.sum((fd_grad / fd_mag) * (adj_grad / adj_mag))
    overlap_deg = np.arccos(dot) * 180.0 / np.pi

    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{test_number}")
    print(f"Mesh and adjoint wavelengths: {mesh_wvl_um}, {adj_wvl_um}")
    print(f"Input plane wave angle (deg): {pw_angle_deg}")
    print(f"Background index for monitor: {monitor_bg_index}")
    print(f"Eval function: {eval_fn_name}")
    print(f"RMS Error: {rms_error}")
    print(f"Gradient overlap (deg): {overlap_deg}")
    print(f"FD, Adj magnitudes: {fd_mag}, {adj_mag}")
    print("-" * 20)
    print("\n" * 3)

    assert overlap_deg < OVERLAP_ERROR_THRESHOLD_DEG, (
        "Adjoint and finite difference gradients misaligned."
    )

    test_results[SAVE_FD_LOC, :] = fd_grad
    test_results[SAVE_ADJ_LOC, :] = adj_grad

    test_number += 1

    if PLOT_FD_ADJ_COMPARISON:
        plt.plot(adj_grad, color="g", linewidth=2.0)
        plt.plot(fd_grad, color="b", linewidth=1.5, linestyle="--")
        plt.title(f"Gradient for objective: {eval_fn_name}")
        plt.legend(["Adjoint", "Finite difference"])
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.show()

    if SAVE_FD_ADJ_DATA:
        np.save(f"{NUMERICAL_RESULTS_DATA_DIR}/results_{test_number}.npy", test_results)
