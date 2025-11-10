# tests user_vjp and numerical_structures autograd hooks for ComponentModeler and compares to numerically computed finite difference gradients
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
from tidy3d.plugins.smatrix import ComponentModeler, Port
from tidy3d.plugins.smatrix.run import _run_local

PLOT_FD_ADJ_COMPARISON = True
NUM_FINITE_DIFFERENCE = 10
SAVE_FD_ADJ_DATA = True
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = True
NUMERICAL_RESULTS_DATA_DIR = "./numerical_cm_user_vjp_numerical_structures_test/"
SHOW_PRINT_STATEMENTS = True

OVERLAP_ERROR_THRESHOLD_DEG = 10.0

ADJOINT_PERMITTIVITY = 1.5**2

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

if SHOW_PRINT_STATEMENTS:
    sys.stdout = sys.stderr


SIMULATION_SIZE_MESH_WVL_FACTOR = 7
SIMULATION_HEIGHT_WVL_FACTOR = 3

SPHERE_OFFSET_MAX_MESH_WVL_FACTOR = 0.25
SPHERE_MIN_RADIUS_MESH_WVL_FACTOR = 0.3
SPHERE_MAX_RADIUS_MESH_WVL_FACTOR = 0.4

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

    input_waveguide = td.Structure(
        geometry=td.Box(
            center=(-0.35 * sim_size_um[0], sim_center_um[1], sim_center_um[2]),
            size=(0.5 * sim_size_um[0], 0.35 * adj_wvl_um, 0.2 * adj_wvl_um),
        ),
        medium=td.Medium(permittivity=3.5**2),
    )

    output_waveguide = td.Structure(
        geometry=td.Box(
            center=(0.35 * sim_size_um[0], sim_center_um[1], sim_center_um[2]),
            size=(0.5 * sim_size_um[0], 0.35 * adj_wvl_um, 0.2 * adj_wvl_um),
        ),
        medium=td.Medium(permittivity=3.5**2),
    )

    num_modes = 1

    port_left = Port(
        center=input_waveguide.geometry.center,
        size=(0.0, adj_wvl_um, adj_wvl_um),
        mode_spec=td.ModeSpec(num_modes=num_modes),
        direction="+",
        name="left",
    )

    port_right = Port(
        center=output_waveguide.geometry.center,
        size=(0.0, adj_wvl_um, adj_wvl_um),
        mode_spec=td.ModeSpec(num_modes=num_modes),
        direction="-",
        name="right",
    )

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    ports = [port_left, port_right]

    return ports, td.Simulation(
        center=sim_center_um,
        size=sim_size_um,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=30,
            wavelength=1.5,
        ),
        boundary_spec=boundary_spec,
        sources=[],
        monitors=[],
        structures=[input_waveguide, output_waveguide],
        run_time=1e-11,
    )


def vjp_sphere(sphere, derivative_info):
    max_frequency = np.max(derivative_info.frequencies)
    min_wvl = td.C_0 / max_frequency

    step_size = min_wvl / 20.0

    ps_paths = set()
    ps_paths.update({("permittivity",)})

    update_kwargs = {
        "paths": list(ps_paths),
        "deep": False,
    }
    derivative_info_custom_medium = derivative_info.updated_copy(**update_kwargs)

    def finite_difference_gradient(perturb_up, perturb_down, derivative_info_):
        eps_up = derivative_info.updated_epsilon(sphere_up)
        eps_down = derivative_info.updated_epsilon(sphere_down)
        eps_grad = (eps_up - eps_down) / (2 * step_size)

        custom_medium = td.CustomMedium(permittivity=xr.ones_like(eps_grad.isel(f=0, drop=True)))
        vjps_custom_medium = custom_medium._compute_derivatives(derivative_info_custom_medium)

        total_grad = np.real(np.sum(eps_grad.sum("f").data * vjps_custom_medium[("permittivity",)]))

        return total_grad

    vjps = {}
    for path in derivative_info.paths:
        if path == ("radius",):
            sphere_up = sphere.updated_copy(radius=sphere.radius + step_size)
            sphere_down = sphere.updated_copy(radius=sphere.radius - step_size)
            vjps[path] = finite_difference_gradient(sphere_up, sphere_down, derivative_info)
        elif "center" in path:
            if len(path) == 1:
                center_indices = (0, 1, 2)
            else:
                _, center_index = path
                center_indices = [center_index]

            vjp_result = []
            for center_index in center_indices:
                center_up = list(sphere.center)
                center_down = list(sphere.center)

                center_up[center_index] += step_size
                center_down[center_index] -= step_size

                sphere_up = sphere.updated_copy(center=center_up)
                sphere_down = sphere.updated_copy(center=center_down)

                vjp_result.append(
                    finite_difference_gradient(sphere_up, sphere_down, derivative_info)
                )

            vjps[path] = vjp_result if len(path) == 1 else vjp_result[0]

    return vjps


def create_ring(params):
    ring_mesh = trimesh.creation.annulus(
        r_min=params[0], r_max=params[1], height=params[2], sections=100
    )

    rotator = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    ring_mesh.apply_transform(rotator)

    translate = trimesh.transformations.translation_matrix([-0.65, 0, 0])
    ring_mesh.apply_transform(translate)

    ring_geo = td.TriangleMesh.from_trimesh(ring_mesh)

    return td.Structure(geometry=ring_geo, medium=td.Medium(permittivity=ADJOINT_PERMITTIVITY))


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

        ring_up = create_ring(params_up)
        ring_down = create_ring(params_down)

        eps_up = derivative_info.updated_epsilon(ring_up.geometry)
        eps_down = derivative_info.updated_epsilon(ring_down.geometry)

        eps_grad = (eps_up - eps_down) / (2 * step_size)

        custom_medium = td.CustomMedium(permittivity=xr.ones_like(eps_grad.isel(f=0, drop=True)))
        vjps_custom_medium = custom_medium._compute_derivatives(derivative_info_custom_medium)

        total_grad = np.real(np.sum(eps_grad.sum("f").data * vjps_custom_medium[("permittivity",)]))

        vjps[path] = total_grad

    return vjps


def create_objective_function(geometry, create_sim_base, adj_wvl_um, sim_path_dir):
    def objective(geom_parameters_lists):
        ports, sim_base = create_sim_base()

        simulation_dict = {}
        geom_dict = {}
        for idx, geom_parameters in enumerate(geom_parameters_lists):
            sphere_structure = td.Structure(
                geometry=td.Sphere(center=geom_parameters[0:3], radius=geom_parameters[3]),
                medium=td.Medium(permittivity=ADJOINT_PERMITTIVITY),
            )

            sim_with_sphere = sim_base.updated_copy(
                structures=(*sim_base.structures, sphere_structure)
            )

            simulation_dict[f"numerical_user_vjp_testing_{idx}"] = sim_with_sphere.copy()
            geom_dict[f"numerical_user_vjp_testing_{idx}"] = geom_parameters

        sim_data = {}
        for key, sim_val in simulation_dict.items():
            modeler = ComponentModeler(
                simulation=sim_val,
                ports=ports,
                freqs=[td.C_0 / adj_wvl_um],
            )

            ring_generator = {
                0: {
                    "function": create_ring,
                    "parameters": geom_dict[key][4:],
                    "vjp": vjp_ring,
                }
            }

            sim_data[key] = _run_local(
                modeler,
                local_gradient=LOCAL_GRADIENT,
                verbose=VERBOSE,
                user_vjp=((1, "radius", vjp_sphere), (1, "center", vjp_sphere)),
                numerical_structures=ring_generator,
            )

        objective_vals = []
        for idx in range(len(geom_parameters_lists)):
            smatrix = sim_data[f"numerical_user_vjp_testing_{idx}"]
            objective_vals.append(np.sum(np.abs(smatrix.smatrix().values) ** 2))

        if len(geom_parameters_lists) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


background_indices = [1.0]
mesh_wvls_um = [1.5]
adj_wvls_um = [1.5]

orders_x = [(1,)]
orders_y = [(0,)]
polarizations = ["p"]


pw_angles_deg = [0.0]

test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for pw_angle_deg in pw_angles_deg:
        for monitor_bg_index in background_indices:
            test_parameters.append(
                {
                    "mesh_wvl_um": mesh_wvl_um,
                    "adj_wvl_um": adj_wvl_um,
                    "monitor_bg_index": monitor_bg_index,
                    "pw_angle_deg": pw_angle_deg,
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
def test_finite_difference_user_vjp(test_parameters, rng, tmp_path, create_directory):
    """Test a variety of autograd permittivity gradients for DiffractionData by"""
    """comparing them to numerical finite difference."""

    test_number = test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_bg_index,
        pw_angle_deg,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_bg_index",
        "pw_angle_deg",
        "test_number",
    )(test_parameters)

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(
        center=(sim_geometry.center[0], sim_geometry.center[1], 0),
        size=(dim_um, dim_um, thickness_um),
    )

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
        adj_wvl_um,
        sim_path_dir=str(sim_path_dir),
    )

    obj_val_and_grad = ag.value_and_grad(objective)

    sphere_init = (
        *rng.uniform(
            low=-SPHERE_OFFSET_MAX_MESH_WVL_FACTOR * mesh_wvl_um,
            high=SPHERE_OFFSET_MAX_MESH_WVL_FACTOR * mesh_wvl_um,
            size=2,
        ),
        0.0,
        *rng.uniform(
            low=SPHERE_MIN_RADIUS_MESH_WVL_FACTOR * mesh_wvl_um,
            high=SPHERE_MAX_RADIUS_MESH_WVL_FACTOR * mesh_wvl_um,
            size=1,
        ),
    )

    ring_init_mesh_wvl_factor = [0.15, 0.30, 0.2]
    ring_init = [r * mesh_wvl_um for r in ring_init_mesh_wvl_factor]

    geom_init = sphere_init + ring_init

    test_results = np.zeros((2, len(geom_init)))

    obj, adj_grad = obj_val_and_grad([geom_init])
    adj_grad = np.squeeze(np.array(adj_grad))

    # empirical step size for finite difference calculation
    fd_step = FD_STEP_MESH_WVL_FACTOR * mesh_wvl_um

    all_params = []

    for fd_idx in range(len(geom_init)):
        geom_up = geom_init.copy()
        geom_down = geom_init.copy()

        geom_up[fd_idx] += fd_step
        geom_down[fd_idx] -= fd_step

        all_params.append(geom_up)
        all_params.append(geom_down)

    all_obj = objective(all_params)

    fd_grad = np.zeros(len(geom_init))
    for fd_idx in range(len(geom_init)):
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
    print(f"RMS Error: {rms_error}")
    print(f"Gradient overlap (deg): {overlap_deg}")
    print(f"FD, Adj magnitudes: {fd_mag}, {adj_mag}")
    print("-" * 20)
    print("\n" * 3)

    assert overlap_deg < OVERLAP_ERROR_THRESHOLD_DEG * fd_mag, (
        "Adjoint and finite difference gradients misaligned."
    )

    test_results[SAVE_FD_LOC, :] = fd_grad
    test_results[SAVE_ADJ_LOC, :] = adj_grad

    test_number += 1

    if PLOT_FD_ADJ_COMPARISON:
        plt.plot(adj_grad, color="g", linewidth=2.0)
        plt.plot(fd_grad, color="b", linewidth=1.5, linestyle="--")
        plt.legend(["Adjoint", "Finite difference"])
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.show()

    if SAVE_FD_ADJ_DATA:
        np.save(f"{NUMERICAL_RESULTS_DATA_DIR}/results_{test_number}.npy", test_results)
