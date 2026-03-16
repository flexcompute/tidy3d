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
NUMERICAL_RESULTS_SUBDIR = "numerical_gaussian_test"

RMS_THRESHOLD = 0.25

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


FINITE_DIFF_PERM_SEED = 2.5**2
MESH_FACTOR_DESIGN = 30.0


def get_sim_geometry(mesh_wvl_um):
    return td.Box(size=(5 * mesh_wvl_um, 5 * mesh_wvl_um, 7 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    box_for_override,
    run_time=1e-11,
    pol_angle=np.pi / 2,
    angle_theta=0.0,
    angle_phi=0.0,
    monitor_side="transmission",
    monitor_type="gaussian",
    waist_distance_offset=0.0,
    monitor_freqs=None,
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

    mesh_overrides = [
        td.MeshOverrideStructure(
            geometry=box_for_override,
            dl=[dl_design, dl_design, dl_design],
        ),
    ]

    src_size = sim_size_um[0:2] + (0,)

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um

    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um
    if monitor_freqs is None:
        monitor_freqs = [freq0]

    src_z_pos = -2 * mesh_wvl_um
    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)
    src = td.GaussianBeam(
        center=(0, 0, src_z_pos),
        size=src_size,
        source_time=pulse,
        direction="+",
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        pol_angle=pol_angle,
        waist_radius=1.75 * mesh_wvl_um,
        waist_distance=0.0,
    )

    if monitor_side == "reflection":
        sim_z_min = sim_center_um[2] - 0.5 * sim_size_um[2]
        monitor_z_pos = 0.5 * (src_z_pos + sim_z_min)
    else:
        monitor_z_pos = 0.25 * sim_size_um[2]

    monitor_kwargs = {
        "center": (0, 0, monitor_z_pos),
        "size": (*sim_size_um[0:2], 0),
        "name": "monitor_overlap",
        "freqs": monitor_freqs,
        "pol_angle": pol_angle,
        "angle_theta": angle_theta,
        "angle_phi": angle_phi,
    }
    if monitor_type == "gaussian":
        overlap_monitor = td.GaussianOverlapMonitor(
            waist_radius=1.5 * mesh_wvl_um,
            waist_distance=waist_distance_offset * mesh_wvl_um,
            **monitor_kwargs,
        )
    elif monitor_type == "astigmatic":
        overlap_monitor = td.AstigmaticGaussianOverlapMonitor(
            waist_sizes=(1.5 * mesh_wvl_um, 1.0 * mesh_wvl_um),
            waist_distances=(
                waist_distance_offset * mesh_wvl_um,
                (waist_distance_offset + 0.2) * mesh_wvl_um,
            ),
            **monitor_kwargs,
        )
    else:
        raise ValueError(f"Unsupported monitor_type='{monitor_type}'.")

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
        monitors=[overlap_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_objective_function(geometry, create_sim_base, eval_fn, sim_path_dir, perm_init):
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

            sim_with_block = sim_base.updated_copy(
                structures=(*sim_base.structures, block_structure),
                grid_spec=td.GridSpec.from_grid(grid_fixed),
            )

            simulation_dict[f"numerical_gaussian_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(perm_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_gaussian_testing_{idx}"]))

        if len(perm_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def make_eval_fns(objective_direction="+"):
    def gaussian_overlap_power(sim_data):
        amps = sim_data["monitor_overlap"].amps.sel(direction=objective_direction)
        return np.sum(np.abs(amps.values) ** 2)

    return [gaussian_overlap_power], ["gaussian_overlap_power"]


def random_monitor_freqs(rng, freq0):
    """Create reproducible random monitor frequencies between 0.9*freq0 and 1.1*freq0."""
    num_freqs = int(rng.integers(1, 6))
    freqs = rng.uniform(0.9 * freq0, 1.1 * freq0, size=num_freqs)
    return np.sort(freqs).tolist()


mesh_wvls_um = [1.55]
adj_wvls_um = [1.55]
angle_test_cases = [
    {"pol_angle": 0.0, "angle_theta": 0.0, "angle_phi": 0.0},
    {"pol_angle": np.pi / 2, "angle_theta": 0.0, "angle_phi": 0.0},
    {"pol_angle": 0.0, "angle_theta": np.deg2rad(20), "angle_phi": 0.0},
    {"pol_angle": np.pi / 2, "angle_theta": np.deg2rad(20), "angle_phi": np.deg2rad(45)},
]
monitor_side_cases = [
    {"monitor_side": "transmission", "objective_direction": "+"},
    {"monitor_side": "reflection", "objective_direction": "-"},
]
monitor_type_cases = [
    {"monitor_type": "gaussian"},
    {"monitor_type": "astigmatic"},
]
waist_distance_offset_cases = [-1.0, 0.0, 1.0]

field_data_test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for monitor_type_case in monitor_type_cases:
        for monitor_side_case in monitor_side_cases:
            eval_fns, eval_fn_names = make_eval_fns(
                objective_direction=monitor_side_case["objective_direction"]
            )
            for eval_fn_idx, eval_fn in enumerate(eval_fns):
                for angle_case in angle_test_cases:
                    for waist_distance_offset in waist_distance_offset_cases:
                        field_data_test_parameters.append(
                            {
                                "mesh_wvl_um": mesh_wvl_um,
                                "adj_wvl_um": adj_wvl_um,
                                "monitor_type": monitor_type_case["monitor_type"],
                                "eval_fn": eval_fn,
                                "eval_fn_name": eval_fn_names[eval_fn_idx],
                                "monitor_side": monitor_side_case["monitor_side"],
                                "objective_direction": monitor_side_case["objective_direction"],
                                "pol_angle": angle_case["pol_angle"],
                                "angle_theta": angle_case["angle_theta"],
                                "angle_phi": angle_case["angle_phi"],
                                "waist_distance_offset": waist_distance_offset,
                                "test_number": test_number,
                            }
                        )

                        test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize("field_data_test_parameters", field_data_test_parameters)
def test_finite_difference_gaussian_overlap_data(
    field_data_test_parameters, rng, numerical_case_dir, redirect_stdout_to_stderr
):
    """Compare autograd permittivity gradients against finite-difference for Gaussian overlap power."""

    test_results = np.zeros((2, NUM_FINITE_DIFFERENCE))

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_type,
        eval_fn,
        eval_fn_name,
        monitor_side,
        objective_direction,
        pol_angle,
        angle_theta,
        angle_phi,
        waist_distance_offset,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_type",
        "eval_fn",
        "eval_fn_name",
        "monitor_side",
        "objective_direction",
        "pol_angle",
        "angle_theta",
        "angle_phi",
        "waist_distance_offset",
        "test_number",
    )(field_data_test_parameters)

    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(center=(0, 0, 0), size=(dim_um, dim_um, thickness_um))

    dim = 1 + int(dim_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

    sim_geometry = get_sim_geometry(mesh_wvl_um)
    box_for_override = td.Box(
        center=(0, 0, 0), size=sim_geometry.size[0:2] + (thickness_um + mesh_wvl_um,)
    )

    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, nz))
    freq0 = td.C_0 / adj_wvl_um
    for _ in range(test_number):
        # spin the rng so we get different randomness for each test
        ignore_rng = rng.random(1)
    monitor_freqs = random_monitor_freqs(rng=rng, freq0=freq0)

    objective = create_objective_function(
        block,
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        box_for_override=box_for_override,
        pol_angle=pol_angle,
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        monitor_side=monitor_side,
        monitor_type=monitor_type,
        waist_distance_offset=waist_distance_offset: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            box_for_override=box_for_override,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            monitor_side=monitor_side,
            monitor_type=monitor_type,
            waist_distance_offset=waist_distance_offset,
            monitor_freqs=monitor_freqs,
        ),
        eval_fn,
        sim_path_dir=str(sim_path_dir),
        perm_init=perm_init,
    )

    obj_val_and_grad = ag.value_and_grad(objective)
    _, adj_grad = obj_val_and_grad([perm_init])

    fd_step = 0.1
    all_perm = []
    pattern_dot_adj_gradient = np.zeros(NUM_FINITE_DIFFERENCE)

    for fd_idx in range(NUM_FINITE_DIFFERENCE):
        random_pattern = rng.random((dim, dim, nz)) - 0.5
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
    print(f"Monitor type: {monitor_type}")
    print(f"Monitor side / objective direction: {monitor_side} / {objective_direction}")
    print(f"Angles (pol, theta, phi): {pol_angle}, {angle_theta}, {angle_phi}")
    print(f"Waist distance offset (wvl): {waist_distance_offset}")
    print(f"Monitor frequencies ({len(monitor_freqs)}): {monitor_freqs}")
    print(f"Eval function: {eval_fn_name}")
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
