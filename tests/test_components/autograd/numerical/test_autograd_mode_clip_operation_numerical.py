"""Numerical autograd checks for clip-operation geometries in mode simulations."""

from __future__ import annotations

import autograd as ag
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import tidy3d as td
import tidy3d.web as web

pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

LOCAL_GRADIENT = True
VERBOSE = False
MESH_FACTOR_DESIGN = 60.0
NUM_MODE_MONITOR_FREQUENCIES = 4
NUM_FINITE_DIFFERENCE = 3
RMS_THRESHOLD = 0.25

MODE_LAYER_HEIGHT_WVL = 0.25
POLYSLAB_HEIGHT_WVL = MODE_LAYER_HEIGHT_WVL / 8.0
WG_WIDTH_WVL = 0.275
SUBSTRATE_INDEX = 1.5
WG_INDEX = 3.5
POLYSLAB_INDEX = 2.6

NUM_VERTICES_PER_POLY = 8


def get_sim_geometry(mesh_wvl_um: float) -> td.Box:
    return td.Box(size=(7 * mesh_wvl_um, 7 * mesh_wvl_um, 3 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um: float,
    adj_wvl_um: float,
    geometry_size_wvl: tuple[float, float, float],
    box_for_override: td.Box,
    run_time: float = 3e-11,
) -> td.Simulation:
    """Create the base simulation used by all clip-operation objective evaluations."""
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

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um
    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    wg_input_left = -0.75 * sim_size_um[0]
    wg_input_right = sim_center_um[0] - 0.5 * geometry_size_wvl[0] * mesh_wvl_um
    wg_output_left = sim_center_um[0] + 0.5 * geometry_size_wvl[0] * mesh_wvl_um
    wg_output_right = 0.75 * sim_size_um[0]

    wg_input_center = 0.5 * (wg_input_left + wg_input_right)
    wg_output_center = 0.5 * (wg_output_left + wg_output_right)
    wg_input_length = wg_input_right - wg_input_left
    wg_output_length = wg_output_right - wg_output_left

    src_input_center = 0.5 * (-0.5 * sim_size_um[0] + wg_input_right)
    monitor_output_center = 0.5 * (0.5 * sim_size_um[0] + wg_output_left)
    output_wg_y_offset_um = 0.5 * mesh_wvl_um

    mode_layer_height_um = MODE_LAYER_HEIGHT_WVL * adj_wvl_um
    input_waveguide = td.Structure(
        geometry=td.Box(
            center=(wg_input_center, 0, 0.5 * mode_layer_height_um),
            size=(wg_input_length, WG_WIDTH_WVL * adj_wvl_um, mode_layer_height_um),
        ),
        medium=td.Medium(permittivity=WG_INDEX**2),
    )
    output_waveguide = td.Structure(
        geometry=td.Box(
            center=(wg_output_center, output_wg_y_offset_um, 0.5 * mode_layer_height_um),
            size=(wg_output_length, WG_WIDTH_WVL * adj_wvl_um, mode_layer_height_um),
        ),
        medium=td.Medium(permittivity=WG_INDEX**2),
    )
    output_waveguide2 = td.Structure(
        geometry=td.Box(
            center=(wg_output_center, -output_wg_y_offset_um, 0.5 * mode_layer_height_um),
            size=(wg_output_length, WG_WIDTH_WVL * adj_wvl_um, mode_layer_height_um),
        ),
        medium=td.Medium(permittivity=WG_INDEX**2),
    )

    substrate_max = 0
    substrate_min = -0.75 * sim_size_um[2]
    substrate = td.Structure(
        geometry=td.Box(
            center=(sim_center_um[0], sim_center_um[1], 0.5 * (substrate_max + substrate_min)),
            size=(1.5 * sim_size_um[0], 1.5 * sim_size_um[1], substrate_max - substrate_min),
        ),
        medium=td.Medium(permittivity=SUBSTRATE_INDEX**2),
    )

    mode_monitor_freqs = np.linspace(0.9 * freq0, 1.05 * freq0, NUM_MODE_MONITOR_FREQUENCIES)
    mode_monitor_top = td.ModeMonitor(
        center=(
            monitor_output_center + 0.15 * mesh_wvl_um,
            output_wg_y_offset_um,
            0.5 * mode_layer_height_um,
        ),
        size=(0, 5 * WG_WIDTH_WVL * mesh_wvl_um, 5 * mode_layer_height_um),
        name="monitor_mode_top",
        freqs=mode_monitor_freqs,
    )
    mode_monitor_bottom = td.ModeMonitor(
        center=(monitor_output_center, -output_wg_y_offset_um, 0.5 * mode_layer_height_um),
        size=(0, 5 * WG_WIDTH_WVL * mesh_wvl_um, 5 * mode_layer_height_um),
        name="monitor_mode_bottom",
        freqs=mode_monitor_freqs,
    )

    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)
    mode_src = td.ModeSource(
        center=(src_input_center, 0, 0.5 * mode_layer_height_um),
        size=(0, 5 * WG_WIDTH_WVL * mesh_wvl_um, 5 * mode_layer_height_um),
        name="src_mode",
        source_time=pulse,
        direction="+",
    )

    return td.Simulation(
        center=sim_center_um,
        size=sim_size_um,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20,
            wavelength=mesh_wvl_um,
            override_structures=mesh_overrides,
        ),
        structures=[input_waveguide, output_waveguide, output_waveguide2, substrate],
        sources=[mode_src],
        monitors=[mode_monitor_top, mode_monitor_bottom],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )


def _make_two_polyslabs_from_vertices(
    vertices: np.ndarray,
    mode_layer_height_um: float,
    polyslab_height_um: float,
) -> tuple[td.PolySlab, td.PolySlab]:
    n = NUM_VERTICES_PER_POLY
    vertices_1_x = vertices[0:n]
    vertices_1_y = vertices[n : 2 * n]
    vertices_2_x = vertices[2 * n : 3 * n]
    vertices_2_y = vertices[3 * n : 4 * n]

    polyslab_1 = td.PolySlab(
        slab_bounds=(
            0.5 * mode_layer_height_um - 0.5 * polyslab_height_um,
            0.5 * mode_layer_height_um + 0.5 * polyslab_height_um,
        ),
        axis=2,
        vertices=tuple(zip(vertices_1_x, vertices_1_y)),
    )
    polyslab_2 = td.PolySlab(
        slab_bounds=(
            0.5 * mode_layer_height_um - 0.5 * polyslab_height_um,
            0.5 * mode_layer_height_um + 0.5 * polyslab_height_um,
        ),
        axis=2,
        vertices=tuple(zip(vertices_2_x, vertices_2_y)),
    )
    return polyslab_1, polyslab_2


def _make_three_polyslabs_from_vertices(
    vertices: np.ndarray,
    mode_layer_height_um: float,
    polyslab_height_um: float,
) -> tuple[td.PolySlab, td.PolySlab, td.PolySlab]:
    n = NUM_VERTICES_PER_POLY
    vertices_1_x = vertices[0:n]
    vertices_1_y = vertices[n : 2 * n]
    vertices_2_x = vertices[2 * n : 3 * n]
    vertices_2_y = vertices[3 * n : 4 * n]
    vertices_3_x = vertices[4 * n : 5 * n]
    vertices_3_y = vertices[5 * n : 6 * n]

    def make_poly(vx, vy):
        return td.PolySlab(
            slab_bounds=(
                0.5 * mode_layer_height_um - 0.5 * polyslab_height_um,
                0.5 * mode_layer_height_um + 0.5 * polyslab_height_um,
            ),
            axis=2,
            vertices=tuple(zip(vx, vy)),
        )

    return (
        make_poly(vertices_1_x, vertices_1_y),
        make_poly(vertices_2_x, vertices_2_y),
        make_poly(vertices_3_x, vertices_3_y),
    )


def create_objective_function(
    create_sim_base,
    eval_fn,
    sim_path_dir: str,
    mode_layer_height_um: float,
    polyslab_height_um: float,
    polyslab_permittivity: float,
    clip_operation: str,
):
    """Create objective for finite-difference/adjoint comparison with clipped polyslabs."""

    def objective(vertex_batches):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx, vertex_set in enumerate(vertex_batches):
            vertices = np.asarray(vertex_set)
            polyslab_1, polyslab_2 = _make_two_polyslabs_from_vertices(
                vertices=vertices,
                mode_layer_height_um=mode_layer_height_um,
                polyslab_height_um=polyslab_height_um,
            )

            clipped_geometry = td.ClipOperation(
                operation=clip_operation,
                geometry_a=polyslab_1,
                geometry_b=polyslab_2,
            )
            clipped_structure = td.Structure(
                geometry=clipped_geometry,
                medium=td.Medium(permittivity=polyslab_permittivity),
            )

            simulation_dict[f"numerical_mode_clip_operation_{clip_operation}_{idx}"] = (
                sim_base.updated_copy(structures=(*sim_base.structures, clipped_structure)).copy()
            )

        sim_data = web.run_async(
            simulation_dict,
            path_dir=sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
            verbose=VERBOSE,
        )

        objective_vals = []
        for idx in range(len(vertex_batches)):
            objective_vals.append(
                eval_fn(sim_data[f"numerical_mode_clip_operation_{clip_operation}_{idx}"])
            )

        if len(vertex_batches) == 1:
            return objective_vals[0]
        return objective_vals

    return objective


def _initial_vertices(mesh_wvl_um: float, geometry_case: int = 0) -> np.ndarray:
    """Create overlapping baseline polygons for the two clipped polyslabs."""
    angles = np.linspace(0, 2 * np.pi, NUM_VERTICES_PER_POLY, endpoint=False)

    if geometry_case == 0:
        vertices_1_x = 0.95 * mesh_wvl_um * np.cos(angles) - 0.15 * mesh_wvl_um
        vertices_1_y = 0.70 * mesh_wvl_um * np.sin(angles) + 0.05 * mesh_wvl_um
        vertices_2_x = 0.90 * mesh_wvl_um * np.cos(angles) + 0.25 * mesh_wvl_um
        vertices_2_y = 0.65 * mesh_wvl_um * np.sin(angles) + 0.10 * mesh_wvl_um
    elif geometry_case == 1:
        vertices_1_x = 0.85 * mesh_wvl_um * np.cos(angles) - 0.30 * mesh_wvl_um
        vertices_1_y = 0.75 * mesh_wvl_um * np.sin(angles) - 0.05 * mesh_wvl_um
        vertices_2_x = 0.80 * mesh_wvl_um * np.cos(angles) + 0.10 * mesh_wvl_um
        vertices_2_y = 0.60 * mesh_wvl_um * np.sin(angles) + 0.15 * mesh_wvl_um
    else:
        raise ValueError(f"Unsupported geometry_case={geometry_case}.")

    return np.concatenate((vertices_1_x, vertices_1_y, vertices_2_x, vertices_2_y))


def _initial_vertices_nested(mesh_wvl_um: float, geometry_case: int = 0) -> np.ndarray:
    """Create overlapping baseline polygons for three traced polyslabs."""
    angles = np.linspace(0, 2 * np.pi, NUM_VERTICES_PER_POLY, endpoint=False)

    if geometry_case == 0:
        vertices_1_x = 0.95 * mesh_wvl_um * np.cos(angles) - 0.25 * mesh_wvl_um
        vertices_1_y = 0.70 * mesh_wvl_um * np.sin(angles) + 0.00 * mesh_wvl_um
        vertices_2_x = 0.85 * mesh_wvl_um * np.cos(angles) + 0.15 * mesh_wvl_um
        vertices_2_y = 0.65 * mesh_wvl_um * np.sin(angles) + 0.12 * mesh_wvl_um
        vertices_3_x = 0.80 * mesh_wvl_um * np.cos(angles) + 0.05 * mesh_wvl_um
        vertices_3_y = 0.60 * mesh_wvl_um * np.sin(angles) - 0.16 * mesh_wvl_um
    elif geometry_case == 1:
        vertices_1_x = 0.90 * mesh_wvl_um * np.cos(angles) - 0.35 * mesh_wvl_um
        vertices_1_y = 0.75 * mesh_wvl_um * np.sin(angles) + 0.03 * mesh_wvl_um
        vertices_2_x = 0.88 * mesh_wvl_um * np.cos(angles) + 0.06 * mesh_wvl_um
        vertices_2_y = 0.62 * mesh_wvl_um * np.sin(angles) + 0.20 * mesh_wvl_um
        vertices_3_x = 0.78 * mesh_wvl_um * np.cos(angles) + 0.25 * mesh_wvl_um
        vertices_3_y = 0.58 * mesh_wvl_um * np.sin(angles) - 0.11 * mesh_wvl_um
    else:
        raise ValueError(f"Unsupported geometry_case={geometry_case}.")

    return np.concatenate(
        (
            vertices_1_x,
            vertices_1_y,
            vertices_2_x,
            vertices_2_y,
            vertices_3_x,
            vertices_3_y,
        )
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("clip_operation", "geometry_case"),
    [
        ("union", 0),
        ("union", 1),
        ("difference", 0),
        ("difference", 1),
        ("intersection", 0),
        ("intersection", 1),
        ("symmetric_difference", 0),
        ("symmetric_difference", 1),
    ],
)
def test_finite_difference_mode_data_clip_operation(
    clip_operation, geometry_case, rng, numerical_case_dir, redirect_stdout_to_stderr
):
    """Compare autograd and finite-difference directional derivatives for clip operations."""
    mesh_wvl_um = 1.55
    adj_wvl_um = 1.5
    geometry_size_wvl = (3.0, 3.0, MODE_LAYER_HEIGHT_WVL)
    polyslab_permittivity = POLYSLAB_INDEX**2

    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(np.inf, np.inf, MODE_LAYER_HEIGHT_WVL * mesh_wvl_um + mesh_wvl_um),
    )
    sim_path_dir = numerical_case_dir / "simulations" / f"clip_{clip_operation}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    monitor_top_weights = rng.random(NUM_MODE_MONITOR_FREQUENCIES)
    monitor_bottom_weights = rng.random(NUM_MODE_MONITOR_FREQUENCIES)
    frequency_selection_mask = np.arange(0, NUM_MODE_MONITOR_FREQUENCIES)

    def eval_fn(sim_data):
        return np.sum(
            monitor_top_weights
            * np.abs(
                sim_data["monitor_mode_top"]
                .amps.sel(direction="+")
                .isel(f=frequency_selection_mask)
                .data
            )
            ** 2
        ) + np.sum(
            monitor_bottom_weights
            * np.abs(
                sim_data["monitor_mode_bottom"]
                .amps.sel(direction="+")
                .isel(f=frequency_selection_mask)
                .data
            )
            ** 2
        )

    polyslab_height_um = POLYSLAB_HEIGHT_WVL * adj_wvl_um
    objective = create_objective_function(
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        geometry_size_wvl=geometry_size_wvl,
        box_for_override=box_for_override: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            geometry_size_wvl=geometry_size_wvl,
            box_for_override=box_for_override,
        ),
        eval_fn,
        sim_path_dir=str(sim_path_dir),
        mode_layer_height_um=MODE_LAYER_HEIGHT_WVL * mesh_wvl_um,
        polyslab_height_um=polyslab_height_um,
        polyslab_permittivity=polyslab_permittivity,
        clip_operation=clip_operation,
    )

    obj_val_and_grad = ag.value_and_grad(objective)
    fd_step = 0.2 * adj_wvl_um

    vertices0 = _initial_vertices(mesh_wvl_um=mesh_wvl_um, geometry_case=geometry_case)
    _, adj_grad = obj_val_and_grad([vertices0.tolist()])
    adj_grad = np.asarray(adj_grad).reshape(-1)
    n = NUM_VERTICES_PER_POLY
    gradients_vertices_1_x = adj_grad[0:n]
    gradients_vertices_1_y = adj_grad[n : 2 * n]
    gradients_vertices_2_x = adj_grad[2 * n : 3 * n]
    gradients_vertices_2_y = adj_grad[3 * n : 4 * n]

    pattern_dot_adj_gradient = np.zeros(NUM_FINITE_DIFFERENCE)
    all_vertex = []
    for fd_idx in range(NUM_FINITE_DIFFERENCE):
        random_pattern = rng.random(adj_grad.size) - 0.5
        random_pattern = gaussian_filter(random_pattern, sigma=1)
        random_pattern /= np.linalg.norm(random_pattern)

        pattern_dot_adj_gradient[fd_idx] = np.sum(random_pattern * adj_grad)

        all_vertex.append((vertices0 + random_pattern * fd_step).tolist())
        all_vertex.append((vertices0 - random_pattern * fd_step).tolist())

    all_obj = objective(all_vertex)
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

    print("\n" + "-" * 20)
    print(f"Clip operation: {clip_operation} (case {geometry_case})")
    print(f"Autograd gradients (poly1 x): {gradients_vertices_1_x}")
    print(f"Autograd gradients (poly1 y): {gradients_vertices_1_y}")
    print(f"Autograd gradients (poly2 x): {gradients_vertices_2_x}")
    print(f"Autograd gradients (poly2 y): {gradients_vertices_2_y}")
    print(f"Finite difference directional gradients: {fd_grad}")
    print(f"Autograd directional gradients: {pattern_dot_adj_gradient}")
    print(f"RMS Error: {rms_error}")
    print(f"FD, Adj magnitudes: {fd_mag}, {adj_mag}")
    print(f"Percentage Error: {percentage_error}")
    print("-" * 20 + "\n")

    assert rms_error < RMS_THRESHOLD * fd_mag, "RMS error magnitude too large"


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("inner_clip_operation", "outer_clip_operation", "geometry_case"),
    [
        ("union", "intersection", 0),
        ("difference", "symmetric_difference", 1),
    ],
)
def test_finite_difference_mode_data_nested_clip_operation(
    inner_clip_operation,
    outer_clip_operation,
    geometry_case,
    rng,
    numerical_case_dir,
    redirect_stdout_to_stderr,
):
    """Print directional and per-vertex gradients for nested clip operations."""
    mesh_wvl_um = 1.55
    adj_wvl_um = 1.5
    geometry_size_wvl = (3.0, 3.0, MODE_LAYER_HEIGHT_WVL)
    polyslab_permittivity = POLYSLAB_INDEX**2

    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(np.inf, np.inf, MODE_LAYER_HEIGHT_WVL * mesh_wvl_um + mesh_wvl_um),
    )
    sim_path_dir = (
        numerical_case_dir
        / "simulations"
        / f"nested_clip_{inner_clip_operation}_{outer_clip_operation}"
    )
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    monitor_top_weights = rng.random(NUM_MODE_MONITOR_FREQUENCIES)
    monitor_bottom_weights = rng.random(NUM_MODE_MONITOR_FREQUENCIES)
    frequency_selection_mask = np.arange(0, NUM_MODE_MONITOR_FREQUENCIES)

    def eval_fn(sim_data):
        return np.sum(
            monitor_top_weights
            * np.abs(
                sim_data["monitor_mode_top"]
                .amps.sel(direction="+")
                .isel(f=frequency_selection_mask)
                .data
            )
            ** 2
        ) + np.sum(
            monitor_bottom_weights
            * np.abs(
                sim_data["monitor_mode_bottom"]
                .amps.sel(direction="+")
                .isel(f=frequency_selection_mask)
                .data
            )
            ** 2
        )

    polyslab_height_um = POLYSLAB_HEIGHT_WVL * adj_wvl_um

    def objective(vertex_batches):
        sim_base = make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            geometry_size_wvl=geometry_size_wvl,
            box_for_override=box_for_override,
        )

        simulation_dict = {}
        for idx, vertex_set in enumerate(vertex_batches):
            vertices = np.asarray(vertex_set)
            poly_1, poly_2, poly_3 = _make_three_polyslabs_from_vertices(
                vertices=vertices,
                mode_layer_height_um=MODE_LAYER_HEIGHT_WVL * mesh_wvl_um,
                polyslab_height_um=polyslab_height_um,
            )

            inner_clip = td.ClipOperation(
                operation=inner_clip_operation,
                geometry_a=poly_1,
                geometry_b=poly_2,
            )
            nested_clip = td.ClipOperation(
                operation=outer_clip_operation,
                geometry_a=inner_clip,
                geometry_b=poly_3,
            )
            clipped_structure = td.Structure(
                geometry=nested_clip,
                medium=td.Medium(permittivity=polyslab_permittivity),
            )
            simulation_dict[f"numerical_mode_nested_clip_{idx}"] = sim_base.updated_copy(
                structures=(*sim_base.structures, clipped_structure)
            )

        sim_data = web.run_async(
            simulation_dict,
            path_dir=str(sim_path_dir),
            local_gradient=LOCAL_GRADIENT,
            verbose=VERBOSE,
        )

        objective_vals = [
            eval_fn(sim_data[f"numerical_mode_nested_clip_{idx}"])
            for idx in range(len(vertex_batches))
        ]
        if len(vertex_batches) == 1:
            return objective_vals[0]
        return objective_vals

    obj_val_and_grad = ag.value_and_grad(objective)
    fd_step = 0.2 * adj_wvl_um

    vertices0 = _initial_vertices_nested(mesh_wvl_um=mesh_wvl_um, geometry_case=geometry_case)
    _, adj_grad = obj_val_and_grad([vertices0.tolist()])
    adj_grad = np.asarray(adj_grad).reshape(-1)

    n = NUM_VERTICES_PER_POLY
    gradients_vertices_1_x = adj_grad[0:n]
    gradients_vertices_1_y = adj_grad[n : 2 * n]
    gradients_vertices_2_x = adj_grad[2 * n : 3 * n]
    gradients_vertices_2_y = adj_grad[3 * n : 4 * n]
    gradients_vertices_3_x = adj_grad[4 * n : 5 * n]
    gradients_vertices_3_y = adj_grad[5 * n : 6 * n]

    pattern_dot_adj_gradient = np.zeros(NUM_FINITE_DIFFERENCE)
    all_vertex = []
    for fd_idx in range(NUM_FINITE_DIFFERENCE):
        random_pattern = rng.random(adj_grad.size) - 0.5
        random_pattern = gaussian_filter(random_pattern, sigma=1)
        random_pattern /= np.linalg.norm(random_pattern)
        pattern_dot_adj_gradient[fd_idx] = np.sum(random_pattern * adj_grad)
        all_vertex.append((vertices0 + random_pattern * fd_step).tolist())
        all_vertex.append((vertices0 - random_pattern * fd_step).tolist())

    all_obj = objective(all_vertex)
    fd_grad = np.zeros(NUM_FINITE_DIFFERENCE)
    for fd_idx in range(NUM_FINITE_DIFFERENCE):
        fd_grad[fd_idx] = (all_obj[2 * fd_idx] - all_obj[2 * fd_idx + 1]) / (2 * fd_step)

    rms_error = np.linalg.norm(fd_grad - pattern_dot_adj_gradient)
    fd_mag = np.linalg.norm(fd_grad)
    adj_mag = np.linalg.norm(pattern_dot_adj_gradient)
    percentage_error = 100.0 * np.mean(
        np.abs(fd_grad - pattern_dot_adj_gradient) / (np.abs(fd_grad) + np.finfo(np.float64).eps)
    )

    print("\n" + "-" * 20)
    print(
        f"Nested clip operation: inner={inner_clip_operation}, outer={outer_clip_operation} (case {geometry_case})"
    )
    print(f"Autograd gradients (poly1 x): {gradients_vertices_1_x}")
    print(f"Autograd gradients (poly1 y): {gradients_vertices_1_y}")
    print(f"Autograd gradients (poly2 x): {gradients_vertices_2_x}")
    print(f"Autograd gradients (poly2 y): {gradients_vertices_2_y}")
    print(f"Autograd gradients (poly3 x): {gradients_vertices_3_x}")
    print(f"Autograd gradients (poly3 y): {gradients_vertices_3_y}")
    print(f"Finite difference directional gradients: {fd_grad}")
    print(f"Autograd directional gradients: {pattern_dot_adj_gradient}")
    print(f"RMS Error: {rms_error}")
    print(f"FD, Adj magnitudes: {fd_mag}, {adj_mag}")
    print(f"Percentage Error: {percentage_error}")
    print("-" * 20 + "\n")

    assert rms_error < RMS_THRESHOLD * fd_mag, "RMS error magnitude too large"
