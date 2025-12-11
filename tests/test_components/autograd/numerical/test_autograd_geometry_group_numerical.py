# test GeometryGroup gradient consistency when there are no overlapping structures
from __future__ import annotations

import operator

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web

PLOT_FD_ADJ_COMPARISON = True
SAVE_FD_ADJ_DATA = True
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
NUMERICAL_RESULTS_SUBDIR = "numerical_geometry_group_test"

NUM_MODE_MONITOR_FREQUENCIES = 4

RMS_THRESHOLD = 1e-5

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

MESH_FACTOR_DESIGN = 60.0


def angled_overlap_deg(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if np.isclose(norm_v1, 0.0) or np.isclose(norm_v2, 0.0):
        if not (np.isclose(norm_v1, 0.0) and np.isclose(norm_v2, 0.0)):
            return np.inf

        return 0.0

    dot = np.minimum(1.0, np.sum((v1 / np.linalg.norm(v1)) * (v2 / np.linalg.norm(v2))))
    angle_deg = np.arccos(dot) * 180.0 / np.pi

    return angle_deg


def get_sim_geometry(mesh_wvl_um, offset_y_size_wvl=0):
    return td.Box(size=(7 * mesh_wvl_um, 7 * mesh_wvl_um, 3 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    geometry_size_wvl,
    box_for_override,
    background_medium,
    run_time=1e-11,
):
    """Creates a base simulation with input/output waveguides, mode sources, and mode monitors."""
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
    input_waveguide_geometry = td.Box(
        center=(wg_input_center, 0, 0.5 * mode_layer_height_um),
        size=(wg_input_length, WG_WIDTH_WVL * adj_wvl_um, mode_layer_height_um),
    )
    output_waveguide_geometry = td.Box(
        center=(wg_output_center, output_wg_y_offset_um, 0.5 * mode_layer_height_um),
        size=(wg_output_length, WG_WIDTH_WVL * adj_wvl_um, mode_layer_height_um),
    )
    output_waveguide_geometry2 = td.Box(
        center=(wg_output_center, -output_wg_y_offset_um, 0.5 * mode_layer_height_um),
        size=(wg_output_length, WG_WIDTH_WVL * adj_wvl_um, mode_layer_height_um),
    )

    input_waveguide = td.Structure(
        geometry=input_waveguide_geometry, medium=td.Medium(permittivity=WG_INDEX**2)
    )
    output_waveguide = td.Structure(
        geometry=output_waveguide_geometry, medium=td.Medium(permittivity=WG_INDEX**2)
    )
    output_waveguide2 = td.Structure(
        geometry=output_waveguide_geometry2, medium=td.Medium(permittivity=WG_INDEX**2)
    )

    substrate_max = 0
    substrate_min = -0.75 * sim_size_um[2]
    substrate = td.Structure(
        geometry=td.Box(
            center=(sim_center_um[0], sim_center_um[1], 0.5 * (substrate_max + substrate_min)),
            size=(1.5 * sim_size_um[0], 1.5 * sim_size_um[1], (substrate_max - substrate_min)),
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

    sim_base = td.Simulation(
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
        medium=background_medium,
    )

    return sim_base


def create_objective_functions(
    create_sim_base,
    eval_fn,
    sim_path_dir,
    mode_layer_height_um,
    polyslab_height_um,
    polyslab_permittivity,
    embedded_permittivity,
):
    """Create an objective function to use for the test based on different base simulation creation,
    objective function, geometric, and optical parameters.."""

    num_polyslab_chunks = 3
    offset_angles = np.linspace(0, 2 * np.pi, num_polyslab_chunks, endpoint=False)
    vertices_per_chunk = NUM_VERTICES // num_polyslab_chunks

    def make_objective(geometry_group: bool):
        def objective(vertices):
            sim_base = create_sim_base()

            simulation_dict = {}
            for idx in range(len(vertices)):
                get_vertex_set = vertices[idx]

                vertices_x = np.array(get_vertex_set[0:NUM_VERTICES])
                vertices_y = np.array(get_vertex_set[NUM_VERTICES:])

                polyslab_geometries = []

                # break the single polyslab into multiple chunks that can either be inserted separately or
                # inside a GeometryGroup depending on the boolean flag geometry_group
                for chunk_idx in range(num_polyslab_chunks):
                    chunk_start = chunk_idx * vertices_per_chunk
                    chunk_end = np.minimum(chunk_start + vertices_per_chunk, NUM_VERTICES)

                    vertices_x_offset = vertices_x[chunk_start:chunk_end] + 0.2 * np.cos(
                        offset_angles[chunk_idx]
                    )
                    vertices_y_offset = vertices_y[chunk_start:chunk_end] + 0.2 * np.sin(
                        offset_angles[chunk_idx]
                    )

                    make_polyslab = td.PolySlab(
                        slab_bounds=(
                            0.5 * mode_layer_height_um - 0.5 * polyslab_height_um,
                            0.5 * mode_layer_height_um + 0.5 * polyslab_height_um,
                        ),
                        axis=2,
                        vertices=tuple(zip(vertices_x_offset, vertices_y_offset)),
                    )

                    polyslab_geometries.append(make_polyslab)

                group_geom = td.GeometryGroup(geometries=polyslab_geometries)
                group_geom_box = group_geom.bounding_box

                embedding_structure = td.Structure(
                    geometry=group_geom_box, medium=td.Medium(permittivity=embedded_permittivity)
                )

                if geometry_group:
                    polyslab_structures = [
                        td.Structure(
                            geometry=group_geom,
                            medium=td.Medium(permittivity=polyslab_permittivity),
                        )
                    ]
                else:
                    polyslab_structures = [
                        td.Structure(
                            geometry=g, medium=td.Medium(permittivity=polyslab_permittivity)
                        )
                        for g in polyslab_geometries
                    ]

                sim_with_polyslabs = sim_base.updated_copy(
                    structures=(*sim_base.structures, embedding_structure, *polyslab_structures)
                )

                simulation_dict[f"numerical_geometry_group_testing_{geometry_group}_{idx}"] = (
                    sim_with_polyslabs.copy()
                )

            sim_data = web.run_async(
                simulation_dict,
                path_dir=sim_path_dir,
                local_gradient=LOCAL_GRADIENT,
                verbose=VERBOSE,
            )

            objective_vals = []
            for idx in range(len(vertices)):
                objective_vals.append(
                    eval_fn(sim_data[f"numerical_geometry_group_testing_{geometry_group}_{idx}"])
                )

            if len(vertices) == 1:
                return objective_vals[0]

            return objective_vals

        return objective

    return make_objective(geometry_group=False), make_objective(geometry_group=True)


# Parameters for controlling the test geometry and material parameters as well as the
# array of tests to run.
MODE_LAYER_HEIGHT_WVL = 0.25
POLYSLAB_HEIGHT_WVL = MODE_LAYER_HEIGHT_WVL / 8.0
WG_WIDTH_WVL = 0.275
GEOMETRY_SIZE_WVL = 4.0
SUBSTRATE_INDEX = 1.5
WG_INDEX = 3.5

FINITE_DIFF_PERM_SEED = 0.5 * (1.0**2 + WG_INDEX**2)

# Number of vertices to put in the test polyslab.
NUM_VERTICES = 15
MESH_WVL_UM = 1.55
ADJ_WVL_UM = 1.5
POLYSLAB_INDEX = 2.5

embedded_permittivities = [1.0**2, 1.5**2]
simulation_background_permittivites = [1.0**2, 1.75**2]

geometry_group_test_parameters = []

test_number = 0
for embedded_permittivity in embedded_permittivities:
    for simulation_background_permittivity in simulation_background_permittivites:
        geometry_group_test_parameters.append(
            {
                "embedded_permittivity": embedded_permittivity,
                "simulation_background_permittivity": simulation_background_permittivity,
                "test_number": test_number,
            }
        )

        test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize("geometry_group_test_parameters", geometry_group_test_parameters)
def test_finite_difference_mode_data_polyslab(
    geometry_group_test_parameters, rng, numerical_case_dir, redirect_stdout_to_stderr
):
    """Test that GeometryGroup gradients are consistent with not using a GeometryGroup when there
    are no overlapping structures."""

    test_number = geometry_group_test_parameters["test_number"]

    (
        embedded_permittivity,
        simulation_background_permittivity,
        test_number,
    ) = operator.itemgetter(
        "embedded_permittivity",
        "simulation_background_permittivity",
        "test_number",
    )(geometry_group_test_parameters)

    adj_wvl_um = ADJ_WVL_UM
    mesh_wvl_um = MESH_WVL_UM
    geometry_size_wvl = (GEOMETRY_SIZE_WVL, GEOMETRY_SIZE_WVL, MODE_LAYER_HEIGHT_WVL)
    polyslab_permittivity = POLYSLAB_INDEX**2

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0),
        size=(np.inf, np.inf, MODE_LAYER_HEIGHT_WVL * mesh_wvl_um + mesh_wvl_um),
    )

    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    # Weights for creating a random objective function over multiple frequencies by
    # summing their contributions by random weights. This helps verify gradient errors
    # due to a multifrequency objective function.
    monitor_top_weights = rng.random(NUM_MODE_MONITOR_FREQUENCIES)
    monitor_bottom_weights = rng.random(NUM_MODE_MONITOR_FREQUENCIES)
    frequency_selection_mask = np.arange(0, NUM_MODE_MONITOR_FREQUENCIES)

    # sometimes, test what happens when we only use one of the frequencies from the mode monitors
    # to catch handling of different frequencies being present in the forward and adjoint monitors
    if rng.random() > 0.5:
        frequency_selection_mask = rng.integers(1, NUM_MODE_MONITOR_FREQUENCIES)
        monitor_top_weights = monitor_top_weights[frequency_selection_mask]
        monitor_bottom_weights = monitor_bottom_weights[frequency_selection_mask]

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

    objective_no_geom_group, objective_geom_group = create_objective_functions(
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        geometry_size_wvl=geometry_size_wvl,
        polyslab_permittivity=polyslab_permittivity,
        box_for_override=box_for_override: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            geometry_size_wvl=geometry_size_wvl,
            box_for_override=box_for_override,
            background_medium=td.Medium(permittivity=simulation_background_permittivity),
        ),
        eval_fn,
        sim_path_dir=str(sim_path_dir),
        mode_layer_height_um=MODE_LAYER_HEIGHT_WVL * mesh_wvl_um,
        polyslab_height_um=polyslab_height_um,
        polyslab_permittivity=polyslab_permittivity,
        embedded_permittivity=embedded_permittivity,
    )

    obj_val_and_grad_no_geom_group = ag.value_and_grad(objective_no_geom_group)
    obj_val_and_grad_geom_group = ag.value_and_grad(objective_geom_group)

    angles = np.linspace(0, 2 * np.pi, NUM_VERTICES + 1)[0:-1]
    vertex_centers_x = 1.1 * mesh_wvl_um * np.cos(angles)
    vertex_centers_y = 0.8 * mesh_wvl_um * np.sin(angles)

    obj_no_geom_group, adj_grad_no_geom_group = obj_val_and_grad_no_geom_group(
        [list(vertex_centers_x) + list(vertex_centers_y)]
    )
    obj_geom_group, adj_grad_geom_group = obj_val_and_grad_geom_group(
        [list(vertex_centers_x) + list(vertex_centers_y)]
    )

    adj_grad_no_geom_group = np.squeeze(np.array(adj_grad_no_geom_group))
    adj_grad_geom_group = np.squeeze(np.array(adj_grad_geom_group))

    rms_error = np.linalg.norm(adj_grad_no_geom_group - adj_grad_geom_group)
    no_geom_group_mag = np.linalg.norm(adj_grad_no_geom_group)
    geom_group_mag = np.linalg.norm(adj_grad_geom_group)

    overlap_deg = angled_overlap_deg(adj_grad_no_geom_group, adj_grad_geom_group)

    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{test_number}")
    print(f"Background permittivity: {simulation_background_permittivity}")
    print(f"Embedded permittivity: {embedded_permittivity}")
    print(f"RMS Error: {rms_error}")
    print(f"No Geom Group, Geom Group magnitudes: {no_geom_group_mag}, {geom_group_mag}")
    print(f"Overlap (deg): {overlap_deg}")
    print("-" * 20)
    print("\n" * 3)

    test_results = np.zeros((2, len(adj_grad_no_geom_group)))

    test_results[0, :] = adj_grad_no_geom_group
    test_results[1, :] = adj_grad_geom_group

    save_idx = test_number + 1
    save_path = None
    if SAVE_FD_ADJ_DATA:
        results_dir = numerical_case_dir / NUMERICAL_RESULTS_SUBDIR
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"results_{save_idx}.npy"

    try:
        assert rms_error < RMS_THRESHOLD * np.sqrt(no_geom_group_mag * geom_group_mag), (
            "RMS error magnitude too large"
        )
    finally:
        if save_path is not None:
            np.save(save_path, test_results)

    test_number += 1

    if PLOT_FD_ADJ_COMPARISON:
        plt.plot(adj_grad_no_geom_group, color="g", linewidth=2.0)
        plt.plot(adj_grad_geom_group, color="b", linewidth=1.5, linestyle="--")
        plt.title("Gradient:")
        plt.legend(["No Geom Group", "Geom Group"])
        plt.xlabel("Vertex")
        plt.ylabel("Gradient value")
        plt.show()
