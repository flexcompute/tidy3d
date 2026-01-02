# test autograd postprocess for frequency slicing to ensure gradients are consistent
from __future__ import annotations

import operator

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web
from tidy3d.config import config

PLOT_ADJ_COMPARISON = True  # False
NUM_FINITE_DIFFERENCE = 10
SAVE_FD_ADJ_DATA = False
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
NUMERICAL_RESULTS_SUBDIR = "numerical_test_frequency_selection"

if PLOT_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

MESH_FACTOR_DESIGN = 60.0


def get_sim_geometry(mesh_wvl_um, offset_y_size_wvl=0):
    return td.Box(size=(7 * mesh_wvl_um, 7 * mesh_wvl_um, 3 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    geometry_size_wvl,
    box_for_override,
    required_freqs,
    num_extra_freqs,
    rng,
    run_time=5e-11,
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

    min_required_freq = np.min(required_freqs)
    max_required_freq = np.max(required_freqs)

    required_freq_span = max_required_freq - min_required_freq

    random_other_freqs = rng.uniform(
        low=min_required_freq - 0.1 * required_freq_span,
        high=max_required_freq + 0.1 * required_freq_span,
        size=num_extra_freqs,
    )

    mode_monitor_freqs = sorted(list(required_freqs) + list(random_other_freqs))
    mode_monitor_freqs = np.flip(mode_monitor_freqs)
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
    )

    return sim_base


def create_objective_function(
    create_sim_base,
    eval_fn,
    sim_path_dir,
    mode_layer_height_um,
    polyslab_height_um,
    polyslab_permittivity,
):
    """Create an objective function for the test based on the base simulation, type of evaluation function on
    the electromagnetic data, and geometric and material parameters."""

    def objective(vertices):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(vertices)):
            vertices_x = vertices[idx][0:NUM_VERTICES]
            vertices_y = vertices[idx][NUM_VERTICES:]

            make_polyslab = td.PolySlab(
                slab_bounds=(
                    0.5 * mode_layer_height_um - 0.5 * polyslab_height_um,
                    0.5 * mode_layer_height_um + 0.5 * polyslab_height_um,
                ),
                axis=2,
                vertices=tuple(zip(vertices_x, vertices_y)),
            )

            polyslab_structure = td.Structure(
                geometry=make_polyslab, medium=td.Medium(permittivity=polyslab_permittivity)
            )

            sim_with_polyslab = sim_base.updated_copy(
                structures=(*sim_base.structures, polyslab_structure)
            )

            simulation_dict[f"numerical_mode_polyslab_testing_{idx}"] = sim_with_polyslab.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(vertices)):
            objective_vals.append(eval_fn(sim_data[f"numerical_mode_polyslab_testing_{idx}"]))

        if len(vertices) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


# Parameters for controlling the test geometry and material parameters as well as the
# array of tests to run.
MODE_LAYER_HEIGHT_WVL = 0.25
POLYSLAB_HEIGHT_WVL = MODE_LAYER_HEIGHT_WVL / 8.0
WG_WIDTH_WVL = 0.275
SUBSTRATE_INDEX = 1.5

WG_INDEX = 3.5

# Number of vertices to put in the test polyslab.
NUM_VERTICES = 15
NUM_EXTRA_FREQS = 10
NUM_OBJECTIVES = 5

mesh_wvls_um = [1.55]
adj_wvls_um = [1.5]
geometry_sizes_wvl = [(3.0, 3.0, MODE_LAYER_HEIGHT_WVL)]
polyslab_indices = np.linspace(SUBSTRATE_INDEX, WG_INDEX, 3)

mode_data_test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for geometry_size_wvl in geometry_sizes_wvl:
        for polyslab_index in polyslab_indices:
            polyslab_permittivity = polyslab_index**2

            mode_data_test_parameters.append(
                {
                    "mesh_wvl_um": mesh_wvl_um,
                    "adj_wvl_um": adj_wvl_um,
                    "geometry_size_wvl": geometry_size_wvl,
                    "polyslab_permittivity": polyslab_permittivity,
                    "test_number": test_number,
                }
            )

            test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize("mode_data_test_parameters", mode_data_test_parameters)
def test_finite_difference_mode_data_polyslab(
    mode_data_test_parameters, rng, monkeypatch, numerical_case_dir, redirect_stdout_to_stderr
):
    """Test a variety of frequency combinations in the forward monitor to ensure we get
    the same adjoint gradient when using a fixed set of frequencies in the objective function."""

    monkeypatch.setattr(config.adjoint, "solver_freq_chunk_size", 2)

    test_results = np.zeros((2, NUM_FINITE_DIFFERENCE))

    test_number = mode_data_test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        geometry_size_wvl,
        polyslab_permittivity,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "geometry_size_wvl",
        "polyslab_permittivity",
        "test_number",
    )(mode_data_test_parameters)

    adj_freq = td.C_0 / adj_wvl_um

    dim_x_um = geometry_size_wvl[0] * mesh_wvl_um * 2
    dim_y_um = geometry_size_wvl[1] * mesh_wvl_um * 2
    thickness_um = geometry_size_wvl[2] * mesh_wvl_um

    dim_x = 1 + int(dim_x_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    dim_y = 1 + int(dim_y_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    Nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

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
    monitor_top_weights = rng.random(2)
    monitor_bottom_weights = rng.random(3)

    def make_random_objective():
        required_freqs = [0.95 * adj_freq, 1.01 * adj_freq, 1.03 * adj_freq]

        def eval_fn(sim_data):
            return np.sum(
                monitor_top_weights
                * np.abs(
                    sim_data["monitor_mode_top"]
                    .amps.sel(direction="+")
                    .sel(f=required_freqs[0:2])
                    .data
                )
                ** 2
            ) + np.sum(
                monitor_bottom_weights
                * np.abs(
                    sim_data["monitor_mode_bottom"]
                    .amps.sel(direction="+")
                    .sel(f=required_freqs)
                    .data
                )
                ** 2
            )

        polyslab_height_um = POLYSLAB_HEIGHT_WVL * adj_wvl_um

        objective = create_objective_function(
            lambda mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            geometry_size_wvl=geometry_size_wvl,
            polyslab_permittivity=polyslab_permittivity,
            box_for_override=box_for_override,
            required_freqs=required_freqs,
            rng=rng: make_base_sim(
                mesh_wvl_um=mesh_wvl_um,
                adj_wvl_um=adj_wvl_um,
                geometry_size_wvl=geometry_size_wvl,
                box_for_override=box_for_override,
                required_freqs=required_freqs,
                num_extra_freqs=NUM_EXTRA_FREQS,
                rng=rng,
                run_time=2e-11,
            ),
            eval_fn,
            sim_path_dir=str(sim_path_dir),
            mode_layer_height_um=MODE_LAYER_HEIGHT_WVL * mesh_wvl_um,
            polyslab_height_um=polyslab_height_um,
            polyslab_permittivity=polyslab_permittivity,
        )

        return objective

    objectives = [make_random_objective() for idx in range(NUM_OBJECTIVES)]
    objective_grads = [ag.grad(obj) for obj in objectives]

    angles = np.linspace(0, 2 * np.pi, NUM_VERTICES + 1)[0:-1]
    vertex_centers_x = 1.1 * mesh_wvl_um * np.cos(angles)
    vertex_centers_y = 0.8 * mesh_wvl_um * np.sin(angles)

    input_data = [list(vertex_centers_x) + list(vertex_centers_y)]
    gradients = [np.squeeze(np.array(grad_fn(input_data)).flatten()) for grad_fn in objective_grads]

    if PLOT_ADJ_COMPARISON:
        for grad in gradients:
            plt.plot(grad)
        plt.xlabel("vertex")
        plt.ylabel("gradient")
        plt.show()

    cumulative_rms_error = 0.0
    num = 0
    for grad_1_idx in range(len(gradients)):
        for grad_2_idx in range(grad_1_idx + 1, len(gradients)):
            grad_1 = gradients[grad_1_idx]
            grad_2 = gradients[grad_2_idx]

            cumulative_rms_error += np.sqrt(np.mean((grad_1 - grad_2) ** 2))
            num += 1

    avg_rms_error = cumulative_rms_error / num

    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{test_number}")
    print(f"Mesh and adjoint wavelengths: {mesh_wvl_um}, {adj_wvl_um}")
    print(f"Geometry size: {geometry_size_wvl}")
    print(f"Average RMS Error: {avg_rms_error}")
    print("-" * 20)
    print("\n" * 3)

    save_path = None
    if SAVE_FD_ADJ_DATA:
        results_dir = numerical_case_dir / NUMERICAL_RESULTS_SUBDIR
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"results_{test_number}.npy"

    try:
        assert np.isclose(avg_rms_error, 0.0), "RMS error magnitude too large"
    finally:
        if save_path is not None:
            np.save(save_path, test_results)
