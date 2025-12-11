# test autograd for dielectric-dielectric and dielectric-PEC hybrid shape gradients and gradient consistency
# between different structures
from __future__ import annotations

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static
from tidy3d.config import config

ADJ_WVL_UM = 1.5
ADJ_FREQ0 = td.C_0 / ADJ_WVL_UM
DIELECTRIC_LEFT_PERMITTIVITY = 1.5**2
DIELECTRIC_CENTER_PERMITTIVITY = 2.5**2
LOCAL_GRADIENT = True
PLOT_FD_ADJ_COMPARISON = True
SAVE_TEST_DATA = True
NUMERICAL_RESULTS_SUBDIR = "flexible_dielectric_pec_boundary"
SAVE_EDGE_ON_CENTER_LOC = 0
SAVE_EDGE_ON_OUTSIDE_LOC = 1
RMS_NORMALIZED_THRESHOLD = 1e-12

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


@pytest.mark.numerical
@pytest.mark.parametrize("simulation_background_permittivity", (1.0**2, 1.25**2))
@pytest.mark.parametrize(
    "is_2d",
    (
        True,
        False,
    ),
)
def test_flexible_boundary_integration(
    is_2d,
    simulation_background_permittivity,
    rng,
    numerical_case_dir,
    monkeypatch,
    redirect_stdout_to_stderr,
):
    """Test that we can integrate dielectric-dielectric and dielectric-PEC boundaries on the same structure and that
    we are not sensitive to the structure we choose for autograd when the boundary is shared."""
    monkeypatch.setattr(config.adjoint, "default_wavelength_fraction", 0.01)
    thickness = 0.0 if is_2d else 0.3 * ADJ_WVL_UM
    pec_material = td.PEC2D if is_2d else td.PECMedium()
    box_2d_dielectric_material_left = td.Medium2D.from_medium(
        td.Medium(permittivity=DIELECTRIC_LEFT_PERMITTIVITY), 0.0
    )
    box_2d_dielectric_material_center = td.Medium2D.from_medium(
        td.Medium(permittivity=DIELECTRIC_CENTER_PERMITTIVITY), 0.0
    )

    box_material_left = (
        box_2d_dielectric_material_left
        if is_2d
        else td.Medium(permittivity=DIELECTRIC_LEFT_PERMITTIVITY)
    )
    box_material_center = (
        box_2d_dielectric_material_center
        if is_2d
        else td.Medium(permittivity=DIELECTRIC_CENTER_PERMITTIVITY)
    )

    def make_structures(left_center_shared_edge, center_right_shared_edge, grad_edge_on_center):
        left_box_edge = (
            get_static(left_center_shared_edge) if grad_edge_on_center else left_center_shared_edge
        )
        center_box_left_edge = (
            left_center_shared_edge if grad_edge_on_center else get_static(left_center_shared_edge)
        )

        right_box_edge = (
            get_static(center_right_shared_edge)
            if grad_edge_on_center
            else center_right_shared_edge
        )
        center_box_right_edge = (
            center_right_shared_edge
            if grad_edge_on_center
            else get_static(center_right_shared_edge)
        )

        left_box_dielectric = td.Box.from_bounds(
            (-ADJ_WVL_UM, -ADJ_WVL_UM, -0.5 * thickness),
            (left_box_edge, ADJ_WVL_UM, 0.5 * thickness),
        )

        center_box_dielectric = td.Box.from_bounds(
            (center_box_left_edge, -ADJ_WVL_UM, -0.5 * thickness),
            (center_box_right_edge, ADJ_WVL_UM, 0.5 * thickness),
        )

        right_box_pec = td.Box.from_bounds(
            (right_box_edge, -ADJ_WVL_UM, -0.5 * thickness),
            (ADJ_WVL_UM, ADJ_WVL_UM, 0.5 * thickness),
        )

        left_box = td.Structure(geometry=left_box_dielectric, medium=box_material_left)

        center_box = td.Structure(
            geometry=center_box_dielectric,
            medium=box_material_center,
            background_medium=td.PECMedium(),
        )

        right_box = td.Structure(geometry=right_box_pec, medium=pec_material)

        return [left_box, center_box, right_box]

    def make_base_sim():
        source = td.PlaneWave(
            center=(0.0, 0.0, -2 * ADJ_WVL_UM),
            size=(td.inf, td.inf, 0.0),
            source_time=td.GaussianPulse(freq0=ADJ_FREQ0, fwidth=0.2 * ADJ_FREQ0),
            direction="+",
        )

        monitor = td.FieldMonitor(
            center=(0.5 * ADJ_WVL_UM, 0.0, 2 * ADJ_WVL_UM),
            size=(0.5 * ADJ_WVL_UM, 0.5 * ADJ_WVL_UM, 0.0),
            freqs=[ADJ_FREQ0],
            name="fields",
        )

        layer_spec = td.LayerRefinementSpec.from_layer_bounds(
            axis=2,
            bounds=(-thickness / 2, thickness / 2),
        )

        grid_spec = td.GridSpec.auto(
            wavelength=ADJ_WVL_UM,
            min_steps_per_wvl=20,
            layer_refinement_specs=[layer_spec],
        )

        boundary_spec = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        )

        return td.Simulation(
            center=(0.0, 0.0, 0.0),
            size=(4.0 * ADJ_WVL_UM, 4 * ADJ_WVL_UM, 6 * ADJ_WVL_UM),
            grid_spec=grid_spec,
            boundary_spec=boundary_spec,
            run_time=1e-11,
            structures=[],
            sources=[source],
            monitors=[monitor],
            medium=td.Medium(permittivity=simulation_background_permittivity),
        )

    def make_obj_fn(grad_edge_on_center):
        def obj_fn(edges):
            left_center_shared_edge, center_right_shared_edge = edges

            base_sim = make_base_sim()

            structures = make_structures(
                left_center_shared_edge, center_right_shared_edge, grad_edge_on_center
            )

            sim_with_structures = base_sim.updated_copy(structures=structures)

            sim_data = web.run(sim_with_structures, local_gradient=LOCAL_GRADIENT)

            fields = sim_data["fields"]

            return np.sum(np.abs(fields.Ex.data) ** 2)

        return obj_fn

    obj_fn_edge_on_center = make_obj_fn(True)
    obj_fn_edge_on_outside = make_obj_fn(False)

    grad_fn_edge_on_center = ag.value_and_grad(obj_fn_edge_on_center)
    grad_fn_edge_on_outside = ag.value_and_grad(obj_fn_edge_on_outside)

    edges = [-0.5 * ADJ_WVL_UM, 0.5 * ADJ_WVL_UM]

    f_edge_on_center, g_edge_on_center = grad_fn_edge_on_center(edges)
    f_edge_on_outside, g_edge_on_outside = grad_fn_edge_on_outside(edges)

    g_edge_on_center = np.array(g_edge_on_center)
    g_edge_on_outside = np.array(g_edge_on_outside)

    test_results = np.zeros((2, len(g_edge_on_center)))
    test_results[SAVE_EDGE_ON_CENTER_LOC, :] = g_edge_on_center
    test_results[SAVE_EDGE_ON_OUTSIDE_LOC, :] = g_edge_on_outside

    rms_error_normalized = np.sqrt(
        np.mean((g_edge_on_center - g_edge_on_outside) ** 2)
    ) / np.linalg.norm(g_edge_on_outside)

    print("\n" * 3)
    print("-" * 20)
    print("Results")
    print(f"is 2d: {is_2d}")
    print(f"simulation background permittivity: {simulation_background_permittivity}")
    print(
        f"function vals (edge on center, edge on outside): {f_edge_on_center}, {f_edge_on_outside}"
    )
    print(f"gradients (edge on center, edge on outside): {g_edge_on_center}, {g_edge_on_outside}")
    print(f"rms error (normalized): {rms_error_normalized}")
    print("\n" * 3)
    print("-" * 20)

    if PLOT_FD_ADJ_COMPARISON:
        plt.scatter([0, 1], g_edge_on_center, color="b", marker="o", facecolors="none")
        plt.scatter([0, 1], g_edge_on_outside, color="g", marker="x")
        plt.legend(["edges on center box", "edges on outside boxes"])
        plt.title(
            f"Test: is_2d: {is_2d}\nsimulation background n: {np.sqrt(simulation_background_permittivity)}"
        )
        plt.show()

    try:
        assert rms_error_normalized < RMS_NORMALIZED_THRESHOLD, "RMS error too large"
    finally:
        if SAVE_TEST_DATA:
            results_dir = numerical_case_dir / NUMERICAL_RESULTS_SUBDIR
            results_dir.mkdir(parents=True, exist_ok=True)
            save_path = (
                results_dir
                / f"results_is2d_{int(is_2d)}_eps_{simulation_background_permittivity}.npy"
            )
            np.save(save_path, test_results)
