# test autograd and compares to numerically computed finite difference gradients for Box
# and PolySlab geometries.
from __future__ import annotations

from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad

import tidy3d as td
import tidy3d.web as web

WL_UM = 0.65
FREQ0 = td.C_0 / WL_UM
PERIODS_UM = (3 * WL_UM, 4 * WL_UM)
INFINITE_DIM_SIZE_UM = 0.1
SRC_OFFSET = -2.5
MONITOR_OFFSET = 2.5
PERMITTIVITY = 2.5**2
MESH_SPACING_UM = WL_UM / 40.0
FINITE_DIFFERENCE_STEP = MESH_SPACING_UM
LOCAL_GRADIENT = True
VERBOSE = False
PLOT_FD_ADJ_COMPARISON = False
SAVE_OUTPUT_DATA = True
COMPARE_TO_FINITE_DIFFERENCE = True

ANGLE_OVERLAP_THRESH_DEG = 1.0
ANGLE_OVERLAP_FD_ADJ_THRESH_DEG = 10.0

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


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


def case_identifier(is_3d: bool, infinite_dim_2d: int | None, shift_box_center: bool) -> str:
    geometry_tag = "3d" if is_3d else f"2d_infinite_dim_{infinite_dim_2d}"
    shift_tag = "shifted" if shift_box_center else "centered"
    return f"box_polyslab_{geometry_tag}_{shift_tag}"


def dimension_permutation(infinite_dim: int) -> tuple[int, int]:
    offset_dim = (1 + infinite_dim) % 3
    final_dim = (1 + offset_dim) % 3
    return offset_dim, final_dim


def make_base_simulation(is_3d: bool, infinite_dim: int | None) -> tuple[td.Simulation, callable]:
    sim_size_3d = [PERIODS_UM[0], PERIODS_UM[1], (MONITOR_OFFSET - SRC_OFFSET) + 4]

    plane_wave = td.PlaneWave(
        center=(0.0, 0.0, SRC_OFFSET),
        size=(PERIODS_UM[0], PERIODS_UM[1], 0.0),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=0.2 * FREQ0),
        direction="+",
    )

    diffraction_monitor = td.DiffractionMonitor(
        center=(0.0, 0.0, MONITOR_OFFSET),
        size=(np.inf, np.inf, 0.0),
        freqs=[FREQ0],
        name="diffraction",
        normal_dir="+",
    )

    boundary_spec_3d = td.BoundarySpec(
        x=td.Boundary.periodic(),
        y=td.Boundary.periodic(),
        z=td.Boundary.pml(),
    )

    if is_3d:
        base_sim = td.Simulation(
            center=(0.0, 0.0, 0.0),
            size=tuple(sim_size_3d),
            monitors=[diffraction_monitor],
            sources=[plane_wave],
            structures=[],
            run_time=2e-11,
            boundary_spec=boundary_spec_3d,
            grid_spec=td.GridSpec(
                grid_x=td.UniformGrid(dl=MESH_SPACING_UM),
                grid_y=td.UniformGrid(dl=MESH_SPACING_UM),
                grid_z=td.UniformGrid(dl=MESH_SPACING_UM),
            ),
        )

        def fom(sim_data):
            amps = sim_data["diffraction"].amps.sel(orders_x=1, orders_y=0).data
            return anp.sum(anp.abs(amps) ** 2)

        return base_sim, fom

    if infinite_dim is None:
        raise ValueError("infinite_dim must be provided for 2D simulations")

    offset_dim, final_dim = dimension_permutation(infinite_dim)

    sim_size_2d = sim_size_3d[:]
    sim_size_2d[infinite_dim] = 0.0
    sim_size_2d[offset_dim] = MONITOR_OFFSET - SRC_OFFSET + 4
    sim_size_2d[final_dim] = PERIODS_UM[0]

    src_center = [0.0, 0.0, 0.0]
    src_center[offset_dim] = SRC_OFFSET

    src_size = [0.0, 0.0, 0.0]
    src_size[infinite_dim] = INFINITE_DIM_SIZE_UM
    src_size[final_dim] = PERIODS_UM[0]

    diffraction_size = [np.inf, np.inf, np.inf]
    diffraction_size[offset_dim] = 0.0

    mode_monitor_center = [0.0, 0.0, 0.0]
    mode_monitor_center[offset_dim] = MONITOR_OFFSET

    input_src_2d = td.PlaneWave(
        center=tuple(src_center),
        size=tuple(src_size),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=0.2 * FREQ0),
        direction="+",
    )

    diffraction_monitor_2d = td.DiffractionMonitor(
        center=tuple(mode_monitor_center),
        size=tuple(diffraction_size),
        freqs=[FREQ0],
        name="diffraction",
        normal_dir="+",
    )

    boundary_spec_list = [td.Boundary.periodic() for _ in range(3)]
    boundary_spec_list[offset_dim] = td.Boundary.pml()

    coords = "xyz"
    boundary_spec_2d = td.BoundarySpec(**{coords[idx]: boundary_spec_list[idx] for idx in range(3)})

    base_sim = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=tuple(sim_size_2d),
        monitors=[diffraction_monitor_2d],
        sources=[input_src_2d],
        structures=[],
        run_time=1e-11,
        boundary_spec=boundary_spec_2d,
        grid_spec=td.GridSpec(
            grid_x=td.UniformGrid(dl=MESH_SPACING_UM),
            grid_y=td.UniformGrid(dl=MESH_SPACING_UM),
            grid_z=td.UniformGrid(dl=MESH_SPACING_UM),
        ),
    )

    def fom(sim_data):
        choose_order_x = 1 if infinite_dim > 0 else 0
        choose_order_y = 0 if infinite_dim > 0 else 1

        amps = (
            sim_data["diffraction"].amps.sel(orders_x=choose_order_x, orders_y=choose_order_y).data
        )
        return anp.sum(anp.abs(amps) ** 2)

    return base_sim, fom


def make_box_geometry(params, box_center):
    return td.Box(center=box_center, size=tuple(params))


def make_polyslab_geometry(params, box_center, axis: int) -> td.PolySlab:
    half_size = 0.5 * params
    slab_bounds = (
        box_center[axis] - half_size[axis],
        box_center[axis] + half_size[axis],
    )
    plane_axes = [idx for idx in range(3) if idx != axis]

    vertices = []
    for sign_0, sign_1 in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        coord_0 = box_center[plane_axes[0]] + sign_0 * half_size[plane_axes[0]]
        coord_1 = box_center[plane_axes[1]] + sign_1 * half_size[plane_axes[1]]
        vertices.append((coord_0, coord_1))

    return td.PolySlab(vertices=tuple(vertices), slab_bounds=slab_bounds, axis=axis)


def run_parameter_simulations(
    parameter_sets: list[anp.ndarray],
    make_geometry,
    box_center,
    tag: str,
    base_sim: td.Simulation,
    fom,
    artifact_dir: Path,
    *,
    local_gradient: bool,
):
    simulation_dict = {}
    output_dir = artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, param_values in enumerate(parameter_sets):
        geometry = make_geometry(param_values, box_center)
        structure = td.Structure(
            geometry=geometry,
            medium=td.Medium(permittivity=PERMITTIVITY),
        )

        sim = base_sim.updated_copy(structures=[structure])
        simulation_dict[f"sim_{idx}"] = sim

    if len(simulation_dict) == 1:
        key, sim = next(iter(simulation_dict.items()))
        result_path = output_dir / f"{key}.hdf5"
        sim_data = web.run(
            sim,
            task_name=key,
            path=str(result_path),
            local_gradient=local_gradient,
            verbose=VERBOSE,
        )
        return fom(sim_data)

    sim_data_map = web.run_async(
        simulation_dict,
        path_dir=str(output_dir),
        local_gradient=local_gradient,
        verbose=VERBOSE,
    )

    return [fom(sim_data_map[key]) for key in simulation_dict]


def make_objective(
    make_geometry,
    box_center,
    tag: str,
    base_sim: td.Simulation,
    fom,
    case_dir: Path,
    *,
    local_gradient: bool,
):
    artifact_dir = case_dir / tag
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def objective(parameters):
        results = run_parameter_simulations(
            parameters,
            make_geometry,
            box_center,
            tag,
            base_sim,
            fom,
            artifact_dir,
            local_gradient=local_gradient,
        )

        return results

    return objective


def finite_difference(objective, params, is_3d: bool, infinite_dim: int | None):
    fd_grad = anp.zeros(3)

    perturbations = []
    valid_indices = []

    for idx in range(3):
        if (not is_3d) and infinite_dim is not None and idx == infinite_dim:
            continue

        params_up = anp.array(params)
        params_down = anp.array(params)

        params_up[idx] += FINITE_DIFFERENCE_STEP
        params_down[idx] -= FINITE_DIFFERENCE_STEP

        perturbations.append(params_up)
        perturbations.append(params_down)
        valid_indices.append(idx)

    objectives = objective(anp.stack(perturbations))

    for pair_idx, param_idx in enumerate(valid_indices):
        obj_up = objectives[2 * pair_idx]
        obj_down = objectives[2 * pair_idx + 1]
        fd_grad = anp.array(fd_grad)
        fd_grad = fd_grad.copy()
        fd_grad[param_idx] = (obj_up - obj_down) / (2.0 * FINITE_DIFFERENCE_STEP)

    return fd_grad


def squeeze_dimension(array: np.ndarray, is_3d: bool, infinite_dim: int | None) -> np.ndarray:
    if is_3d or infinite_dim is None:
        return array

    squeezed = np.squeeze(array)
    return np.delete(squeezed, infinite_dim)


@pytest.mark.numerical
@pytest.mark.parametrize(
    "is_3d, infinite_dim_2d",
    [
        (True, 2),
        (False, 0),
        (False, 1),
        (False, 2),
    ],
)
@pytest.mark.parametrize("shift_box_center", (True, False))
def test_box_and_polyslab_gradients_match(
    is_3d, infinite_dim_2d, shift_box_center, numerical_case_dir, redirect_stdout_to_stderr
):
    """Test that the box and polyslab gradients match for rectangular slab geometries. Allow
    comparison as well to finite difference values."""

    base_sim, fom = make_base_simulation(is_3d, infinite_dim_2d if not is_3d else None)

    if shift_box_center:
        box_init_size = [2.0 * WL_UM, 2.5 * WL_UM, 0.75 * WL_UM]
    else:
        box_init_size = [1.0 * WL_UM, 1.25 * WL_UM, 0.75 * WL_UM]

    initial_params = anp.array(box_init_size)

    polyslab_axis = 2 if is_3d else infinite_dim_2d

    box_center = [0.0, 0.0, 0.0]
    if shift_box_center:
        # test what happens when part of the box falls outside the simulation domain
        # but don't shift along source axis
        if is_3d:
            box_center[0:2] = [0.5 * p for p in PERIODS_UM]
        else:
            _, final_dim_2d = dimension_permutation(infinite_dim_2d)
            box_center[infinite_dim_2d] = 0.5 * INFINITE_DIM_SIZE_UM
            box_center[final_dim_2d] = 0.5 * PERIODS_UM[0]

    case_id = case_identifier(is_3d, None if is_3d else infinite_dim_2d, shift_box_center)
    case_dir = numerical_case_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    box_objective = make_objective(
        make_box_geometry,
        box_center,
        "box",
        base_sim,
        fom,
        case_dir,
        local_gradient=LOCAL_GRADIENT,
    )
    polyslab_objective = make_objective(
        lambda p, box_center: make_polyslab_geometry(p, box_center, polyslab_axis),
        box_center,
        "polyslab",
        base_sim,
        fom,
        case_dir,
        local_gradient=LOCAL_GRADIENT,
    )

    box_objective_fd = make_objective(
        make_box_geometry,
        box_center,
        "box_fd",
        base_sim,
        fom,
        case_dir,
        local_gradient=False,
    )
    polyslab_objective_fd = make_objective(
        lambda p, box_center: make_polyslab_geometry(p, box_center, polyslab_axis),
        box_center,
        "polyslab_fd",
        base_sim,
        fom,
        case_dir,
        local_gradient=False,
    )

    _box_value, box_grad = value_and_grad(box_objective)([initial_params])
    _polyslab_value, polyslab_grad = value_and_grad(polyslab_objective)([initial_params])

    box_grad_np = box_grad
    polyslab_grad_np = polyslab_grad

    box_grad_filtered = squeeze_dimension(box_grad_np, is_3d, infinite_dim_2d)
    polyslab_grad_filtered = squeeze_dimension(polyslab_grad_np, is_3d, infinite_dim_2d)

    fd_box = squeeze_dimension(
        finite_difference(box_objective_fd, initial_params, is_3d, infinite_dim_2d),
        is_3d,
        infinite_dim_2d,
    )
    fd_polyslab = squeeze_dimension(
        finite_difference(polyslab_objective_fd, initial_params, is_3d, infinite_dim_2d),
        is_3d,
        infinite_dim_2d,
    )

    test_data = {
        "fd box": fd_box,
        "fd polyslab": fd_polyslab,
        "grad box": box_grad_filtered,
        "grad polyslab": polyslab_grad_filtered,
    }

    if SAVE_OUTPUT_DATA:
        npz_path = case_dir / (
            f"test_diff_init_{'3' if is_3d else '2'}d_infinite_dim_{infinite_dim_2d}.npz"
        )
        np.savez(npz_path, **test_data)

    box_polyslab_overlap_deg = angled_overlap_deg(box_grad_filtered, polyslab_grad_filtered)
    fd_overlap_deg = angled_overlap_deg(fd_box, fd_polyslab)
    box_fd_adj_overlap_deg = angled_overlap_deg(box_grad_filtered, fd_box)
    polyslab_fd_adj_overlap_deg = angled_overlap_deg(polyslab_grad_filtered, fd_polyslab)

    print("Overlaps (deg):")
    print(f"Box vs. PolySlab Adjoint: {box_polyslab_overlap_deg}")
    print(f"Box vs. PolySlab Finite Difference: {fd_overlap_deg}")
    print(f"Box Finite Difference vs. Adjoint: {box_fd_adj_overlap_deg}")
    print(f"PolySlab Finite Difference vs. Adjoint: {polyslab_fd_adj_overlap_deg}")

    assert box_polyslab_overlap_deg < ANGLE_OVERLAP_THRESH_DEG, (
        "Autograd gradients for Box and PolySlab disagree"
    )
    assert fd_overlap_deg < ANGLE_OVERLAP_THRESH_DEG, (
        "Finite-difference gradients for Box and PolySlab disagree"
    )

    if COMPARE_TO_FINITE_DIFFERENCE:
        assert box_fd_adj_overlap_deg < ANGLE_OVERLAP_FD_ADJ_THRESH_DEG, (
            "Autograd and finite-difference gradients for the Box geometry disagree"
        )
        assert polyslab_fd_adj_overlap_deg < ANGLE_OVERLAP_FD_ADJ_THRESH_DEG, (
            "Autograd and finite-difference gradients for the PolySlab geometry disagree"
        )
