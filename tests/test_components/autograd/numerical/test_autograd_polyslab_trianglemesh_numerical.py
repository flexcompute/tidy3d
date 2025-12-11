# Test autograd and compare to numerically computed finite difference gradients for
# PolySlab and TriangleMesh geometries representing the same rectangular slab.
from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad

import tidy3d as td
from tests.test_components.autograd.numerical.test_autograd_box_polyslab_numerical import (
    angled_overlap_deg,
    dimension_permutation,
    finite_difference,
    make_base_simulation,
    run_parameter_simulations,
    squeeze_dimension,
)
from tidy3d import config

config.local_cache.enabled = True
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
COMPARE_TO_POLYSLAB = True

ANGLE_OVERLAP_THRESH_DEG = 10.0
ANGLE_OVERLAP_FD_ADJ_THRESH_DEG = 10.0

VERTEX_SIGNS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ]
)

TRIANGLE_FACE_VERTEX_IDS = np.array(
    [
        (1, 3, 0),
        (4, 1, 0),
        (0, 3, 2),
        (2, 4, 0),
        (1, 7, 3),
        (5, 1, 4),
        (5, 7, 1),
        (3, 7, 2),
        (6, 4, 2),
        (2, 7, 6),
        (6, 5, 4),
        (7, 5, 6),
    ],
    dtype=int,
)

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


def _triangles_from_params(params, box_center):
    params_arr = anp.array(params)
    center_arr = anp.array(box_center)
    half_size = 0.5 * params_arr
    vertices = center_arr + anp.array(VERTEX_SIGNS) * half_size
    return vertices[anp.array(TRIANGLE_FACE_VERTEX_IDS)]


def make_trianglemesh_geometry(params, box_center):
    triangles = _triangles_from_params(params, box_center)
    mesh = td.TriangleMesh.from_triangles(triangles)
    return mesh


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


def make_objective(
    make_geometry,
    box_center,
    tag: str,
    base_sim: td.Simulation,
    fom,
    tmp_path,
    *,
    local_gradient: bool,
):
    def objective(parameters):
        results = run_parameter_simulations(
            parameters,
            make_geometry,
            box_center,
            tag,
            base_sim,
            fom,
            tmp_path,
            local_gradient=local_gradient,
        )

        return results

    return objective


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
def test_polyslab_and_trianglemesh_gradients_match(
    is_3d, infinite_dim_2d, shift_box_center, tmp_path, redirect_stdout_to_stderr
):
    """Test that the triangle mesh and polyslab gradients match for rectangular slab geometries. Allow
    comparison as well to finite difference values."""

    base_sim, fom = make_base_simulation(is_3d, infinite_dim_2d if not is_3d else None)

    if shift_box_center:
        slab_init_size = [2.0 * WL_UM, 2.5 * WL_UM, 0.75 * WL_UM]
    else:
        slab_init_size = [1.0 * WL_UM, 1.25 * WL_UM, 0.75 * WL_UM]

    initial_params = anp.array(slab_init_size)

    polyslab_axis = 2 if is_3d else infinite_dim_2d

    box_center = [0.0, 0.0, 0.0]
    if shift_box_center:
        # test what happens when part of the structure falls outside the simulation domain
        # but don't shift along source axis
        if is_3d:
            box_center[0:2] = [0.5 * p for p in PERIODS_UM]
        else:
            _, final_dim_2d = dimension_permutation(infinite_dim_2d)
            box_center[infinite_dim_2d] = 0.5 * INFINITE_DIM_SIZE_UM
            box_center[final_dim_2d] = 0.5 * PERIODS_UM[0]

    triangle_objective = make_objective(
        make_trianglemesh_geometry,
        box_center,
        "trianglemesh",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )

    polyslab_objective = make_objective(
        lambda p, box_center: make_polyslab_geometry(p, box_center, polyslab_axis),
        box_center,
        "polyslab",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )

    triangle_objective_fd = make_objective(
        make_trianglemesh_geometry,
        box_center,
        "trianglemesh_fd",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    _triangle_value, triangle_grad = value_and_grad(triangle_objective)([initial_params])
    assert triangle_grad is not None
    if is_3d or infinite_dim_2d not in [1, 2]:
        grad_norm_triangle = np.linalg.norm(triangle_grad)
        assert grad_norm_triangle > 1e-6, (
            f"Assumed norm to be bigger than 1e-6, got {grad_norm_triangle}"
        )
    triangle_grad_filtered = squeeze_dimension(triangle_grad, is_3d, infinite_dim_2d)
    polyslab_grad_filtered = None
    if COMPARE_TO_POLYSLAB:
        _polyslab_value, polyslab_grad = value_and_grad(polyslab_objective)([initial_params])
        polyslab_grad_filtered = squeeze_dimension(polyslab_grad, is_3d, infinite_dim_2d)
        print(
            "polyslab_grad_filtered\t",
            polyslab_grad_filtered.tolist()
            if not isinstance(polyslab_grad_filtered, list)
            else polyslab_grad_filtered,
        )

    fd_triangle = None
    if COMPARE_TO_FINITE_DIFFERENCE:
        fd_triangle = squeeze_dimension(
            finite_difference(triangle_objective_fd, initial_params, is_3d, infinite_dim_2d),
            is_3d,
            infinite_dim_2d,
        )

    if SAVE_OUTPUT_DATA:
        test_data = {
            "fd trianglemesh": fd_triangle,
            "grad trianglemesh": triangle_grad_filtered,
            "grad polyslab": polyslab_grad_filtered,
        }
        np.savez(
            f"test_diff_triangle_poly_{'3' if is_3d else '2'}d_infinite_dim_{infinite_dim_2d}.npz",
            **test_data,
        )

    if COMPARE_TO_POLYSLAB:
        triangle_polyslab_overlap_deg = angled_overlap_deg(
            triangle_grad_filtered, polyslab_grad_filtered
        )
        print(f"TriangleMesh FD vs. polyslab overlap: {triangle_polyslab_overlap_deg:.3f}° ")
        assert triangle_polyslab_overlap_deg < ANGLE_OVERLAP_THRESH_DEG, (
            f"[TriangleMesh vs. PolySlab] Autograd gradients disagree: "
            f"angle overlap = {triangle_polyslab_overlap_deg:.3f}° "
            f"(threshold = {ANGLE_OVERLAP_THRESH_DEG:.3f}°, "
            f"difference = {triangle_polyslab_overlap_deg - ANGLE_OVERLAP_THRESH_DEG:+.3f}°)"
        )

    if COMPARE_TO_FINITE_DIFFERENCE:
        triangle_fd_adj_overlap_deg = angled_overlap_deg(triangle_grad_filtered, fd_triangle)
        print(f"TriangleMesh FD vs. Adjoint angle overlap: {triangle_fd_adj_overlap_deg:.3f}° ")
        assert triangle_fd_adj_overlap_deg < ANGLE_OVERLAP_FD_ADJ_THRESH_DEG, (
            f"Autograd and finite-difference gradients disagree: "
            f"angle overlap = {triangle_fd_adj_overlap_deg:.3f}° "
            f"(threshold = {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG:.3f}°, "
            f"difference = {triangle_fd_adj_overlap_deg - ANGLE_OVERLAP_FD_ADJ_THRESH_DEG:+.3f}°)"
        )
