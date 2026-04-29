# Test autograd gradients for scaled spheres represented as TriangleMesh
# geometries, comparing to finite differences for validation.
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from matplotlib import pyplot as plt

import tidy3d as td
import tidy3d.web as web
from tests.test_components.autograd.numerical.test_autograd_box_polyslab_numerical import (
    angled_overlap_deg,
)
from tests.test_components.autograd.test_autograd_triangle_mesh import subdivide_triangles
from tidy3d import config
from tidy3d.components.autograd import get_static
from tidy3d.components.geometry.primitives import _base_icosahedron

config.local_cache.enabled = True
config.logging.level = "DEBUG"

WL_UM = 0.65
SPHERE_RADIUS_UM = 0.5 * WL_UM
SCALE_FACTORS = (0.2, 1.0, 5.0)
SCALE_AXES = (0, 1, 2)

FREQ0 = td.C_0 / WL_UM
SRC_OFFSET = -2.5
MONITOR_OFFSET = 2.5
N_MAT = 2
PERMITTIVITY = N_MAT**2
ICOSAHEDRON_SUBDIVISIONS = 3
LOCAL_GRADIENT = True
VERBOSE = False
SAVE_OUTPUT_DATA = True
ANGLE_OVERLAP_FD_ADJ_THRESH_DEG = 10.0
VERTEX_FD_STEP = 1e-3
FINITE_DIFF_STEP = 1e-2
FINITE_DIFF_STEP_NATIVE = 1e-3
GRID_STEPS_PER_WVL = 40
td.config.adjoint.points_per_wavelength = 10
measure_flux_spec = False

freqs = td.C_0 / np.linspace(0.6, 0.7, 101)


def make_base_simulation(
    radii: list[float],
    *,
    extra_structures: Sequence[td.Structure] | None = None,
    is_2d: bool = False,
) -> tuple[td.Simulation, callable]:
    sim_size_3d = [
        2 * radii[0] + 2 * WL_UM,
        2 * radii[1] + 2 * WL_UM,
        (MONITOR_OFFSET - SRC_OFFSET) + 2 * WL_UM + 2 * radii[2],
    ]

    if is_2d:
        sim_size_3d[1] = 0.0

    source_time = td.GaussianPulse(freq0=FREQ0, fwidth=0.2 * FREQ0)
    if is_2d:
        primary_source = td.PointDipole(
            center=(0.0, 0.0, SRC_OFFSET),
            source_time=source_time,
            polarization="Ez",
        )
    else:
        primary_source = td.PlaneWave(
            center=(0.0, 0.0, SRC_OFFSET),
            size=(*sim_size_3d[:2], 0.0),
            source_time=source_time,
            direction="+",
        )

    flux_monitors = [
        td.FieldMonitor(
            center=(0.0, 0.0, MONITOR_OFFSET),
            size=(*sim_size_3d[:2], 0.0),
            freqs=FREQ0,
            name="field",
        )
    ]
    if measure_flux_spec:
        flux_monitors.append(
            td.FieldMonitor(
                center=(0.0, 0.0, MONITOR_OFFSET),
                size=(*sim_size_3d[:2], 0.0),
                freqs=freqs,
                name="field_spectrum",
            )
        )

    boundary_spec_3d = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml() if not is_2d else td.Boundary.periodic(),
        z=td.Boundary.pml(),
    )

    base_sim = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=tuple(sim_size_3d),
        monitors=flux_monitors,
        sources=[primary_source],
        structures=list(extra_structures) if extra_structures else [],
        run_time=2e-11,
        boundary_spec=boundary_spec_3d,
        grid_spec=td.GridSpec(
            grid_x=td.UniformGrid(dl=WL_UM / GRID_STEPS_PER_WVL),
            grid_y=td.UniformGrid(dl=WL_UM / GRID_STEPS_PER_WVL),
            grid_z=td.UniformGrid(dl=WL_UM / GRID_STEPS_PER_WVL),
        ),
    )

    def fom(sim_data):
        dataset = sim_data["field"]
        if is_2d:
            ex_vals = dataset.Ex.values
            ey_vals = dataset.Ey.values
            ez_vals = dataset.Ez.values
            intensity = anp.abs(ex_vals) ** 2 + anp.abs(ey_vals) ** 2 + anp.abs(ez_vals) ** 2
            return anp.real(anp.mean(intensity))
        return dataset.flux.values

    return base_sim, fom


def make_overlap_cube_structure(radii: Sequence[float]) -> td.Structure:
    radii_arr = np.asarray(radii, dtype=float)
    size_x = float(radii_arr[0])
    size_y = float(2.0 * radii_arr[1])
    size_z = float(2.0 * radii_arr[2])
    cube_center = (size_x / 2.0, 0.0, 0.0)
    cube = td.Box(center=cube_center, size=(size_x, size_y, size_z))
    cube_medium = td.Medium(permittivity=PERMITTIVITY)
    return td.Structure(geometry=cube, medium=cube_medium)


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

        base_structures = list(getattr(base_sim, "structures", ()))
        structures = [structure, *base_structures]
        grid_spec = td.GridSpec.auto(
            min_steps_per_wvl=GRID_STEPS_PER_WVL, override_structures=[structure]
        )
        sim = base_sim.updated_copy(structures=structures, grid_spec=grid_spec, validate=True)
        task_name = f"{tag}_idx{idx}"
        simulation_dict[task_name] = sim

    if len(simulation_dict) == 1:
        key, sim = next(iter(simulation_dict.items()))
        result_path = output_dir / f"{sim._hash_self()}.hdf5"
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


_ICOSAHEDRON_VERTS, _ICOSAHEDRON_FACES = _base_icosahedron()


def make_sphere_triangle_geometry(
    params: anp.ndarray,
    center: Sequence[float],
    scale_factor: float,
    scale_axis: int,
    subdivisions: int = ICOSAHEDRON_SUBDIVISIONS,
    in_plane_subdivisions: int = 0,
) -> td.Geometry:
    radii = anp.array(params, dtype=float)
    triangles = td.Sphere.unit_sphere_triangles(subdivisions=subdivisions)
    if in_plane_subdivisions > 0:
        for _ in range(in_plane_subdivisions):
            triangles = subdivide_triangles(triangles)
    triangles = anp.array(triangles)
    triangles = triangles * radii
    axis_selector = anp.equal(anp.arange(3), scale_axis)
    scale_vec = anp.where(axis_selector, scale_factor, 1.0)
    triangles = triangles * scale_vec
    center_arr = anp.array(center, dtype=float)
    triangles = triangles + center_arr
    mesh = td.TriangleMesh.from_triangles(triangles)
    return mesh


def make_native_sphere_geometry(params: anp.ndarray, center: Sequence[float]) -> td.Geometry:
    return td.Sphere(center=tuple(center), radius=params[0])


def make_mesh_sphere_from_radius(params: anp.ndarray, center: Sequence[float]) -> td.Geometry:
    radius = params[0]
    radii = anp.full(3, radius)
    return make_sphere_triangle_geometry(radii, center, scale_factor=1.0, scale_axis=0)


def finite_difference_params(objective, params: anp.ndarray, finite_diff_step) -> np.ndarray:
    step = np.full_like(np.asarray(params, dtype=float), finite_diff_step, dtype=float)
    perturbations = []
    valid_indices = []

    for idx in range(params.size):
        params_up = anp.array(params)
        params_down = anp.array(params)
        params_up = params_up.copy()
        params_down = params_down.copy()
        params_up[idx] += step[idx]
        params_down[idx] -= step[idx]
        perturbations.extend([params_up, params_down])
        valid_indices.append(idx)

    objectives = objective(anp.stack(perturbations))
    objectives = np.squeeze(np.asarray(objectives, dtype=float))
    fd = np.zeros_like(np.asarray(params, dtype=float))
    for pair_idx, param_idx in enumerate(valid_indices):
        obj_up = objectives[2 * pair_idx]
        obj_down = objectives[2 * pair_idx + 1]
        fd[param_idx] = float((obj_up - obj_down) / (2.0 * step[param_idx]))

    return fd


def finite_difference_params_step_batch(
    objective, params: anp.ndarray, finite_diff_step
) -> np.ndarray:
    """Compute central finite-difference gradients for one or more FD step sizes.

    Args:
        objective: Callable accepting a batch of parameter vectors and returning objectives
                   (shape (N,) or (N,1)).
        params: Parameter vector (1D array).
        finite_diff_step: Either a scalar, array of per-parameter step sizes, or list of such.

    Returns:
        np.ndarray: Finite-difference gradient(s).
                    Shape is (len(steps), n_params) if multiple steps,
                    or (n_params,) for a single step.
    """
    params = anp.asarray(params, dtype=float)

    # Normalize `finite_diff_step` to a list of arrays, one per step value
    if np.isscalar(finite_diff_step):
        step_list = [anp.full_like(params, float(finite_diff_step), dtype=float)]
    else:
        finite_diff_step = np.atleast_1d(finite_diff_step)
        if finite_diff_step.ndim == 1 and finite_diff_step.size == params.size:
            # one per parameter
            step_list = [anp.asarray(finite_diff_step, dtype=float)]
        else:
            # list/array of scalar step values
            step_list = [anp.full_like(params, float(s), dtype=float) for s in finite_diff_step]

    grads = []

    for step in step_list:
        perturbations = []
        valid_indices = []
        for idx in range(params.size):
            params_up = anp.array(params)
            params_down = anp.array(params)
            params_up[idx] += step[idx]
            params_down[idx] -= step[idx]
            perturbations.extend([params_up, params_down])
            valid_indices.append(idx)

        objectives = objective(anp.stack(perturbations))
        objectives = np.asarray(objectives, dtype=float).ravel()

        fd = np.zeros_like(params, dtype=float)
        for pair_idx, param_idx in enumerate(valid_indices):
            obj_up = objectives[2 * pair_idx]
            obj_down = objectives[2 * pair_idx + 1]
            fd[param_idx] = (obj_up - obj_down) / (2.0 * step[param_idx])

        grads.append(fd)

    grads = np.stack(grads, axis=0)
    return grads[0] if len(grads) == 1 else grads


def make_objective(
    make_geometry: Callable[[anp.ndarray, Sequence[float]], td.Geometry],
    center: Sequence[float],
    tag: str,
    base_sim: td.Simulation,
    fom: Callable,
    tmp_path,
    *,
    local_gradient: bool,
):
    def objective(parameters):
        return run_parameter_simulations(
            parameters,
            make_geometry,
            center,
            tag,
            base_sim,
            fom,
            tmp_path,
            local_gradient=local_gradient,
        )

    return objective


@pytest.mark.numerical
@pytest.mark.parametrize("scale_factor", (1,))
@pytest.mark.parametrize("scale_axis", (0,))
@pytest.mark.parametrize("overlap_cube", (False,))
def test_sphere_triangles_match_fd(
    scale_factor, scale_axis, overlap_cube, tmp_path, numerical_case_dir, redirect_stdout_to_stderr
):
    """
    Compares FD gradients with gradients from _compute_derivatives in TriangleMesh.
    Note that FD gradients are very noise which is why there will be some failing tests with a fixed FD-step
    """
    if scale_factor == 1 and scale_axis > 0:
        pytest.skip("Skipping duplicate test.")

    initial_params = [SPHERE_RADIUS_UM, SPHERE_RADIUS_UM, SPHERE_RADIUS_UM]
    params0 = anp.array(initial_params)

    radii = initial_params.copy()
    radii[scale_axis] *= scale_factor
    extra_structures = [make_overlap_cube_structure(radii)] if overlap_cube else []
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=extra_structures)

    center = [0.0, 0.0, 0.0]

    part_make_geom = lambda p, c: make_sphere_triangle_geometry(p, c, scale_factor, scale_axis)

    triangle_objective = make_objective(
        part_make_geom,
        center,
        f"sphere_mesh_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )
    triangle_objective_fd = make_objective(
        part_make_geom,
        center,
        f"sphere_mesh_fd_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    _, triangle_grad = value_and_grad(triangle_objective)([params0])
    assert triangle_grad is not None

    triangle_grad = np.squeeze(np.asarray(triangle_grad, dtype=float))

    fd_grad = finite_difference_params(triangle_objective_fd, params0, FINITE_DIFF_STEP)

    print("scale", scale_factor, "axis", scale_axis, "overlap_cube", overlap_cube)
    print("triangle_grad\t", triangle_grad.tolist())
    print("fd_grad\t\t", fd_grad.tolist())

    mesh_fd_overlap = angled_overlap_deg(triangle_grad, fd_grad)
    print(
        f"TriangleMesh FD vs. Adjoint angle overlap: {mesh_fd_overlap:.3f}° "
        f"(threshold = {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG}°)"
    )
    assert mesh_fd_overlap < ANGLE_OVERLAP_FD_ADJ_THRESH_DEG, (
        f"FD–adjoint angle overlap too large: {mesh_fd_overlap:.3f}° "
        f"(threshold {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG}°, "
    )

    if SAVE_OUTPUT_DATA:
        np.savez(
            numerical_case_dir
            / f"sphere_gradients_mesh_scale_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}.npz",
            triangle_grad=triangle_grad,
            fd_grad=fd_grad,
        )


@pytest.mark.skip
def test_grad_insensitive_to_face_splitting(
    tmp_path, numerical_case_dir, redirect_stdout_to_stderr
):
    scale_factor = 1
    scale_axis = 0

    initial_params = [SPHERE_RADIUS_UM] * 3
    params0 = anp.array(initial_params)

    radii = initial_params.copy()
    radii[scale_axis] *= scale_factor
    base_sim, fom = make_base_simulation(radii=radii)

    center = [0.0, 0.0, 0.0]

    # clean objective names
    obj_name_base = "sphere_mesh_subdiv_0"
    obj_name_subdiv_1 = "sphere_mesh_subdiv_1"
    obj_name_subdiv_2 = "sphere_mesh_subdiv_2"

    triangle_objective_base = make_objective(
        lambda p, c: make_sphere_triangle_geometry(
            p, c, scale_factor, scale_axis, subdivisions=0, in_plane_subdivisions=0
        ),
        center,
        obj_name_base,
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )

    triangle_objective_subdiv_1 = make_objective(
        lambda p, c: make_sphere_triangle_geometry(
            p, c, scale_factor, scale_axis, subdivisions=0, in_plane_subdivisions=1
        ),
        center,
        obj_name_subdiv_1,
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )

    triangle_objective_subdiv_2 = make_objective(
        lambda p, c: make_sphere_triangle_geometry(
            p, c, scale_factor, scale_axis, subdivisions=0, in_plane_subdivisions=2
        ),
        center,
        obj_name_subdiv_2,
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )

    # ---- Evaluate adjoint gradients for base and subdivided meshes ----
    _, grad_base = value_and_grad(triangle_objective_base)([params0])
    _, grad_subdiv_1 = value_and_grad(triangle_objective_subdiv_1)([params0])
    _, grad_subdiv_2 = value_and_grad(triangle_objective_subdiv_2)([params0])

    assert grad_base is not None
    assert grad_subdiv_1 is not None
    assert grad_subdiv_2 is not None

    grad_base = np.squeeze(np.asarray(grad_base, dtype=float))
    grad_subdiv_1 = np.squeeze(np.asarray(grad_subdiv_1, dtype=float))
    grad_subdiv_2 = np.squeeze(np.asarray(grad_subdiv_2, dtype=float))

    print("grad_base       \t", grad_base.tolist())
    print("grad_subdiv_1   \t", grad_subdiv_1.tolist())
    print("grad_subdiv_2   \t", grad_subdiv_2.tolist())

    # Optional: angles for debugging / log inspection.
    angle_base_vs_1 = angled_overlap_deg(grad_base, grad_subdiv_1)
    angle_base_vs_2 = angled_overlap_deg(grad_base, grad_subdiv_2)
    print(
        f"Base vs subdiv-1 angle: {angle_base_vs_1:.3f}°; "
        f"Base vs subdiv-2 angle: {angle_base_vs_2:.3f}°"
    )

    np.testing.assert_allclose(grad_base, grad_subdiv_1, rtol=0.1, atol=5e-6)
    np.testing.assert_allclose(grad_base, grad_subdiv_2, rtol=0.1, atol=5e-6)

    if SAVE_OUTPUT_DATA:
        np.savez(
            numerical_case_dir / "sphere_gradients_mesh_subdiv.npz",
            grad_base=grad_base,
            grad_subdiv_1=grad_subdiv_1,
            grad_subdiv_2=grad_subdiv_2,
        )


@pytest.mark.numerical
@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
@pytest.mark.parametrize("scale_axis", SCALE_AXES)
@pytest.mark.parametrize("overlap_cube", (False, True))
def test_triangle_sphere_fd_step_sweep_ref(
    tmp_path, scale_factor, scale_axis, overlap_cube, numerical_case_dir, redirect_stdout_to_stderr
):
    initial_params = [SPHERE_RADIUS_UM, SPHERE_RADIUS_UM, SPHERE_RADIUS_UM]
    params0 = anp.array(initial_params)

    radii = initial_params.copy()
    radii[scale_axis] *= scale_factor
    extra_structures = [make_overlap_cube_structure(radii)] if overlap_cube else []
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=extra_structures)

    center = [0.0, 0.0, 0.0]

    part_make_geom = lambda p, c: make_sphere_triangle_geometry(p, c, scale_factor, scale_axis)

    # ---- Build objectives ----
    triangle_objective_fd = make_objective(
        part_make_geom,
        center,
        "sphere_mesh_fd_step_sweep",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )
    triangle_objective_autograd = make_objective(
        part_make_geom,
        center,
        "sphere_mesh_autograd_ref",
        base_sim,
        fom,
        tmp_path,
        local_gradient=True,
    )

    # ---- Compute autograd gradient (for reference) ----
    _, autograd_grad = value_and_grad(triangle_objective_autograd)([params0])
    autograd_grad = np.squeeze(np.asarray(autograd_grad, dtype=float))

    # ---- Finite difference sweep ----
    steps = np.logspace(-4, -1, num=12)
    fd_grads = finite_difference_params_step_batch(triangle_objective_fd, params0, steps)

    fd_grads = np.asarray(fd_grads, dtype=float)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, label in enumerate(["radius_x", "radius_y", "radius_z"]):
        ax.plot(steps, fd_grads[:, idx], marker="o", label=label)
        # Add horizontal reference line for autograd
        ax.axhline(
            autograd_grad[idx],
            color=ax.get_lines()[-1].get_color(),
            linestyle="--",
            alpha=0.7,
            label=f"{label} (autograd)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step (µm)")
    ax.set_ylabel("Gradient value")
    ax.set_title("FD gradients vs. step size")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig_path = (
        numerical_case_dir
        / f"fd_step_sweep_scale_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}.png"
    )

    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    np.savez(
        numerical_case_dir
        / f"fd_step_sweep_scale_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}.npz",
        steps=steps,
        gradients=fd_grads,
        autograd_grad=autograd_grad,
    )


@pytest.mark.numerical
@pytest.mark.parametrize("radius_scale", (0.25, 0.5, 1))
@pytest.mark.parametrize("overlap_cube", (False, True))
@pytest.mark.parametrize("parametrization", ("radius", "center"))
@pytest.mark.parametrize("is_2d", (False,), ids=lambda x: "2D" if x else "3D")
def test_native_sphere_match_fd(
    radius_scale, overlap_cube, parametrization, is_2d, tmp_path, numerical_case_dir
):
    """
    Compares FD gradients with gradients from _compute_derivatives in Sphere.
    Note that FD gradients are very noise which is why there will be some failing tests with a fixed FD-step.
    Currently, numerical tests fail for 2D as FD gradients are very shaky.
    """
    radius = SPHERE_RADIUS_UM * radius_scale
    radii = [radius, radius, radius]
    extra_structures = (
        [make_overlap_cube_structure([radius, radius / 2, radius / 2])] if overlap_cube else []
    )
    base_sim, fom = make_base_simulation(
        radii=radii, extra_structures=extra_structures, is_2d=is_2d
    )

    center = [0.0, 0.0, 0.0]
    center_params = anp.array(center)

    if parametrization == "radius":
        params0 = anp.array([radius])
        geometry_factory = make_native_sphere_geometry
        objective_suffix = "radius"
    else:
        params0 = center_params

        def geometry_factory(params, _unused_center, radius_fixed=radius):
            return td.Sphere(center=tuple(params), radius=radius_fixed)

        objective_suffix = "center"

    native_objective = make_objective(
        geometry_factory,
        center,
        f"native_sphere_scale_{radius_scale}_cube_{overlap_cube}_param_{objective_suffix}_dim_{'2d' if is_2d else '3d'}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )
    native_objective_fd = make_objective(
        geometry_factory,
        center,
        f"native_sphere_fd_scale_{radius_scale}_cube_{overlap_cube}_param_{objective_suffix}_dim_{'2d' if is_2d else '3d'}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    _, native_grad = value_and_grad(native_objective)([params0])
    native_grad = np.squeeze(np.asarray(native_grad, dtype=float))

    fd_grad = finite_difference_params(native_objective_fd, params0, FINITE_DIFF_STEP_NATIVE)

    print(
        "native radius scale",
        radius_scale,
        "overlap_cube",
        overlap_cube,
        "parametrization",
        parametrization,
        "is_2d",
        is_2d,
    )
    print("native_grad\t", native_grad.tolist())
    print("fd_grad\t\t", fd_grad.tolist())

    if parametrization == "radius":
        abs_diff = float(np.abs(native_grad - fd_grad))
        rel_err = abs_diff / max(np.abs(native_grad), np.abs(fd_grad), 1e-12)
        print(
            f"Native sphere FD vs. Adjoint absolute diff: {abs_diff:.3e}, "
            f"relative error: {float(get_static(rel_err)):.3e}"
        )
        assert rel_err < 1e-1, (
            f"Native sphere gradients mismatch: abs_diff={abs_diff:.3e}, "
            f"rel_err={float(get_static(rel_err)):.3e}, native_grad={native_grad.tolist()}, "
            f"fd_grad={fd_grad.tolist()}"
        )
        grad_metric = rel_err
    else:
        grad_angle_deg = angled_overlap_deg(native_grad, fd_grad)
        print(
            f"Native sphere FD vs. Adjoint angle overlap: {grad_angle_deg:.3f}° "
            f"(threshold = {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG}°)",
        )
        assert grad_angle_deg < ANGLE_OVERLAP_FD_ADJ_THRESH_DEG, (
            f"FD–adjoint angle overlap too large: {grad_angle_deg:.3f}° "
            f"(threshold {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG}°)"
        )
        grad_metric = grad_angle_deg

    if SAVE_OUTPUT_DATA:
        np.savez(
            numerical_case_dir
            / (
                f"native_sphere_gradients_scale_{radius_scale}_cube_{overlap_cube}"
                f"_param_{parametrization}_dim_{'2d' if is_2d else '3d'}.npz"
            ),
            native_grad=native_grad,
            fd_grad=fd_grad,
            grad_metric=np.array([grad_metric], dtype=float),
        )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "radius_factor",
    (
        0.25,
        0.5,
        1,
    ),
)
def test_sphere_cylinder_grads_match_2d(radius_factor, tmp_path, numerical_case_dir):
    """Ensure 2D sphere gradients equal those from an equivalent Cylinder cross section."""
    radius = radius_factor * SPHERE_RADIUS_UM
    params0 = anp.array([radius, 0.0, 0.0, 0.0])
    radii = [radius, radius, radius]
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=None, is_2d=True)

    def make_param_sphere_geometry(params, _center_unused):
        rad = params[0]
        center = tuple(params[1:4])
        return td.Sphere(center=center, radius=rad)

    def make_param_cylinder_geometry(params, _center_unused):
        rad = params[0]
        center_x, center_y, center_z = params[1:4]
        plane_value = 0.0
        rad_plane_sq = rad**2 - (center_y - plane_value) ** 2
        rad_plane = anp.sqrt(anp.maximum(rad_plane_sq, 1e-15))
        cyl_center = (center_x, plane_value, center_z)
        return td.Cylinder(center=cyl_center, radius=rad_plane, length=1.0, axis=1)

    sphere_objective = make_objective(
        make_param_sphere_geometry,
        [0.0, 0.0, 0.0],
        "sphere_parametric_2d",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )
    cylinder_objective = make_objective(
        make_param_cylinder_geometry,
        [0.0, 0.0, 0.0],
        "cylinder_parametric_2d",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )

    sphere_val, sphere_grad = value_and_grad(sphere_objective)([params0])
    cylinder_val, cylinder_grad = value_and_grad(cylinder_objective)([params0])
    sphere_grad = np.squeeze(np.asarray(sphere_grad, dtype=float))
    cylinder_grad = np.squeeze(np.asarray(cylinder_grad, dtype=float))

    center_indices = [1, 3]
    center_angle = angled_overlap_deg(sphere_grad[center_indices], cylinder_grad[center_indices])

    print("sphere_val\t", sphere_val.tolist())
    print("cylinder_val\t", cylinder_val.tolist())

    print("sphere_grad\t", sphere_grad.tolist())
    print("cylinder_grad\t", cylinder_grad.tolist())
    print(f"Sphere vs Cylinder center gradient angle: {center_angle:.6f}°")
    np.testing.assert_allclose(
        sphere_grad[center_indices], cylinder_grad[center_indices], rtol=8e-2, atol=5e-3
    )

    radius_rel_err = abs(sphere_grad[0] - cylinder_grad[0]) / max(abs(cylinder_grad[0]), 1e-12)
    assert radius_rel_err < 0.08, (
        f"Sphere vs cylinder radius gradient mismatch: "
        f"{sphere_grad[0]} vs {cylinder_grad[0]} (relative error {radius_rel_err:.3f})"
    )

    if SAVE_OUTPUT_DATA:
        np.savez(
            numerical_case_dir / "sphere_vs_cylinder_gradients_2d.npz",
            sphere_grad=sphere_grad,
            cylinder_grad=cylinder_grad,
            center_angle_deg=np.array([center_angle], dtype=float),
        )


@pytest.mark.numerical
@pytest.mark.parametrize("radius_scale", (0.5, 1, 2))
@pytest.mark.parametrize("overlap_cube", (True, False))
@pytest.mark.parametrize("is_2d", (False, True))
def test_native_sphere_fd_step_sweep_ref(
    tmp_path, radius_scale, overlap_cube, is_2d, numerical_case_dir
):
    """FD step sweep for native sphere with autograd reference."""
    radius = SPHERE_RADIUS_UM * radius_scale
    params0 = anp.array([radius])

    radii = [radius, radius, radius]
    extra_structures = [make_overlap_cube_structure(radii)] if overlap_cube else []
    base_sim, fom = make_base_simulation(
        radii=radii, extra_structures=extra_structures, is_2d=is_2d
    )

    center = [0.0, 0.0, 0.0]

    # FD objective
    native_objective_fd = make_objective(
        make_native_sphere_geometry,
        center,
        f"native_sphere_fd_step_sweep_{radius_scale}_cube_{overlap_cube}_dim_{'2d' if is_2d else '3d'}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    # Autograd objective (reference)
    native_objective_autograd = make_objective(
        make_native_sphere_geometry,
        center,
        f"native_sphere_autograd_ref_{radius_scale}_cube_{overlap_cube}_dim_{'2d' if is_2d else '3d'}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=True,
    )

    # ---- Autograd gradient ----
    _, autograd_grad = value_and_grad(native_objective_autograd)([params0])
    autograd_grad = float(np.squeeze(np.asarray(autograd_grad, dtype=float)))
    print(
        f"native autograd gradient (radius_scale={radius_scale}, overlap_cube={overlap_cube}, "
        f"is_2d={is_2d}): {autograd_grad}"
    )

    # ---- Finite-difference sweep ----
    min_log = -4
    max_log = -1
    n = (max_log - min_log + 1) * 2 + 1
    steps = np.logspace(min_log, max_log, num=n)

    fd_grads = finite_difference_params_step_batch(native_objective_fd, params0, steps)
    fd_grads = np.asarray(fd_grads, dtype=float)

    for step, grad in zip(steps, fd_grads):
        print(
            f"native finite difference step {step:.1e}: gradient {grad.tolist()} "
            f"cube={overlap_cube} radius_scale={radius_scale} is_2d={is_2d}"
        )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 4))
    # FD curve
    ax.plot(steps, fd_grads[:, 0], marker="o", label="radius (FD)")
    # Autograd reference line in same color
    ax.axhline(
        autograd_grad,
        color=ax.get_lines()[-1].get_color(),
        linestyle="--",
        alpha=0.7,
        label="radius (autograd)",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step (µm)")
    ax.set_ylabel("Gradient value")
    ax.set_title(
        f"Native sphere FD vs autograd (radius_scale={radius_scale}, overlap_cube={overlap_cube})"
    )
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig_path = (
        numerical_case_dir
        / f"rad_{radius_scale}_cube_{overlap_cube}_dim_{'2d' if is_2d else '3d'}.png"
    )
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    np.savez(
        numerical_case_dir
        / f"rad_{radius_scale}_cube_{overlap_cube}_dim_{'2d' if is_2d else '3d'}.npz",
        steps=steps,
        gradients=fd_grads,
        autograd_grad=np.array([autograd_grad], dtype=float),
    )
