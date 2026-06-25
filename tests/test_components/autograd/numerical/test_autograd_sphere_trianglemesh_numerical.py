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
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tests.test_components.autograd.numerical.test_autograd_box_polyslab_numerical import (
    angled_overlap_deg,
)
from tests.test_components.autograd.test_autograd_triangle_mesh import subdivide_triangles
from tidy3d import config
from tidy3d.components.autograd import get_static
from tidy3d.components.geometry.primitives import _base_icosahedron

from .numerical_test_helpers import (
    EvaluationData,
    case_identity_id,
    condition_metric,
    evaluate_allclose_agreement,
    finalize_result,
    gradient_angle_deg,
    load_or_collect_evaluation_data,
)
from .result_models import Metric

config.local_cache.enabled = True

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
SAVE_OUTPUT_DATA = False
ANGLE_OVERLAP_FD_ADJ_THRESH_DEG = 10.0
NATIVE_SPHERE_RADIUS_REL_ERR_THRESH = 0.1
FD_STEP_SWEEP_REL_ERR_THRESH = NATIVE_SPHERE_RADIUS_REL_ERR_THRESH
VERTEX_FD_STEP = 1e-3
FINITE_DIFF_STEP = 1e-2
FINITE_DIFF_STEP_NATIVE = 1e-3
GRID_STEPS_PER_WVL = 40
FINE_GRID_STEPS_PER_WVL = 60
FINE_GRID_MAX_RADIUS_UM = 0.25 * WL_UM
td.config.adjoint.points_per_wavelength = 10
measure_flux_spec = False
NATIVE_SPHERE_RADIUS_SCALES = (0.25, 0.5, 1)
NATIVE_SPHERE_STEP_SWEEP_RADIUS_SCALES = (*NATIVE_SPHERE_RADIUS_SCALES, 2)

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
    fixed_grid_spec: td.GridSpec | None = None,
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
        if fixed_grid_spec is None:
            grid_spec = td.GridSpec.auto(
                min_steps_per_wvl=GRID_STEPS_PER_WVL, override_structures=[structure]
            )
        else:
            grid_spec = fixed_grid_spec
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
    fixed_grid_spec: td.GridSpec | None = None,
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
            fixed_grid_spec=fixed_grid_spec,
        )

    return objective


def fixed_grid_spec_for_parameters(
    make_geometry: Callable[[anp.ndarray, Sequence[float]], td.Geometry],
    center: Sequence[float],
    params: anp.ndarray,
    base_sim: td.Simulation,
    *,
    grid_steps_per_wvl: int = GRID_STEPS_PER_WVL,
) -> td.GridSpec:
    """Resolve the unperturbed auto grid and freeze it for finite-difference probes."""
    geometry = make_geometry(params, center)
    structure = td.Structure(geometry=geometry, medium=td.Medium(permittivity=PERMITTIVITY))
    base_structures = list(getattr(base_sim, "structures", ()))
    structures = [structure, *base_structures]
    grid_spec = td.GridSpec.auto(
        min_steps_per_wvl=grid_steps_per_wvl,
        override_structures=[structure],
    )
    sim = base_sim.updated_copy(structures=structures, grid_spec=grid_spec, validate=True)
    return td.GridSpec.from_grid(sim.grid)


class TriangleSphereCaseIdentity(BaseModel):
    """Semantic identity for one TriangleMesh sphere gradient case."""

    scale_factor: float
    scale_axis: int
    overlap_cube: bool


class NativeSphereCaseIdentity(BaseModel):
    """Semantic identity for one native sphere finite-difference comparison."""

    radius_scale: float
    overlap_cube: bool
    parametrization: str
    is_2d: bool


class SphereCylinder2DCaseIdentity(BaseModel):
    """Semantic identity for one sphere-vs-cylinder 2D comparison."""

    radius_factor: float


class NativeSphereStepSweepCaseIdentity(BaseModel):
    """Semantic identity for one native sphere FD-step sweep."""

    radius_scale: float
    overlap_cube: bool
    is_2d: bool


TRIANGLE_SPHERE_CASES = [
    TriangleSphereCaseIdentity(
        scale_factor=scale_factor,
        scale_axis=scale_axis,
        overlap_cube=overlap_cube,
    )
    for scale_factor in (1,)
    for scale_axis in (0,)
    for overlap_cube in (False,)
]

TRIANGLE_SPHERE_STEP_SWEEP_CASES = [
    TriangleSphereCaseIdentity(
        scale_factor=scale_factor,
        scale_axis=scale_axis,
        overlap_cube=overlap_cube,
    )
    for scale_factor in SCALE_FACTORS
    for scale_axis in SCALE_AXES
    for overlap_cube in (False, True)
]

NATIVE_SPHERE_CASES = [
    NativeSphereCaseIdentity(
        radius_scale=radius_scale,
        overlap_cube=overlap_cube,
        parametrization=parametrization,
        is_2d=is_2d,
    )
    for radius_scale in NATIVE_SPHERE_RADIUS_SCALES
    for overlap_cube in (False, True)
    for parametrization in ("radius", "center")
    for is_2d in (False,)
]

SPHERE_CYLINDER_2D_CASES = [
    SphereCylinder2DCaseIdentity(radius_factor=radius_factor) for radius_factor in (0.25, 0.5, 1)
]

NATIVE_SPHERE_STEP_SWEEP_CASES = [
    NativeSphereStepSweepCaseIdentity(
        radius_scale=radius_scale,
        overlap_cube=overlap_cube,
        is_2d=is_2d,
    )
    for radius_scale in NATIVE_SPHERE_STEP_SWEEP_RADIUS_SCALES
    for overlap_cube in (True, False)
    for is_2d in (False, True)
]


def _native_sphere_grid_steps_per_wvl(case: NativeSphereCaseIdentity) -> int:
    """Use a finer grid for small native-sphere radius comparisons."""
    radius = SPHERE_RADIUS_UM * case.radius_scale
    if case.parametrization == "radius" and radius <= FINE_GRID_MAX_RADIUS_UM:
        return FINE_GRID_STEPS_PER_WVL

    return GRID_STEPS_PER_WVL


def _triangle_sphere_setup(case: TriangleSphereCaseIdentity):
    initial_params = [SPHERE_RADIUS_UM, SPHERE_RADIUS_UM, SPHERE_RADIUS_UM]
    params0 = anp.array(initial_params)
    radii = initial_params.copy()
    radii[case.scale_axis] *= case.scale_factor
    extra_structures = [make_overlap_cube_structure(radii)] if case.overlap_cube else []
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=extra_structures)
    center = [0.0, 0.0, 0.0]
    part_make_geom = lambda p, c: make_sphere_triangle_geometry(
        p,
        c,
        case.scale_factor,
        case.scale_axis,
    )
    return params0, radii, base_sim, fom, center, part_make_geom


def _relative_error(actual: np.ndarray | float, desired: np.ndarray | float) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    desired_arr = np.asarray(desired, dtype=float)
    abs_diff = np.abs(actual_arr - desired_arr)
    scale = np.maximum(np.maximum(np.abs(actual_arr), np.abs(desired_arr)), 1e-12)
    return float(np.max(abs_diff / scale))


def _evaluate_step_sweep_data(
    evaluation_data: EvaluationData,
    *,
    gradient_key: str,
    autograd_key: str,
    relative_error_threshold: float = FD_STEP_SWEEP_REL_ERR_THRESH,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    fd_grads = np.asarray(evaluation_data[gradient_key], dtype=float)
    autograd_grad = np.asarray(evaluation_data[autograd_key], dtype=float)
    fd_grads_2d = np.atleast_2d(fd_grads)
    rel_errors = np.asarray(
        [_relative_error(fd_grad, autograd_grad) for fd_grad in fd_grads_2d],
        dtype=float,
    )
    best_rel_error = float(np.min(rel_errors)) if rel_errors.size else np.inf
    regression_metrics = [
        condition_metric("fd_gradients_finite", bool(np.all(np.isfinite(fd_grads)))),
        condition_metric("autograd_gradient_finite", bool(np.all(np.isfinite(autograd_grad)))),
        Metric(
            name="best_fd_autograd_relative_error",
            observed=best_rel_error,
            expected=relative_error_threshold,
            comparator="lte",
        ),
    ]
    observation_metrics: list[Metric] = []
    diagnostics = {"best_fd_autograd_relative_error": best_rel_error}
    return regression_metrics, observation_metrics, diagnostics


def _save_triangle_step_sweep_plot(
    case: TriangleSphereCaseIdentity,
    numerical_case_dir,
    evaluation_data: EvaluationData,
) -> None:
    steps = np.asarray(evaluation_data["steps"], dtype=float)
    fd_grads = np.asarray(evaluation_data["fd_grads"], dtype=float)
    autograd_grad = np.asarray(evaluation_data["autograd_grad"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, label in enumerate(["radius_x", "radius_y", "radius_z"]):
        ax.plot(steps, fd_grads[:, idx], marker="o", label=label)
        ax.axhline(
            autograd_grad[idx],
            color=ax.get_lines()[-1].get_color(),
            linestyle="--",
            alpha=0.7,
            label=f"{label} (autograd)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step (um)")
    ax.set_ylabel("Gradient value")
    ax.set_title("FD gradients vs. step size")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig_path = numerical_case_dir / (
        f"fd_step_sweep_scale_{case.scale_factor}_axis_{case.scale_axis}"
        f"_cube_{case.overlap_cube}.png"
    )
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def _save_native_step_sweep_plot(
    case: NativeSphereStepSweepCaseIdentity,
    numerical_case_dir,
    evaluation_data: EvaluationData,
) -> None:
    steps = np.asarray(evaluation_data["steps"], dtype=float)
    fd_grads = np.asarray(evaluation_data["fd_grads"], dtype=float)
    autograd_grad = float(np.asarray(evaluation_data["autograd_grad"], dtype=float).reshape(-1)[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, fd_grads[:, 0], marker="o", label="radius (FD)")
    ax.axhline(
        autograd_grad,
        color=ax.get_lines()[-1].get_color(),
        linestyle="--",
        alpha=0.7,
        label="radius (autograd)",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step (um)")
    ax.set_ylabel("Gradient value")
    ax.set_title(
        f"Native sphere FD vs autograd (radius_scale={case.radius_scale}, "
        f"overlap_cube={case.overlap_cube})"
    )
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig_path = (
        numerical_case_dir
        / f"rad_{case.radius_scale}_cube_{case.overlap_cube}_dim_{'2d' if case.is_2d else '3d'}.png"
    )
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    TRIANGLE_SPHERE_CASES,
    ids=lambda case: case_identity_id(case, prefix="triangle-sphere"),
)
def test_sphere_triangles_match_fd(
    request: pytest.FixtureRequest,
    case: TriangleSphereCaseIdentity,
    numerical_case_dir,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """
    Compares FD gradients with gradients from _compute_derivatives in TriangleMesh.
    Note that FD gradients are very noise which is why there will be some failing tests with a fixed FD-step
    """
    if case.scale_factor == 1 and case.scale_axis > 0:
        pytest.skip("Skipping duplicate test.")

    def collect() -> EvaluationData:
        params0, _radii, base_sim, fom, center, part_make_geom = _triangle_sphere_setup(case)
        sim_path_dir = numerical_case_dir / "simulations"
        sim_path_dir.mkdir(parents=True, exist_ok=True)
        triangle_objective = make_objective(
            part_make_geom,
            center,
            f"sphere_mesh_{case.scale_factor}_axis_{case.scale_axis}_cube_{case.overlap_cube}",
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
        )
        triangle_objective_fd = make_objective(
            part_make_geom,
            center,
            f"sphere_mesh_fd_{case.scale_factor}_axis_{case.scale_axis}_cube_{case.overlap_cube}",
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=False,
        )

        _, triangle_grad = value_and_grad(triangle_objective)([params0])
        triangle_grad = np.squeeze(np.asarray(triangle_grad, dtype=float))
        fd_grad = finite_difference_params(triangle_objective_fd, params0, FINITE_DIFF_STEP)
        return {
            "triangle_grad": np.asarray(triangle_grad, dtype=float),
            "fd_grad": np.asarray(fd_grad, dtype=float),
        }

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case,
        collect_evaluation_data=collect,
    )
    triangle_grad = np.asarray(evaluation_data["triangle_grad"], dtype=float)
    fd_grad = np.asarray(evaluation_data["fd_grad"], dtype=float)

    print("scale", case.scale_factor, "axis", case.scale_axis, "overlap_cube", case.overlap_cube)
    print("triangle_grad\t", triangle_grad.tolist())
    print("fd_grad\t\t", fd_grad.tolist())

    mesh_fd_overlap = gradient_angle_deg(triangle_grad, fd_grad)
    print(
        f"TriangleMesh FD vs. Adjoint angle overlap: {mesh_fd_overlap:.3f} deg "
        f"(threshold = {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG} deg)"
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=[
            Metric(
                name="mesh_fd_overlap_deg",
                observed=mesh_fd_overlap,
                expected=ANGLE_OVERLAP_FD_ADJ_THRESH_DEG,
                comparator="lt",
            )
        ],
        observation_metrics=[],
        failure_message=(
            "TriangleMesh sphere gradient mismatch; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
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

    # Optional angles for log inspection.
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
@pytest.mark.parametrize(
    "case",
    TRIANGLE_SPHERE_STEP_SWEEP_CASES,
    ids=lambda case: case_identity_id(case, prefix="triangle-sphere-step"),
)
def test_triangle_sphere_fd_step_sweep_ref(
    request: pytest.FixtureRequest,
    case: TriangleSphereCaseIdentity,
    numerical_case_dir,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    def collect() -> EvaluationData:
        params0, _radii, base_sim, fom, center, part_make_geom = _triangle_sphere_setup(case)
        sim_path_dir = numerical_case_dir / "simulations"
        sim_path_dir.mkdir(parents=True, exist_ok=True)
        triangle_objective_fd = make_objective(
            part_make_geom,
            center,
            "sphere_mesh_fd_step_sweep",
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=False,
        )
        triangle_objective_autograd = make_objective(
            part_make_geom,
            center,
            "sphere_mesh_autograd_ref",
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=True,
        )
        _, autograd_grad = value_and_grad(triangle_objective_autograd)([params0])
        autograd_grad = np.squeeze(np.asarray(autograd_grad, dtype=float))
        steps = np.logspace(-4, -1, num=12)
        fd_grads = finite_difference_params_step_batch(triangle_objective_fd, params0, steps)
        return {
            "steps": np.asarray(steps, dtype=float),
            "fd_grads": np.asarray(fd_grads, dtype=float),
            "autograd_grad": np.asarray(autograd_grad, dtype=float),
        }

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case,
        collect_evaluation_data=collect,
    )
    regression_metrics, observation_metrics, _diagnostics = _evaluate_step_sweep_data(
        evaluation_data,
        gradient_key="fd_grads",
        autograd_key="autograd_grad",
    )
    _save_triangle_step_sweep_plot(case, numerical_case_dir, evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "TriangleMesh sphere FD-step sweep produced invalid gradients or failed "
            "to match the autograd reference; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    NATIVE_SPHERE_CASES,
    ids=lambda case: case_identity_id(case, prefix="native-sphere"),
)
def test_native_sphere_match_fd(
    request: pytest.FixtureRequest,
    case: NativeSphereCaseIdentity,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    """
    Compares FD gradients with gradients from _compute_derivatives in Sphere.
    Note that FD gradients are very noise which is why there will be some failing tests with a fixed FD-step.
    Currently, numerical tests fail for 2D as FD gradients are very shaky.
    """

    def collect() -> EvaluationData:
        radius = SPHERE_RADIUS_UM * case.radius_scale
        radii = [radius, radius, radius]
        extra_structures = (
            [make_overlap_cube_structure([radius, radius / 2, radius / 2])]
            if case.overlap_cube
            else []
        )
        base_sim, fom = make_base_simulation(
            radii=radii, extra_structures=extra_structures, is_2d=case.is_2d
        )

        center = [0.0, 0.0, 0.0]
        center_params = anp.array(center)

        if case.parametrization == "radius":
            params0 = anp.array([radius])
            geometry_factory = make_native_sphere_geometry
            objective_suffix = "radius"
        else:
            params0 = center_params

            def geometry_factory(params, _unused_center, radius_fixed=radius):
                return td.Sphere(center=tuple(params), radius=radius_fixed)

            objective_suffix = "center"

        sim_path_dir = numerical_case_dir / "simulations"
        sim_path_dir.mkdir(parents=True, exist_ok=True)
        grid_steps_per_wvl = _native_sphere_grid_steps_per_wvl(case)
        fixed_grid_spec = fixed_grid_spec_for_parameters(
            geometry_factory,
            center,
            params0,
            base_sim,
            grid_steps_per_wvl=grid_steps_per_wvl,
        )
        native_objective = make_objective(
            geometry_factory,
            center,
            (
                f"native_sphere_scale_{case.radius_scale}_cube_{case.overlap_cube}"
                f"_param_{objective_suffix}_dim_{'2d' if case.is_2d else '3d'}"
            ),
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
            fixed_grid_spec=fixed_grid_spec,
        )
        native_objective_fd = make_objective(
            geometry_factory,
            center,
            (
                f"native_sphere_fd_scale_{case.radius_scale}_cube_{case.overlap_cube}"
                f"_param_{objective_suffix}_dim_{'2d' if case.is_2d else '3d'}"
            ),
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=False,
            fixed_grid_spec=fixed_grid_spec,
        )

        _, native_grad = value_and_grad(native_objective)([params0])
        native_grad = np.squeeze(np.asarray(native_grad, dtype=float))
        fd_grad = finite_difference_params(native_objective_fd, params0, FINITE_DIFF_STEP_NATIVE)
        return {
            "native_grad": np.asarray(native_grad, dtype=float),
            "fd_grad": np.asarray(fd_grad, dtype=float),
            "radius_um": np.asarray(radius, dtype=float),
            "grid_steps_per_wvl": np.asarray(grid_steps_per_wvl, dtype=float),
        }

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case,
        collect_evaluation_data=collect,
    )
    native_grad = np.asarray(evaluation_data["native_grad"], dtype=float)
    fd_grad = np.asarray(evaluation_data["fd_grad"], dtype=float)
    radius_um = float(
        np.asarray(
            evaluation_data.get("radius_um", SPHERE_RADIUS_UM * case.radius_scale),
            dtype=float,
        )
    )
    grid_steps_per_wvl = float(
        np.asarray(evaluation_data.get("grid_steps_per_wvl", GRID_STEPS_PER_WVL), dtype=float)
    )

    print(
        "native radius scale",
        case.radius_scale,
        "overlap_cube",
        case.overlap_cube,
        "parametrization",
        case.parametrization,
        "is_2d",
        case.is_2d,
    )
    print("radius_um", radius_um)
    print("grid_steps_per_wvl", grid_steps_per_wvl)
    print("native_grad\t", native_grad.tolist())
    print("fd_grad\t\t", fd_grad.tolist())

    if case.parametrization == "radius":
        abs_diff = float(np.max(np.abs(native_grad - fd_grad)))
        rel_err = _relative_error(native_grad, fd_grad)
        print(
            f"Native sphere FD vs. Adjoint absolute diff: {abs_diff:.3e}, "
            f"relative error: {float(get_static(rel_err)):.3e}"
        )
        regression_metrics = [
            Metric(
                name="native_fd_relative_error",
                observed=rel_err,
                expected=NATIVE_SPHERE_RADIUS_REL_ERR_THRESH,
                comparator="lt",
            )
        ]
        observation_metrics = [
            Metric(
                name="native_fd_abs_diff",
                observed=abs_diff,
                expected=0.0,
                comparator="gte",
            )
        ]
    else:
        grad_angle_deg = gradient_angle_deg(native_grad, fd_grad)
        print(
            f"Native sphere FD vs. Adjoint angle overlap: {grad_angle_deg:.3f} deg "
            f"(threshold = {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG} deg)",
        )
        regression_metrics = [
            Metric(
                name="native_fd_angle_deg",
                observed=grad_angle_deg,
                expected=ANGLE_OVERLAP_FD_ADJ_THRESH_DEG,
                comparator="lt",
            )
        ]
        observation_metrics = []

    observation_metrics.append(
        Metric(
            name="radius_um",
            observed=radius_um,
            expected=0.0,
            comparator="gte",
        )
    )
    observation_metrics.append(
        Metric(
            name="grid_steps_per_wvl",
            observed=grid_steps_per_wvl,
            expected=float(GRID_STEPS_PER_WVL),
            comparator="gte",
        )
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Native sphere finite-difference comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    SPHERE_CYLINDER_2D_CASES,
    ids=lambda case: case_identity_id(case, prefix="sphere-cylinder-2d"),
)
def test_sphere_cylinder_grads_match_2d(
    request: pytest.FixtureRequest,
    case: SphereCylinder2DCaseIdentity,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    """Ensure 2D sphere gradients equal those from an equivalent Cylinder cross section."""

    def collect() -> EvaluationData:
        radius = case.radius_factor * SPHERE_RADIUS_UM
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

        sim_path_dir = numerical_case_dir / "simulations"
        sim_path_dir.mkdir(parents=True, exist_ok=True)
        sphere_objective = make_objective(
            make_param_sphere_geometry,
            [0.0, 0.0, 0.0],
            "sphere_parametric_2d",
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
        )
        cylinder_objective = make_objective(
            make_param_cylinder_geometry,
            [0.0, 0.0, 0.0],
            "cylinder_parametric_2d",
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
        )

        sphere_val, sphere_grad = value_and_grad(sphere_objective)([params0])
        cylinder_val, cylinder_grad = value_and_grad(cylinder_objective)([params0])
        return {
            "sphere_val": np.asarray(sphere_val, dtype=float),
            "cylinder_val": np.asarray(cylinder_val, dtype=float),
            "sphere_grad": np.squeeze(np.asarray(sphere_grad, dtype=float)),
            "cylinder_grad": np.squeeze(np.asarray(cylinder_grad, dtype=float)),
        }

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case,
        collect_evaluation_data=collect,
    )
    sphere_val = np.asarray(evaluation_data["sphere_val"], dtype=float)
    cylinder_val = np.asarray(evaluation_data["cylinder_val"], dtype=float)
    sphere_grad = np.asarray(evaluation_data["sphere_grad"], dtype=float)
    cylinder_grad = np.asarray(evaluation_data["cylinder_grad"], dtype=float)

    center_indices = [1, 3]
    center_angle = gradient_angle_deg(sphere_grad[center_indices], cylinder_grad[center_indices])

    print("sphere_val\t", sphere_val.tolist())
    print("cylinder_val\t", cylinder_val.tolist())

    print("sphere_grad\t", sphere_grad.tolist())
    print("cylinder_grad\t", cylinder_grad.tolist())
    print(f"Sphere vs Cylinder center gradient angle: {center_angle:.6f} deg")
    center_metrics, center_observations, _ = evaluate_allclose_agreement(
        sphere_grad[center_indices],
        cylinder_grad[center_indices],
        rtol=8e-2,
        atol=5e-3,
        metric_name="center_gradient_scaled_error",
    )

    radius_rel_err = abs(sphere_grad[0] - cylinder_grad[0]) / max(abs(cylinder_grad[0]), 1e-12)
    regression_metrics = [
        *center_metrics,
        Metric(
            name="radius_relative_error",
            observed=float(radius_rel_err),
            expected=0.08,
            comparator="lt",
        ),
    ]
    observation_metrics = [
        *center_observations,
        Metric(
            name="center_angle_deg",
            observed=center_angle,
            expected=0.0,
            comparator="gte",
        ),
    ]
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "2D sphere/cylinder gradient comparison failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    NATIVE_SPHERE_STEP_SWEEP_CASES,
    ids=lambda case: case_identity_id(case, prefix="native-sphere-step"),
)
def test_native_sphere_fd_step_sweep_ref(
    request: pytest.FixtureRequest,
    case: NativeSphereStepSweepCaseIdentity,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    """FD step sweep for native sphere with autograd reference."""

    def collect() -> EvaluationData:
        radius = SPHERE_RADIUS_UM * case.radius_scale
        params0 = anp.array([radius])
        radii = [radius, radius, radius]
        extra_structures = [make_overlap_cube_structure(radii)] if case.overlap_cube else []
        base_sim, fom = make_base_simulation(
            radii=radii, extra_structures=extra_structures, is_2d=case.is_2d
        )
        center = [0.0, 0.0, 0.0]
        sim_path_dir = numerical_case_dir / "simulations"
        sim_path_dir.mkdir(parents=True, exist_ok=True)
        fixed_grid_spec = fixed_grid_spec_for_parameters(
            make_native_sphere_geometry,
            center,
            params0,
            base_sim,
        )
        native_objective_fd = make_objective(
            make_native_sphere_geometry,
            center,
            (
                f"native_sphere_fd_step_sweep_{case.radius_scale}_cube_{case.overlap_cube}"
                f"_dim_{'2d' if case.is_2d else '3d'}"
            ),
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=False,
            fixed_grid_spec=fixed_grid_spec,
        )
        native_objective_autograd = make_objective(
            make_native_sphere_geometry,
            center,
            (
                f"native_sphere_autograd_ref_{case.radius_scale}_cube_{case.overlap_cube}"
                f"_dim_{'2d' if case.is_2d else '3d'}"
            ),
            base_sim,
            fom,
            sim_path_dir,
            local_gradient=True,
            fixed_grid_spec=fixed_grid_spec,
        )
        _, autograd_grad = value_and_grad(native_objective_autograd)([params0])
        autograd_grad = float(np.squeeze(np.asarray(autograd_grad, dtype=float)))
        min_log = -4
        max_log = -1
        n = (max_log - min_log + 1) * 2 + 1
        steps = np.logspace(min_log, max_log, num=n)
        steps = steps[steps < radius]
        fd_grads = finite_difference_params_step_batch(native_objective_fd, params0, steps)
        return {
            "steps": np.asarray(steps, dtype=float),
            "fd_grads": np.asarray(fd_grads, dtype=float),
            "autograd_grad": np.asarray([autograd_grad], dtype=float),
        }

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case,
        collect_evaluation_data=collect,
    )
    steps = np.asarray(evaluation_data["steps"], dtype=float)
    fd_grads = np.asarray(evaluation_data["fd_grads"], dtype=float)
    autograd_grad = float(np.asarray(evaluation_data["autograd_grad"], dtype=float).reshape(-1)[0])
    print(
        f"native autograd gradient (radius_scale={case.radius_scale}, "
        f"overlap_cube={case.overlap_cube}, is_2d={case.is_2d}): {autograd_grad}"
    )

    for step, grad in zip(steps, fd_grads):
        print(
            f"native finite difference step {step:.1e}: gradient {grad.tolist()} "
            f"cube={case.overlap_cube} radius_scale={case.radius_scale} is_2d={case.is_2d}"
        )

    regression_metrics, observation_metrics, _diagnostics = _evaluate_step_sweep_data(
        evaluation_data,
        gradient_key="fd_grads",
        autograd_key="autograd_grad",
    )
    _save_native_step_sweep_plot(case, numerical_case_dir, evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Native sphere FD-step sweep produced invalid gradients or failed to match "
            "the autograd reference; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
