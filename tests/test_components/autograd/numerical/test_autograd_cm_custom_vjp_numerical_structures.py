"""ComponentModeler tests for custom_vjp and numerical_structures gradient behavior."""

from __future__ import annotations

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
import trimesh
import xarray as xr
from pydantic import BaseModel

import tidy3d as td
from tidy3d.plugins.smatrix import ComponentModeler, Port
from tidy3d.plugins.smatrix.run import _run_local
from tidy3d.web.api.autograd.types import CustomVJPConfig, NumericalStructureConfig

from .numerical_derivative_helpers import compute_ring_vjp
from .numerical_test_helpers import (
    EvaluationData,
    GradientComparisonDiagnostics,
    MetricGroups,
    case_identity_from_parameters,
    case_identity_id,
    evaluate_gradient_angle_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)

PLOT_FD_ADJ_COMPARISON = False
LOCAL_GRADIENT = True
VERBOSE = False

OVERLAP_ERROR_THRESHOLD_DEG = 10.0

ADJOINT_PERMITTIVITY = 1.5**2

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

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

    def finite_difference_gradient(perturb_up, perturb_down, derivative_info_):
        eps_up = derivative_info.updated_epsilon(perturb_up)
        eps_down = derivative_info.updated_epsilon(perturb_down)
        eps_grad = (eps_up - eps_down) / (2 * step_size)

        derivative_info_custom_medium = derivative_info_.updated_copy(**update_kwargs)

        custom_medium = td.CustomMedium(permittivity=xr.ones_like(eps_grad.isel(f=0, drop=True)))
        vjps_custom_medium = custom_medium._compute_derivatives(derivative_info_custom_medium)

        total_grad = np.real(np.sum(eps_grad.sum("f").data * vjps_custom_medium[("permittivity",)]))

        return total_grad

    vjps = {}
    for path in derivative_info.paths:
        if path[0:2] == (
            "geometry",
            "radius",
        ):
            sphere_up = sphere.updated_copy(radius=sphere.radius + step_size)
            sphere_down = sphere.updated_copy(radius=sphere.radius - step_size)
            vjps[path] = finite_difference_gradient(sphere_up, sphere_down, derivative_info)
        elif path[0:2] == ("geometry", "center"):
            if len(path) == 2:
                center_indices = (0, 1, 2)
            else:
                _, center_index = path[1:]
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

            vjps[path] = vjp_result if len(path) == 2 else vjp_result[0]

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


def vjp_ring(parameters, derivative_info, derivative_helper):
    return compute_ring_vjp(
        parameters=parameters,
        derivative_info=derivative_info,
        create_ring_fn=create_ring,
        derivative_helper=derivative_helper,
    )


def create_objective_function(create_sim_base, adj_wvl_um):
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

            simulation_dict[f"numerical_custom_vjp_testing_{idx}"] = sim_with_sphere.copy()
            geom_dict[f"numerical_custom_vjp_testing_{idx}"] = geom_parameters

        sim_data = {}
        for key, sim_val in simulation_dict.items():
            modeler = ComponentModeler(
                simulation=sim_val,
                ports=ports,
                freqs=[td.C_0 / adj_wvl_um],
            )

            ring_numerical_structure = NumericalStructureConfig(
                create=create_ring,
                compute_derivatives=vjp_ring,
                parameters=geom_dict[key][4:],
            )

            custom_vjp_single = CustomVJPConfig(
                structure=2,
                compute_derivatives=vjp_sphere,
            )

            sim_data[key] = _run_local(
                modeler,
                local_gradient=LOCAL_GRADIENT,
                verbose=VERBOSE,
                custom_vjp=custom_vjp_single,
                numerical_structures=ring_numerical_structure,
            )

        objective_vals = []
        for idx in range(len(geom_parameters_lists)):
            smatrix = sim_data[f"numerical_custom_vjp_testing_{idx}"]
            objective_vals.append(np.sum(np.abs(smatrix.smatrix().values) ** 2))

        if len(geom_parameters_lists) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


class CMCustomVJPNumericalStructureCaseIdentity(BaseModel):
    """Semantic identity for one ComponentModeler combined-gradient case."""

    mesh_wvl_um: float
    adj_wvl_um: float


class CMCustomVJPNumericalStructureTestParameters(CMCustomVJPNumericalStructureCaseIdentity):
    """Full parameter bundle for one ComponentModeler combined-gradient test invocation."""

    test_number: int


def _case_identity(
    test_parameters: CMCustomVJPNumericalStructureTestParameters,
) -> CMCustomVJPNumericalStructureCaseIdentity:
    return case_identity_from_parameters(CMCustomVJPNumericalStructureCaseIdentity, test_parameters)


mesh_wvls_um = [1.5]
adj_wvls_um = [1.5]

test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    test_parameters.append(
        CMCustomVJPNumericalStructureTestParameters(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            test_number=test_number,
        )
    )

    test_number += 1


def _collect_cm_custom_vjp_numerical_structure_evaluation_data(
    test_parameters: CMCustomVJPNumericalStructureTestParameters,
    rng: np.random.Generator,
) -> EvaluationData:
    mesh_wvl_um = test_parameters.mesh_wvl_um
    adj_wvl_um = test_parameters.adj_wvl_um

    objective = create_objective_function(
        lambda mesh_wvl_um=mesh_wvl_um, adj_wvl_um=adj_wvl_um: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
        ),
        adj_wvl_um,
    )

    obj_val_and_grad = ag.value_and_grad(objective)

    sphere_init = [
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
    ]

    ring_init_mesh_wvl_factor = [0.15, 0.30, 0.2]
    ring_init = [r * mesh_wvl_um for r in ring_init_mesh_wvl_factor]

    geom_init = sphere_init + ring_init

    _obj, adj_grad = obj_val_and_grad([geom_init])
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

    return {
        "fd_grad": np.asarray(fd_grad, dtype=float),
        "adj_grad": np.asarray(adj_grad, dtype=float),
        "sphere_init": np.asarray(sphere_init, dtype=float),
        "ring_init": np.asarray(ring_init, dtype=float),
        "fd_step": np.asarray(fd_step, dtype=float),
    }


def _evaluate_cm_custom_vjp_numerical_structure_evaluation_data(
    evaluation_data: EvaluationData,
) -> MetricGroups:
    return evaluate_gradient_angle_agreement(
        np.asarray(evaluation_data["fd_grad"], dtype=float),
        np.asarray(evaluation_data["adj_grad"], dtype=float),
        angle_threshold_deg=OVERLAP_ERROR_THRESHOLD_DEG,
    )


def _print_cm_custom_vjp_numerical_structure_summary(
    test_parameters: CMCustomVJPNumericalStructureTestParameters,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    if not VERBOSE:
        return
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{test_parameters.test_number}")
    print(f"Evaluation mode: {mode_label}")
    print(
        f"Mesh and adjoint wavelengths: {test_parameters.mesh_wvl_um}, {test_parameters.adj_wvl_um}"
    )
    print(f"RMS Error: {diagnostics['rms_error']}")
    print(f"Gradient overlap (deg): {diagnostics['gradient_overlap_deg']}")
    print(f"FD, Adj magnitudes: {diagnostics['reference_mag']}, {diagnostics['adjoint_mag']}")
    print("-" * 20)
    print("\n" * 3)


def _plot_cm_custom_vjp_numerical_structure_comparison(
    evaluation_data: EvaluationData,
) -> None:
    if PLOT_FD_ADJ_COMPARISON:
        adj_grad = np.asarray(evaluation_data["adj_grad"], dtype=float)
        fd_grad = np.asarray(evaluation_data["fd_grad"], dtype=float)
        plt.plot(adj_grad, color="g", linewidth=2.0)
        plt.plot(fd_grad, color="b", linewidth=1.5, linestyle="--")
        plt.legend(["Adjoint", "Finite difference"])
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "test_parameters",
    test_parameters,
    ids=lambda params: case_identity_id(_case_identity(params), prefix="cm-custom-vjp-num-struct"),
)
def test_finite_difference_numerical_structures(
    request: pytest.FixtureRequest,
    test_parameters: CMCustomVJPNumericalStructureTestParameters,
    rng: np.random.Generator,
    numerical_case_dir,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Compare combined custom-VJP/numerical-structure gradients against finite differences."""
    case_identity = _case_identity(test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: (
            _collect_cm_custom_vjp_numerical_structure_evaluation_data(test_parameters, rng)
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_cm_custom_vjp_numerical_structure_evaluation_data(evaluation_data)
    )
    _print_cm_custom_vjp_numerical_structure_summary(
        test_parameters, diagnostics, eval_only=numerical_eval_only
    )
    _plot_cm_custom_vjp_numerical_structure_comparison(evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Adjoint and finite difference gradients misaligned; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
