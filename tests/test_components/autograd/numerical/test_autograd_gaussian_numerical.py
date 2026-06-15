# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

from pathlib import Path

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter

import tidy3d as td
import tidy3d.web as web

from .numerical_test_helpers import (
    EvalFn,
    EvaluationData,
    GradientComparisonDiagnostics,
    MetricGroups,
    case_identity_from_parameters,
    case_identity_id,
    evaluate_fd_adjoint_gradient_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)

PLOT_FD_ADJ_COMPARISON = False
NUM_FINITE_DIFFERENCE = 10
LOCAL_GRADIENT = True
VERBOSE = False

RMS_THRESHOLD = 0.25

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


FINITE_DIFF_PERM_SEED = 2.5**2
MESH_FACTOR_DESIGN = 30.0


class GaussianOverlapCaseIdentity(BaseModel):
    """Semantic identity for one Gaussian-overlap numerical case."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_type: str
    eval_fn_name: str
    monitor_side: str
    objective_direction: str
    pol_angle: float
    angle_theta: float
    angle_phi: float
    waist_distance_offset: float


class GaussianOverlapTestParameters(GaussianOverlapCaseIdentity):
    """Full parameter bundle for one Gaussian-overlap test invocation."""

    eval_fn: EvalFn
    test_number: int


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

    src_size = (*sim_size_um[0:2], 0)

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
            simulation_dict,
            path_dir=sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
            verbose=VERBOSE,
            lazy=False,
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

field_data_test_parameters: list[GaussianOverlapTestParameters] = []

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
                            GaussianOverlapTestParameters(
                                mesh_wvl_um=mesh_wvl_um,
                                adj_wvl_um=adj_wvl_um,
                                monitor_type=monitor_type_case["monitor_type"],
                                eval_fn=eval_fn,
                                eval_fn_name=eval_fn_names[eval_fn_idx],
                                monitor_side=monitor_side_case["monitor_side"],
                                objective_direction=monitor_side_case["objective_direction"],
                                pol_angle=angle_case["pol_angle"],
                                angle_theta=angle_case["angle_theta"],
                                angle_phi=angle_case["angle_phi"],
                                waist_distance_offset=waist_distance_offset,
                                test_number=test_number,
                            )
                        )

                        test_number += 1


def _case_identity(
    field_data_test_parameters: GaussianOverlapTestParameters,
) -> GaussianOverlapCaseIdentity:
    """Build the semantic case identity used for eval-only replay validation."""
    return case_identity_from_parameters(GaussianOverlapCaseIdentity, field_data_test_parameters)


def _collect_gaussian_overlap_evaluation_data(
    field_data_test_parameters: GaussianOverlapTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect compact Gaussian-overlap FD and adjoint gradient data."""
    mesh_wvl_um = field_data_test_parameters.mesh_wvl_um
    adj_wvl_um = field_data_test_parameters.adj_wvl_um
    monitor_type = field_data_test_parameters.monitor_type
    eval_fn = field_data_test_parameters.eval_fn
    monitor_side = field_data_test_parameters.monitor_side
    pol_angle = field_data_test_parameters.pol_angle
    angle_theta = field_data_test_parameters.angle_theta
    angle_phi = field_data_test_parameters.angle_phi
    waist_distance_offset = field_data_test_parameters.waist_distance_offset
    test_number = field_data_test_parameters.test_number

    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(center=(0, 0, 0), size=(dim_um, dim_um, thickness_um))

    dim = 1 + int(dim_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

    sim_geometry = get_sim_geometry(mesh_wvl_um)
    box_for_override = td.Box(
        center=(0, 0, 0), size=(*sim_geometry.size[0:2], thickness_um + mesh_wvl_um)
    )

    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, nz))
    freq0 = td.C_0 / adj_wvl_um
    for _ in range(test_number):
        # spin the rng so we get different randomness for each test
        _ = rng.random(1)
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

    return {
        "fd_grad": fd_grad,
        "adj_grad_projected": pattern_dot_adj_gradient,
        "monitor_freqs": np.asarray(monitor_freqs, dtype=float),
    }


def _evaluate_gaussian_overlap_evaluation_data(evaluation_data: EvaluationData) -> MetricGroups:
    """Evaluate saved-or-fresh Gaussian-overlap data into RFC-style metrics."""
    return evaluate_fd_adjoint_gradient_agreement(
        fd_grad=np.asarray(evaluation_data["fd_grad"]),
        adj_grad_projected=np.asarray(evaluation_data["adj_grad_projected"]),
        relative_rms_threshold=RMS_THRESHOLD,
    )


def _print_gaussian_overlap_summary(
    field_data_test_parameters: GaussianOverlapTestParameters,
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    """Print the existing Gaussian-overlap comparison summary."""
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    monitor_freqs = np.asarray(evaluation_data["monitor_freqs"], dtype=float)

    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{field_data_test_parameters.test_number}")
    print(f"Evaluation mode: {mode_label}")
    print(
        "Mesh and adjoint wavelengths: "
        f"{field_data_test_parameters.mesh_wvl_um}, {field_data_test_parameters.adj_wvl_um}"
    )
    print(f"Monitor type: {field_data_test_parameters.monitor_type}")
    print(
        "Monitor side / objective direction: "
        f"{field_data_test_parameters.monitor_side} / "
        f"{field_data_test_parameters.objective_direction}"
    )
    print(
        "Angles (pol, theta, phi): "
        f"{field_data_test_parameters.pol_angle}, "
        f"{field_data_test_parameters.angle_theta}, "
        f"{field_data_test_parameters.angle_phi}"
    )
    print(f"Waist distance offset (wvl): {field_data_test_parameters.waist_distance_offset}")
    print(f"Monitor frequencies ({len(monitor_freqs)}): {monitor_freqs.tolist()}")
    print(f"Eval function: {field_data_test_parameters.eval_fn_name}")
    print(f"RMS Error: {diagnostics['rms_error']}")
    print(f"FD, Adj magnitudes: {diagnostics['fd_mag']}, {diagnostics['adj_mag']}")
    print(f"Percentage Error: {diagnostics['percentage_error']}")
    print("-" * 20)
    print("\n" * 3)


def _plot_gaussian_overlap_comparison(
    field_data_test_parameters: GaussianOverlapTestParameters,
    evaluation_data: EvaluationData,
) -> None:
    """Plot saved FD and adjoint arrays when interactive plotting is enabled."""
    plt.plot(evaluation_data["adj_grad_projected"], color="g", linewidth=2.0, label="Adjoint")
    plt.plot(
        evaluation_data["fd_grad"],
        color="b",
        linewidth=1.5,
        linestyle="--",
        label="Finite difference",
    )
    plt.title(f"Gradient for objective: {field_data_test_parameters.eval_fn_name}")
    plt.xlabel("Sample number")
    plt.ylabel("Gradient value")
    plt.legend()
    plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "field_data_test_parameters",
    field_data_test_parameters,
    ids=lambda params: case_identity_id(_case_identity(params), prefix="gaussian-overlap"),
)
def test_finite_difference_gaussian_overlap_data(
    request: pytest.FixtureRequest,
    field_data_test_parameters: GaussianOverlapTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr: None,
) -> None:
    """Compare autograd permittivity gradients against finite-difference for Gaussian overlap power."""
    case_identity = _case_identity(field_data_test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_gaussian_overlap_evaluation_data(
            field_data_test_parameters, rng, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_gaussian_overlap_evaluation_data(evaluation_data)
    )
    _print_gaussian_overlap_summary(
        field_data_test_parameters,
        evaluation_data,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    if PLOT_FD_ADJ_COMPARISON:
        _plot_gaussian_overlap_comparison(field_data_test_parameters, evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "RMS error magnitude too large; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
