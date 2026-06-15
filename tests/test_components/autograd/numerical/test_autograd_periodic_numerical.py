# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.data.sim_data import SimulationData

from .numerical_test_helpers import (
    EvalFn,
    EvalFnResult,
    EvaluationData,
    GradientComparisonDiagnostics,
    MetricGroups,
    case_identity_from_parameters,
    case_identity_id,
    evaluate_fd_adjoint_gradient_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)

OrderPair: TypeAlias = tuple[int, int]

PLOT_FD_ADJ_COMPARISON = False
NUM_FINITE_DIFFERENCE = 10
LOCAL_GRADIENT = True
VERBOSE = False

RMS_THRESHOLD = 0.25

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


FINITE_DIFF_PERM_SEED = 1.5**2
MESH_FACTOR_DESIGN = 30.0


class PeriodicDiffractionCaseIdentity(BaseModel):
    """Semantic identity for one periodic-diffraction numerical case."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_bg_index: float
    pw_angle_deg: float
    order_x: tuple[int, ...]
    order_y: tuple[int, ...]
    polarization: str
    grating_mode: str
    eval_fn_name: str


class PeriodicDiffractionTestParameters(PeriodicDiffractionCaseIdentity):
    """Full parameter bundle for one periodic-diffraction test invocation."""

    eval_fn: EvalFn
    test_number: int


def get_sim_geometry(mesh_wvl_um):
    return td.Box(
        size=(3.5 * mesh_wvl_um, 3.5 * mesh_wvl_um, 7 * mesh_wvl_um),
        center=(3.5 * mesh_wvl_um / 4.0, 0, 0),
    )


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    box_for_override,
    pw_angle_deg,
    grating_mode,
    monitor_bg_index=1.0,
    run_time=1e-11,
):
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    sim_size_um = sim_geometry.size
    sim_center_um = sim_geometry.center

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

    src_size = (*sim_size_um[0:2], 0)

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um

    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)

    src = td.PlaneWave(
        center=(1.0, 0, -0.25 * sim_size_um[2]),
        size=[td.inf, td.inf, 0],
        source_time=pulse,
        direction="+",
        angle_theta=(pw_angle_deg * np.pi / 180.0),
    )

    bloch_x = td.Boundary.bloch_from_source(
        source=src,
        domain_size=sim_size_um[0],
        axis=0,
    )
    bloch_y = td.Boundary.bloch_from_source(
        source=src,
        domain_size=sim_size_um[1],
        axis=1,
    )

    boundary_spec = td.BoundarySpec(
        x=bloch_x,
        y=bloch_y,
        z=td.Boundary.pml(num_layers=48, extrude_structures=False),
    )

    if grating_mode not in ("transmission", "reflection"):
        raise ValueError("Unknown grating mode specified!")
    if grating_mode == "transmission":
        diffraction_monitor = td.DiffractionMonitor(
            center=(
                0,
                sim_center_um[1],
                0.25 * sim_size_um[2],
            ),
            size=(np.inf, np.inf, 0),
            name="monitor_diffraction",
            freqs=[freq0],
            normal_dir="+",
        )
    else:
        diffraction_monitor = td.DiffractionMonitor(
            center=(sim_center_um[0], sim_center_um[1], -0.35 * sim_size_um[2]),
            size=(np.inf, np.inf, 0),
            name="monitor_diffraction",
            freqs=[freq0],
            normal_dir="-",
        )

    monitor_index_zmin = 0.5 * mesh_wvl_um
    monitor_index_zmax = sim_center_um[2] + 0.5 * sim_size_um[2] + 10.0 * mesh_wvl_um
    monitor_index_block = td.Box.from_bounds(
        rmin=(
            sim_center_um[0] - sim_size_um[0],
            sim_center_um[1] - sim_size_um[1],
            monitor_index_zmin,
        ),
        rmax=(
            sim_center_um[0] + sim_size_um[0],
            sim_center_um[1] + sim_size_um[1],
            monitor_index_zmax,
        ),
    )
    monitor_index_block_structure = td.Structure(
        geometry=monitor_index_block, medium=td.Medium(permittivity=monitor_bg_index**2)
    )

    sim_base = td.Simulation(
        center=sim_center_um,
        size=sim_size_um,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=30,
            wavelength=mesh_wvl_um,
            override_structures=mesh_overrides,
        ),
        structures=[monitor_index_block_structure],
        sources=[src],
        monitors=[diffraction_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_objective_function(geometry, create_sim_base, eval_fn, sim_path_dir):
    def objective(perm_arrays):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(perm_arrays)):
            block_structure = td.Structure.from_permittivity_array(
                eps_data=perm_arrays[idx],
                geometry=geometry,
            )

            sim_with_block = sim_base.updated_copy(
                structures=(*sim_base.structures, block_structure)
            )

            simulation_dict[f"numerical_periodic_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict,
            path_dir=sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
            verbose=VERBOSE,
            lazy=False,
        )

        objective_vals = []
        for idx in range(len(perm_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_periodic_testing_{idx}"]))

        if len(perm_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def selected_order_pairs(
    orders_x: tuple[int, ...], orders_y: tuple[int, ...]
) -> tuple[OrderPair, ...]:
    """Return the ordered diffraction-order pairs selected by the objective."""
    return tuple((int(order_x), int(order_y)) for order_x in orders_x for order_y in orders_y)


def make_order_power_eval_fn(order_x: int, order_y: int, polarization: str) -> EvalFn:
    """Create an objective that reads one diffraction order and polarization."""

    def order_pol_amp_sq(sim_data: SimulationData) -> EvalFnResult:
        return np.sum(
            np.abs(
                sim_data["monitor_diffraction"]
                .amps.sel(polarization=polarization, orders_x=order_x, orders_y=order_y)
                .data
            )
            ** 2
        )

    return order_pol_amp_sq


def make_eval_fns(orders_x, orders_y, polarization):
    order_eval_fns = [
        make_order_power_eval_fn(order_x_val, order_y_val, polarization)
        for order_x_val, order_y_val in selected_order_pairs(orders_x, orders_y)
    ]

    def transmission_order_pol_amp_sq(sim_data: SimulationData) -> EvalFnResult:
        total = 0.0

        for order_eval_fn in order_eval_fns:
            total += order_eval_fn(sim_data)

        return total

    eval_fns = [transmission_order_pol_amp_sq]
    eval_fn_names = [f"transmission_order_pol_amp_sq_{orders_x}_{orders_y}_{polarization}"]

    return eval_fns, eval_fn_names


background_indices = [1.0, 1.5]
mesh_wvls_um = [1.55]
adj_wvls_um = [1.55]

orders_x = [(0,), (1,), (2,), (1, 2)]
orders_y = [(0,), (0,), (0,), (1,)]

polarizations = ["p", "p", "p", "s"]

grating_modes = ["transmission", "reflection"]

pw_angles_deg = [10.0]

periodic_test_parameters: list[PeriodicDiffractionTestParameters] = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for grating_mode in grating_modes:
        for order_idx in range(len(orders_x)):
            eval_fns, eval_fn_names = make_eval_fns(
                orders_x=orders_x[order_idx],
                orders_y=orders_y[order_idx],
                polarization=polarizations[order_idx],
            )

            for pw_angle_deg in pw_angles_deg:
                for monitor_bg_index in background_indices:
                    for eval_fn_idx, eval_fn in enumerate(eval_fns):
                        periodic_test_parameters.append(
                            PeriodicDiffractionTestParameters(
                                mesh_wvl_um=mesh_wvl_um,
                                adj_wvl_um=adj_wvl_um,
                                monitor_bg_index=monitor_bg_index,
                                pw_angle_deg=pw_angle_deg,
                                order_x=orders_x[order_idx],
                                order_y=orders_y[order_idx],
                                polarization=polarizations[order_idx],
                                grating_mode=grating_mode,
                                eval_fn=eval_fn,
                                eval_fn_name=eval_fn_names[eval_fn_idx],
                                test_number=test_number,
                            )
                        )

                        test_number += 1


def _case_identity(
    periodic_test_parameters: PeriodicDiffractionTestParameters,
) -> PeriodicDiffractionCaseIdentity:
    """Build the semantic case identity used for eval-only replay validation."""
    return case_identity_from_parameters(PeriodicDiffractionCaseIdentity, periodic_test_parameters)


def _as_single_gradient(adj_grad: object, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Normalize autograd's list-shaped gradient return into one real ndarray."""
    grad_array = np.asarray(adj_grad)
    if grad_array.shape == (1, *expected_shape):
        grad_array = grad_array[0]
    if grad_array.shape != expected_shape:
        raise ValueError(
            "Expected adjoint gradient with shape "
            f"{expected_shape} or {(1, *expected_shape)}, got {grad_array.shape}."
        )
    return np.real(grad_array)


def _collect_adjoint_gradient(
    *,
    block: td.Box,
    create_sim_base: Callable[[], td.Simulation],
    perm_init: np.ndarray,
    eval_fn: EvalFn,
    adjoint_path_dir: Path,
) -> np.ndarray:
    """Run the original combined objective as a single adjoint gradient calculation."""
    objective_adj = create_objective_function(
        block,
        create_sim_base,
        eval_fn,
        sim_path_dir=str(adjoint_path_dir),
    )
    obj_val_and_grad = ag.value_and_grad(objective_adj)
    _, adj_grad = obj_val_and_grad([perm_init])
    return _as_single_gradient(adj_grad, perm_init.shape)


def _collect_periodic_diffraction_evaluation_data(
    periodic_test_parameters: PeriodicDiffractionTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect compact periodic-diffraction FD and adjoint gradient data."""
    mesh_wvl_um = periodic_test_parameters.mesh_wvl_um
    adj_wvl_um = periodic_test_parameters.adj_wvl_um
    monitor_bg_index = periodic_test_parameters.monitor_bg_index
    pw_angle_deg = periodic_test_parameters.pw_angle_deg
    grating_mode = periodic_test_parameters.grating_mode
    eval_fn = periodic_test_parameters.eval_fn
    test_number = periodic_test_parameters.test_number

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(
        center=(sim_geometry.center[0], sim_geometry.center[1], 0),
        size=(dim_um, dim_um, thickness_um),
    )

    dim = 1 + int(dim_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

    box_for_override = td.Box(
        center=(sim_geometry.center[0], sim_geometry.center[1], 0),
        size=(*sim_geometry.size[0:2], thickness_um + mesh_wvl_um),
    )

    with TemporaryDirectory(prefix=f"test{test_number}_", dir=numerical_case_dir) as sim_path_dir:
        sim_path_dir = Path(sim_path_dir)
        adjoint_path_dir = sim_path_dir / "adjoint"
        finite_difference_path_dir = sim_path_dir / "finite_difference"
        adjoint_path_dir.mkdir()
        finite_difference_path_dir.mkdir()

        def create_sim_base(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            box_for_override=box_for_override,
            pw_angle_deg=pw_angle_deg,
            grating_mode=grating_mode,
            monitor_bg_index=monitor_bg_index,
        ) -> td.Simulation:
            return make_base_sim(
                mesh_wvl_um=mesh_wvl_um,
                adj_wvl_um=adj_wvl_um,
                box_for_override=box_for_override,
                pw_angle_deg=pw_angle_deg,
                grating_mode=grating_mode,
                monitor_bg_index=monitor_bg_index,
            )

        objective_fd = create_objective_function(
            block,
            create_sim_base,
            eval_fn,
            sim_path_dir=str(finite_difference_path_dir),
        )

        perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, nz))

        adj_grad = _collect_adjoint_gradient(
            block=block,
            create_sim_base=create_sim_base,
            perm_init=perm_init,
            eval_fn=eval_fn,
            adjoint_path_dir=adjoint_path_dir,
        )

        # empirical step size from running other finite difference tests for field
        # cases with permittivity
        fd_step = 0.1

        all_perm = []
        pattern_dot_adj_gradient = np.zeros(NUM_FINITE_DIFFERENCE)

        for fd_idx in range(NUM_FINITE_DIFFERENCE):
            random_pattern = rng.random((dim, dim, nz)) - 0.5
            random_pattern = gaussian_filter(random_pattern, sigma=3)
            random_pattern /= np.linalg.norm(random_pattern)

            pattern_dot_adj_gradient[fd_idx] = float(np.real(np.sum(random_pattern * adj_grad)))

            perm_up = perm_init.copy() + fd_step * random_pattern
            perm_down = perm_init.copy() - fd_step * random_pattern

            all_perm.append(perm_up)
            all_perm.append(perm_down)

        all_obj = objective_fd(all_perm)

        fd_grad = np.zeros(NUM_FINITE_DIFFERENCE)
        for fd_idx in range(NUM_FINITE_DIFFERENCE):
            obj_up_location = 2 * fd_idx
            obj_down_location = 2 * fd_idx + 1

            fd_grad[fd_idx] = (all_obj[obj_up_location] - all_obj[obj_down_location]) / (
                2 * fd_step
            )

    return {
        "fd_grad": fd_grad,
        "adj_grad_projected": pattern_dot_adj_gradient,
    }


def _evaluate_periodic_diffraction_evaluation_data(evaluation_data: EvaluationData) -> MetricGroups:
    """Evaluate saved-or-fresh periodic-diffraction data into RFC-style metrics."""
    return evaluate_fd_adjoint_gradient_agreement(
        fd_grad=np.asarray(evaluation_data["fd_grad"]),
        adj_grad_projected=np.asarray(evaluation_data["adj_grad_projected"]),
        relative_rms_threshold=RMS_THRESHOLD,
    )


def _print_periodic_diffraction_summary(
    periodic_test_parameters: PeriodicDiffractionTestParameters,
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    """Print the existing periodic-diffraction comparison summary."""
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"

    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{periodic_test_parameters.test_number}")
    print(f"Evaluation mode: {mode_label}")
    print(
        "Mesh and adjoint wavelengths: "
        f"{periodic_test_parameters.mesh_wvl_um}, {periodic_test_parameters.adj_wvl_um}"
    )
    print(f"Input plane wave angle (deg): {periodic_test_parameters.pw_angle_deg}")
    print(
        "(X, Y) order, polarization: "
        f"({periodic_test_parameters.order_x}, {periodic_test_parameters.order_y}), "
        f"{periodic_test_parameters.polarization}"
    )
    print(f"Grating mode: {periodic_test_parameters.grating_mode}")
    print(f"Background index for monitor: {periodic_test_parameters.monitor_bg_index}")
    print(f"Eval function: {periodic_test_parameters.eval_fn_name}")
    print(f"RMS Error: {diagnostics['rms_error']}")
    print(f"FD, Adj magnitudes: {diagnostics['fd_mag']}, {diagnostics['adj_mag']}")
    print(f"Percentage Error: {diagnostics['percentage_error']}")
    print("-" * 20)
    print("\n" * 3)


def _plot_periodic_diffraction_comparison(
    periodic_test_parameters: PeriodicDiffractionTestParameters,
    evaluation_data: EvaluationData,
) -> None:
    """Plot saved FD and adjoint arrays when interactive plotting is enabled."""
    plt.plot(evaluation_data["adj_grad_projected"], color="g", linewidth=2.0, label="Adjoint total")
    plt.plot(
        evaluation_data["fd_grad"],
        color="b",
        linewidth=1.5,
        linestyle="--",
        label="Finite difference",
    )
    plt.title(f"Gradient for objective: {periodic_test_parameters.eval_fn_name}")
    plt.legend()
    plt.xlabel("Sample number")
    plt.ylabel("Gradient value")
    plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "periodic_test_parameters",
    periodic_test_parameters,
    ids=lambda params: case_identity_id(_case_identity(params), prefix="periodic"),
)
def test_finite_difference_diffraction_data(
    request: pytest.FixtureRequest,
    periodic_test_parameters: PeriodicDiffractionTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr: None,
) -> None:
    """Test a variety of autograd permittivity gradients for DiffractionData by"""
    """comparing them to numerical finite difference."""
    case_identity = _case_identity(periodic_test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_periodic_diffraction_evaluation_data(
            periodic_test_parameters, rng, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_periodic_diffraction_evaluation_data(evaluation_data)
    )
    _print_periodic_diffraction_summary(
        periodic_test_parameters,
        evaluation_data,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    if PLOT_FD_ADJ_COMPARISON:
        _plot_periodic_diffraction_comparison(periodic_test_parameters, evaluation_data)

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
