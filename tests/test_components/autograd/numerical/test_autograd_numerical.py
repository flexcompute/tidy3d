# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.types.base import Size

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


class FieldDataCaseIdentity(BaseModel):
    """Semantic identity for one field-data numerical case."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_size_wvl: Size
    monitor_bg_index: float
    eval_fn_name: str
    cm_interp_method: str


class FieldDataTestParameters(FieldDataCaseIdentity):
    """Full parameter bundle for one field-data test invocation."""

    eval_fn: EvalFn
    test_number: int


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


def get_sim_geometry(mesh_wvl_um: float) -> td.Box:
    return td.Box(size=(5 * mesh_wvl_um, 5 * mesh_wvl_um, 7 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um: float,
    adj_wvl_um: float,
    monitor_size_wvl: Size,
    box_for_override: td.Box,
    monitor_bg_index: float = 1.0,
    run_time: float = 1e-11,
) -> td.Simulation:
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

    src_size = (*sim_size_um[0:2], 0)

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um

    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)
    src = td.PlaneWave(
        center=(0, 0, -2 * mesh_wvl_um),
        size=src_size,
        source_time=pulse,
        direction="+",
    )

    field_monitor = td.FieldMonitor(
        center=(0, 0, 0.25 * sim_size_um[2]),
        size=tuple(dim * mesh_wvl_um for dim in monitor_size_wvl),
        name="monitor_fields",
        freqs=[freq0],
    )

    monitor_index_block = td.Box(
        center=(0, 0, 0.25 * sim_size_um[2] + mesh_wvl_um),
        size=(*tuple(2 * size for size in sim_size_um[0:2]), mesh_wvl_um + 0.5 * sim_size_um[2]),
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
        monitors=[field_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_objective_function(
    geometry: td.Box,
    create_sim_base: Any,
    eval_fn: EvalFn,
    sim_path_dir: str,
    perm_init: np.ndarray,
    cm_interp_method: str,
) -> Any:
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

            block_structure = block_structure.updated_copy(
                medium=block_structure.medium.updated_copy(interp_method=cm_interp_method)
            )

            sim_with_block = sim_base.updated_copy(
                structures=(*sim_base.structures, block_structure),
                grid_spec=td.GridSpec.from_grid(grid_fixed),
            )

            simulation_dict[f"numerical_field_testing_{idx}"] = sim_with_block.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(perm_arrays)):
            objective_vals.append(eval_fn(sim_data[f"numerical_field_testing_{idx}"]))

        if len(perm_arrays) == 1:
            return objective_vals[0]

        return objective_vals

    return objective


def make_eval_fns(monitor_size_wvl: Size) -> tuple[list[EvalFn], list[str]]:
    num_nonzero_spatial_dims = 3 - np.sum(np.isclose(monitor_size_wvl, 0))

    def intensity(sim_data: SimulationData) -> EvalFnResult:
        field_data = sim_data["monitor_fields"]
        shape_x, shape_y, shape_z, *_ = field_data.Ex.values.shape
        return np.sum(
            np.abs(field_data.Ex.values[shape_x // 2, shape_y // 2, shape_z // 2]) ** 2
            + np.abs(field_data.Ey.values[shape_x // 2, shape_y // 2, shape_z // 2]) ** 2
            + np.abs(field_data.Ez.values[shape_x // 2, shape_y // 2, shape_z // 2]) ** 2
        )

    eval_fns = [intensity]
    eval_fn_names = ["intensity"]

    if num_nonzero_spatial_dims == 2:

        def flux(sim_data: SimulationData) -> EvalFnResult:
            field_data = sim_data["monitor_fields"]
            return np.sum(field_data.flux.values)

        eval_fns.append(flux)
        eval_fn_names.append("flux")

    return eval_fns, eval_fn_names


background_indices = [1.0, 1.5]
mesh_wvls_um = [1.55, 1.55, 10 * 1.55, 10 * 1.55]
adj_wvls_um = [1.55, 2.2, 10 * 1.55, 10 * 2.2]
monitor_sizes_3d_wvl: list[Size] = [
    (0.5, 0.5, 0),
    (0.5, 0.5, 0.5),
    (0.5, 0, 0),
    (0, 0.5, 0),
    (0, 0, 0),
]
cm_interp_methods = ["nearest", "linear"]

field_data_test_parameters: list[FieldDataTestParameters] = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for monitor_size_wvl in monitor_sizes_3d_wvl:
        eval_fns, eval_fn_names = make_eval_fns(monitor_size_wvl)

        for monitor_bg_index in background_indices:
            for eval_fn_idx, eval_fn in enumerate(eval_fns):
                for cm_interp_method in cm_interp_methods:
                    field_data_test_parameters.append(
                        FieldDataTestParameters(
                            mesh_wvl_um=mesh_wvl_um,
                            adj_wvl_um=adj_wvl_um,
                            monitor_size_wvl=monitor_size_wvl,
                            monitor_bg_index=monitor_bg_index,
                            eval_fn=eval_fn,
                            eval_fn_name=eval_fn_names[eval_fn_idx],
                            cm_interp_method=cm_interp_method,
                            test_number=test_number,
                        )
                    )

                    test_number += 1


def _case_identity(field_data_test_parameters: FieldDataTestParameters) -> FieldDataCaseIdentity:
    """Build the semantic case identity used for eval-only replay validation."""
    return case_identity_from_parameters(FieldDataCaseIdentity, field_data_test_parameters)


def _collect_field_data_evaluation_data(
    field_data_test_parameters: FieldDataTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: str | Path,
) -> EvaluationData:
    """Collect the compact evaluation dataset needed for later offline re-evaluation."""
    mesh_wvl_um = field_data_test_parameters.mesh_wvl_um
    adj_wvl_um = field_data_test_parameters.adj_wvl_um
    monitor_size_wvl = field_data_test_parameters.monitor_size_wvl
    monitor_bg_index = field_data_test_parameters.monitor_bg_index
    eval_fn = field_data_test_parameters.eval_fn
    cm_interp_method = field_data_test_parameters.cm_interp_method
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

    perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, nz))
    with TemporaryDirectory(prefix=f"test{test_number}_", dir=numerical_case_dir) as sim_path_dir:
        objective = create_objective_function(
            block,
            lambda mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            monitor_size_wvl=monitor_size_wvl,
            box_for_override=box_for_override,
            monitor_bg_index=monitor_bg_index: make_base_sim(
                mesh_wvl_um=mesh_wvl_um,
                adj_wvl_um=adj_wvl_um,
                monitor_size_wvl=monitor_size_wvl,
                box_for_override=box_for_override,
                monitor_bg_index=monitor_bg_index,
            ),
            eval_fn,
            sim_path_dir=sim_path_dir,
            perm_init=perm_init,
            cm_interp_method=cm_interp_method,
        )

        obj_val_and_grad = ag.value_and_grad(objective)
        _obj, adj_grad = obj_val_and_grad([perm_init])

        # Empirical step size from earlier field-data finite-difference experiments.
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
            all_perm.extend((perm_up, perm_down))

        all_obj = objective(all_perm)

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


def _evaluate_field_data_evaluation_data(
    evaluation_data: EvaluationData,
) -> MetricGroups:
    """Evaluate a saved-or-fresh field-data dataset into RFC-style metrics."""
    return evaluate_fd_adjoint_gradient_agreement(
        fd_grad=np.asarray(evaluation_data["fd_grad"]),
        adj_grad_projected=np.asarray(evaluation_data["adj_grad_projected"]),
        relative_rms_threshold=RMS_THRESHOLD,
    )


def _print_field_data_summary(
    field_data_test_parameters: FieldDataTestParameters,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    """Print a compact case summary for fresh and eval-only runs."""
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{field_data_test_parameters.test_number}")
    print(f"Evaluation mode: {mode_label}")
    print(
        "Mesh and adjoint wavelengths: "
        f"{field_data_test_parameters.mesh_wvl_um}, {field_data_test_parameters.adj_wvl_um}"
    )
    print(f"Monitor size: {field_data_test_parameters.monitor_size_wvl}")
    print(f"Background index for monitor: {field_data_test_parameters.monitor_bg_index}")
    print(f"Eval function: {field_data_test_parameters.eval_fn_name}")
    print(f"Custom medium interpolation method: {field_data_test_parameters.cm_interp_method}")
    print(f"RMS Error: {diagnostics['rms_error']}")
    print(f"FD, Adj magnitudes: {diagnostics['fd_mag']}, {diagnostics['adj_mag']}")
    print(f"Percentage Error: {diagnostics['percentage_error']}")
    print("-" * 20)
    print("\n" * 3)


def _plot_field_data_comparison(evaluation_data: EvaluationData, *, eval_fn_name: str) -> None:
    """Plot the saved FD and adjoint comparison arrays when interactive plotting is enabled."""
    plt.plot(evaluation_data["adj_grad_projected"], color="g", linewidth=2.0, label="Adjoint")
    plt.plot(
        evaluation_data["fd_grad"],
        color="b",
        linewidth=1.5,
        linestyle="--",
        label="Finite difference",
    )
    plt.title(f"Gradient for objective: {eval_fn_name}")
    plt.xlabel("Sample number")
    plt.ylabel("Gradient value")
    plt.legend()
    plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "field_data_test_parameters",
    field_data_test_parameters,
    ids=lambda params: case_identity_id(_case_identity(params)),
)
def test_finite_difference_field_data(
    request: pytest.FixtureRequest,
    field_data_test_parameters: FieldDataTestParameters,
    rng: np.random.Generator,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr: None,
) -> None:
    """Test a variety of autograd permittivity gradients for FieldData by"""
    """comparing them to numerical finite difference."""
    case_identity = _case_identity(field_data_test_parameters)

    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_field_data_evaluation_data(
            field_data_test_parameters, rng, numerical_case_dir
        ),
    )

    regression_metrics, observation_metrics, diagnostics = _evaluate_field_data_evaluation_data(
        evaluation_data
    )
    _print_field_data_summary(
        field_data_test_parameters,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    if PLOT_FD_ADJ_COMPARISON:
        _plot_field_data_comparison(
            evaluation_data, eval_fn_name=field_data_test_parameters.eval_fn_name
        )

    result_record = finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "RMS error magnitude too large; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
