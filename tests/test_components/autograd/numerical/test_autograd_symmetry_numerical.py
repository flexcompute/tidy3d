# test autograd for field sources when symmetry is used in the simulation
from __future__ import annotations

from pathlib import Path

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.types.base import Size

from .numerical_test_helpers import (
    EvalFn,
    EvaluationData,
    case_identity_from_parameters,
    case_identity_id,
    condition_metric,
    finalize_result,
    load_or_collect_evaluation_data,
)
from .result_models import Metric

PLOT_SYMMETRY_COMPARISON = False
LOCAL_GRADIENT = True
VERBOSE = False

RMS_THRESHOLD = 0.25

if PLOT_SYMMETRY_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


FINITE_DIFF_PERM_SEED = 1.5**2
MESH_FACTOR_DESIGN = 30.0


def get_sim_geometry(mesh_wvl_um):
    return td.Box(size=(5 * mesh_wvl_um, 5 * mesh_wvl_um, 7 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    monitor_size_wvl,
    box_for_override,
    symmetry,
    monitor_bg_index=1.0,
    run_time=1e-11,
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
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=0,
        angle_theta=0,
        source_time=pulse,
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
        symmetry=symmetry,
    )

    return sim_base


def create_objective_functions(geometry, create_sim_base, eval_fn, sim_path_dir):
    sim_path_dir = Path(sim_path_dir)
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    def objective_(perm_array, symmetry):
        sim_base = create_sim_base(symmetry)

        block_structure = td.Structure.from_permittivity_array(
            eps_data=perm_array,
            geometry=geometry,
        )

        sim_with_block = sim_base.updated_copy(structures=(*sim_base.structures, block_structure))

        symmetry_tag = "_".join(str(val) for val in symmetry)
        result_path = sim_path_dir / f"symmetry_{symmetry_tag}.hdf5"

        sim_data = web.run(
            sim_with_block,
            task_name="symmetry_field_testing",
            path=str(result_path),
            local_gradient=LOCAL_GRADIENT,
            verbose=VERBOSE,
        )

        objective_val = eval_fn(sim_data)

        return objective_val

    def objective_no_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(0, 0, 0))

    def objective_x_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(-1, 0, 0))

    def objective_y_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(0, 1, 0))

    def objective_xy_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(-1, 1, 0))

    return objective_no_symmetry, objective_x_symmetry, objective_y_symmetry, objective_xy_symmetry


def make_eval_fns(monitor_size_wvl):
    num_nonzero_spatial_dims = 3 - np.sum(np.isclose(monitor_size_wvl, 0))

    def intensity(sim_data):
        field_data = sim_data["monitor_fields"]

        return np.sum(field_data.field_intensity(components=("Ex", "Ey")).values)

    eval_fns = [intensity]
    eval_fn_names = ["intensity"]

    if num_nonzero_spatial_dims == 2:

        def flux(sim_data):
            field_data = sim_data["monitor_fields"]

            return np.sum(field_data.flux.values)

        eval_fns.append(flux)
        eval_fn_names.append("flux")

    return eval_fns, eval_fn_names


background_indices = [1.0]
mesh_wvls_um = [1.55]
adj_wvls_um = [1.55]
monitor_sizes_3d_wvl = [(0.5, 0.5, 0)]


class FieldSymmetryCaseIdentity(BaseModel):
    """Semantic identity for one field-source symmetry gradient comparison."""

    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_size_wvl: Size
    monitor_bg_index: float
    eval_fn_name: str


class FieldSymmetryTestParameters(FieldSymmetryCaseIdentity):
    """Full parameter bundle for one symmetry test invocation."""

    eval_fn: EvalFn
    test_number: int


field_symmetry_test_parameters: list[FieldSymmetryTestParameters] = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for monitor_size_wvl in monitor_sizes_3d_wvl:
        eval_fns, eval_fn_names = make_eval_fns(monitor_size_wvl)

        for monitor_bg_index in background_indices:
            for eval_fn_idx, eval_fn in enumerate(eval_fns):
                field_symmetry_test_parameters.append(
                    FieldSymmetryTestParameters(
                        mesh_wvl_um=mesh_wvl_um,
                        adj_wvl_um=adj_wvl_um,
                        monitor_size_wvl=monitor_size_wvl,
                        monitor_bg_index=monitor_bg_index,
                        eval_fn=eval_fn,
                        eval_fn_name=eval_fn_names[eval_fn_idx],
                        test_number=test_number,
                    )
                )

                test_number += 1


def _case_identity(parameters: FieldSymmetryTestParameters) -> FieldSymmetryCaseIdentity:
    """Build the semantic case identity used for eval-only replay validation."""
    return case_identity_from_parameters(FieldSymmetryCaseIdentity, parameters)


def _collect_field_symmetry_evaluation_data(
    field_symmetry_test_parameters: FieldSymmetryTestParameters,
    numerical_case_dir: Path,
) -> EvaluationData:
    """Collect objective values and gradients for all symmetry settings."""
    mesh_wvl_um = field_symmetry_test_parameters.mesh_wvl_um
    adj_wvl_um = field_symmetry_test_parameters.adj_wvl_um
    monitor_size_wvl = field_symmetry_test_parameters.monitor_size_wvl
    monitor_bg_index = field_symmetry_test_parameters.monitor_bg_index
    eval_fn = field_symmetry_test_parameters.eval_fn
    test_number = field_symmetry_test_parameters.test_number

    dim_um = mesh_wvl_um
    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(center=(0, 0, 0), size=(dim_um, dim_um, thickness_um))

    dim = 1 + int(dim_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    Nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0), size=(*sim_geometry.size[0:2], thickness_um + mesh_wvl_um)
    )

    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    objective_no_symmetry, objective_x_symmetry, objective_y_symmetry, objective_xy_symmetry = (
        create_objective_functions(
            block,
            lambda symmetry,
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            monitor_size_wvl=monitor_size_wvl,
            box_for_override=box_for_override,
            monitor_bg_index=monitor_bg_index: make_base_sim(
                mesh_wvl_um=mesh_wvl_um,
                adj_wvl_um=adj_wvl_um,
                monitor_size_wvl=monitor_size_wvl,
                box_for_override=box_for_override,
                monitor_bg_index=monitor_bg_index,
                symmetry=symmetry,
            ),
            eval_fn,
            sim_path_dir=str(sim_path_dir),
        )
    )

    obj_val_and_grad_no_symmetry = ag.value_and_grad(objective_no_symmetry)
    obj_val_and_grad_x_symmetry = ag.value_and_grad(objective_x_symmetry)
    obj_val_and_grad_y_symmetry = ag.value_and_grad(objective_y_symmetry)
    obj_val_and_grad_xy_symmetry = ag.value_and_grad(objective_xy_symmetry)

    objs_val_and_grad = [
        obj_val_and_grad_no_symmetry,
        obj_val_and_grad_x_symmetry,
        obj_val_and_grad_y_symmetry,
        obj_val_and_grad_xy_symmetry,
    ]

    symmetries = ["none", "x", "y", "xy"]

    objs = []
    adj_grads = []

    perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, Nz))

    for obj_val_and_grad in objs_val_and_grad:
        obj, adj_grad = obj_val_and_grad(perm_init)

        objs.append(obj)
        adj_grads.append(np.array(adj_grad))

    return {
        "objective_values": np.asarray(objs, dtype=float),
        "adjoint_gradients": np.asarray(adj_grads, dtype=float),
    }


def _evaluate_field_symmetry_evaluation_data(
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    """Evaluate saved-or-fresh symmetry gradients into RFC-style metrics."""
    objs = np.asarray(evaluation_data["objective_values"], dtype=float)
    adj_grads = np.asarray(evaluation_data["adjoint_gradients"], dtype=float)
    symmetries = ["none", "x", "y", "xy"]

    regression_metrics = [
        condition_metric("objective_values_finite", bool(np.all(np.isfinite(objs)))),
        condition_metric("adjoint_gradients_finite", bool(np.all(np.isfinite(adj_grads)))),
    ]
    observation_metrics: list[Metric] = []
    diagnostics: dict[str, float] = {}

    grad_data_base = adj_grads[0] / objs[0]
    for idx in range(1, len(adj_grads)):
        # Field magnitudes can differ across symmetries, so compare objective-normalized gradients.
        grad_data = adj_grads[idx] / objs[idx]

        mag_base = np.sqrt(np.mean(grad_data_base**2))
        mag_compare = np.sqrt(np.mean(grad_data**2))
        rms_error = np.sqrt(np.mean((grad_data_base - grad_data) ** 2))
        normalization = np.sqrt(mag_base * mag_compare)
        normalized_rms_error = (
            float(rms_error / normalization)
            if normalization > 0
            else float(np.finfo(np.float64).max)
        )

        regression_metrics.append(
            Metric(
                name=f"symmetry_{symmetries[idx]}_normalized_rms_error",
                observed=normalized_rms_error,
                expected=0.075,
                comparator="lte",
            )
        )
        diagnostics[f"{symmetries[idx]}_normalized_rms_error"] = normalized_rms_error
        observation_metrics.extend(
            [
                Metric(
                    name=f"symmetry_{symmetries[idx]}_base_gradient_rms",
                    observed=float(mag_base),
                    expected=0.0,
                    comparator="gte",
                ),
                Metric(
                    name=f"symmetry_{symmetries[idx]}_comparison_gradient_rms",
                    observed=float(mag_compare),
                    expected=0.0,
                    comparator="gte",
                ),
            ]
        )

    return regression_metrics, observation_metrics, diagnostics


def _print_field_symmetry_summary(
    field_symmetry_test_parameters: FieldSymmetryTestParameters,
    diagnostics: dict[str, float],
) -> None:
    """Print the original symmetry-comparison diagnostics."""
    for symmetry in ("x", "y", "xy"):
        print(f"Testing {field_symmetry_test_parameters.eval_fn_name} objective")
        print(f"Symmetry comparison: none, {symmetry}")
        print(f"RMS error (normalized): {diagnostics[f'{symmetry}_normalized_rms_error']}")


def _plot_field_symmetry_comparison(evaluation_data: EvaluationData) -> None:
    """Plot symmetry gradient comparisons when interactive plotting is enabled."""
    objs = np.asarray(evaluation_data["objective_values"], dtype=float)
    adj_grads = np.asarray(evaluation_data["adjoint_gradients"], dtype=float)
    symmetries = ["none", "x", "y", "xy"]
    grad_data_base = adj_grads[0] / objs[0]
    plot_grad_data_base = np.squeeze(grad_data_base)

    for idx in range(1, len(adj_grads)):
        grad_data = adj_grads[idx] / objs[idx]
        plot_grad_data = np.squeeze(grad_data)
        plot_diff = plot_grad_data - plot_grad_data_base

        plt.subplot(1, 3, 1)
        plt.imshow(plot_grad_data_base[:, :, plot_grad_data_base.shape[2] // 2])
        plt.title(f"Symmetry: {symmetries[0]}")
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(plot_grad_data[:, :, plot_grad_data.shape[2] // 2])
        plt.title(f"Symmetry: {symmetries[idx]}")
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(plot_diff[:, :, plot_diff.shape[2] // 2])
        plt.title("Difference")
        plt.colorbar()
        plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "field_symmetry_test_parameters",
    field_symmetry_test_parameters,
    ids=lambda params: case_identity_id(_case_identity(params), prefix="symmetry"),
)
def test_adjoint_difference_symmetry(
    request: pytest.FixtureRequest,
    field_symmetry_test_parameters: FieldSymmetryTestParameters,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Test the gradient is not affected by symmetry when using field sources."""
    case_identity = _case_identity(field_symmetry_test_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_field_symmetry_evaluation_data(
            field_symmetry_test_parameters, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_field_symmetry_evaluation_data(
        evaluation_data
    )
    _print_field_symmetry_summary(field_symmetry_test_parameters, diagnostics)

    if PLOT_SYMMETRY_COMPARISON:
        _plot_field_symmetry_comparison(evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Expected adjoint gradients to be the same with and without symmetry; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
