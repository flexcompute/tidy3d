"""Numerical test for CustomPoleResidue adjoint gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static

from .numerical_test_helpers import (
    EvaluationData,
    GradientComparisonDiagnostics,
    MetricGroups,
    coords_for_bounds,
    evaluate_gradient_angle_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
    scale_length_by_wavelength,
)


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


SIM_SIZE_SCALE = (4, 3, 4)
BOX_SIZE_SCALE = (1, 1, 1)
GRID_STEPS_PER_WVL = 30
RUN_TIME = 2e-12
ANGLE_TOL = 10.0
FD_STEP = 5e-2

DENSITY_SHAPE = (2, 2, 2)
EPS_BACKGROUND = 1.0
EPS_INF_BASE = 2.8
A_BASE = -1.5e15
C_BASE = 0.8e15

POLE_RESIDUE_CASE = {
    "name": "custom_pole_residue",
    "wavelength": 1.3,
    "objective_kind": "flux",
    "monitor_size": (np.inf, np.inf, 0.0),
    "polarization": 0.0,
}


class CustomPoleResidueCaseIdentity(BaseModel):
    """Semantic identity for the CustomPoleResidue finite-difference check."""

    name: str
    wavelength: float
    objective_kind: str
    monitor_size: tuple[float, float, float]
    polarization: float
    density_shape: tuple[int, int, int]
    fd_step: float
    angle_tol: float
    grid_steps_per_wvl: int


def _case_identity(case: dict[str, object]) -> CustomPoleResidueCaseIdentity:
    return CustomPoleResidueCaseIdentity(
        name=str(case["name"]),
        wavelength=float(case["wavelength"]),
        objective_kind=str(case["objective_kind"]),
        monitor_size=tuple(float(value) for value in case["monitor_size"]),
        polarization=float(case.get("polarization", 0.0)),
        density_shape=DENSITY_SHAPE,
        fd_step=FD_STEP,
        angle_tol=ANGLE_TOL,
        grid_steps_per_wvl=GRID_STEPS_PER_WVL,
    )


def _box_geometry(wavelength: float) -> td.Box:
    size = tuple(scale * wavelength for scale in BOX_SIZE_SCALE)
    return td.Box(size=size, center=(0.0, 0.0, 0.0))


def _build_base_sim(case):
    wavelength = case["wavelength"]
    freq0 = td.C_0 / wavelength
    sim_size = tuple(scale * wavelength for scale in SIM_SIZE_SCALE)

    plane_wave = td.PlaneWave(
        center=(0.0, 0.0, -0.75 * sim_size[2] / 2),
        size=(sim_size[0], sim_size[1], 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
        direction="+",
        pol_angle=case.get("polarization", 0.0),
    )

    monitor_center = (0.0, 0.0, sim_size[2] / 2 * 0.75)
    monitor_size = tuple(
        scale_length_by_wavelength(dim, wavelength) for dim in case["monitor_size"]
    )
    monitor_name = f"{case['name']}_monitor"
    monitor = td.FieldMonitor(
        center=monitor_center,
        size=monitor_size,
        freqs=[freq0],
        name=monitor_name,
        colocate=False,
    )

    sim = td.Simulation(
        size=sim_size,
        center=(0.0, 0.0, 0.0),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=GRID_STEPS_PER_WVL, wavelength=wavelength),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[plane_wave],
        monitors=[monitor],
        structures=[],
        run_time=RUN_TIME,
    )
    return sim, monitor_name, freq0


def _custom_pole_residue_from_density(density, coords):
    eps_inf = EPS_BACKGROUND + density * (EPS_INF_BASE - EPS_BACKGROUND)
    eps_inf_da = td.SpatialDataArray(eps_inf, coords=coords)

    a_vals = A_BASE * anp.ones_like(density)
    c_vals = C_BASE * density
    a_da = td.SpatialDataArray(a_vals, coords=coords)
    c_da = td.SpatialDataArray(c_vals, coords=coords)

    return td.CustomPoleResidue(eps_inf=eps_inf_da, poles=((a_da, c_da),), interp_method="linear")


def _add_custom_pole_residue(base_sim: td.Simulation, box_geom: td.Box, params) -> td.Simulation:
    density = anp.reshape(params, DENSITY_SHAPE)
    coords = coords_for_bounds(box_geom.bounds, DENSITY_SHAPE)
    medium = _custom_pole_residue_from_density(density, coords)
    structure = td.Structure(geometry=box_geom, medium=medium)
    return base_sim.updated_copy(structures=[structure])


def _base_fixed_grid_spec(
    base_sim: td.Simulation,
    box_geom: td.Box,
    params0: anp.ndarray,
) -> td.GridSpec:
    """Use the unperturbed medium grid for all finite-difference simulations."""
    base_param_sim = _add_custom_pole_residue(base_sim, box_geom, np.asarray(params0, dtype=float))
    return td.GridSpec.from_grid(base_param_sim.grid)


def _metric_value(case, dataset, freq0):
    if case["objective_kind"] == "flux":
        return dataset.flux.values.item()
    ex_vals = dataset.Ex.values
    ey_vals = dataset.Ey.values
    ez_vals = dataset.Ez.values
    intensity = np.abs(ex_vals) ** 2 + np.abs(ey_vals) ** 2 + np.abs(ez_vals) ** 2
    return anp.real(anp.mean(intensity))


def _run_simulation(
    case, base_sim, box_geom, params, label, tmp_path, monitor_name, freq0, local_gradient
):
    sim = _add_custom_pole_residue(base_sim, box_geom, params)
    sim_data = web.run(
        sim,
        task_name=f"custom_pole_residue_grad_{case['name']}_{label}",
        local_gradient=local_gradient,
        verbose=False,
        path=str(tmp_path / f"{case['name']}_{label}.hdf5"),
    )
    return _metric_value(case, sim_data[monitor_name], freq0)


def _collect_custom_pole_residue_evaluation_data(
    numerical_case_dir: Path,
    tmp_path: Path,
) -> EvaluationData:
    case = POLE_RESIDUE_CASE
    base_sim, monitor_name, freq0 = _build_base_sim(case)
    box_geom = _box_geometry(case["wavelength"])
    params0 = anp.linspace(0.2, 0.8, num=int(np.prod(DENSITY_SHAPE)))
    fixed_grid_spec = _base_fixed_grid_spec(base_sim, box_geom, params0)
    base_sim_fixed = base_sim.updated_copy(grid_spec=fixed_grid_spec, validate=True)

    def objective(params):
        return _run_simulation(
            case,
            base_sim_fixed,
            box_geom,
            params,
            label="adjoint",
            tmp_path=tmp_path,
            monitor_name=monitor_name,
            freq0=freq0,
            local_gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = get_static(grad_adj).reshape(-1)

    fd_sims = {}
    base_params = get_static(params0)
    for idx in range(base_params.size):
        delta = np.zeros_like(base_params)
        delta[idx] = FD_STEP
        fd_sims[f"fd_plus_{idx}"] = _add_custom_pole_residue(
            base_sim_fixed, box_geom, base_params + delta
        )
        fd_sims[f"fd_minus_{idx}"] = _add_custom_pole_residue(
            base_sim_fixed, box_geom, base_params - delta
        )

    fd_results = web.run_async(
        fd_sims,
        path_dir=str(numerical_case_dir / f"fd_batch_{case['name']}"),
        local_gradient=False,
        verbose=False,
        lazy=False,
    )

    grad_fd = np.zeros_like(grad_adj)
    for idx in range(base_params.size):
        plus = _metric_value(case, fd_results[f"fd_plus_{idx}"][monitor_name], freq0)
        minus = _metric_value(case, fd_results[f"fd_minus_{idx}"][monitor_name], freq0)
        grad_fd[idx] = (plus - minus) / (2.0 * FD_STEP)

    return {
        "grad_adj": np.asarray(grad_adj, dtype=float),
        "grad_fd": np.asarray(grad_fd, dtype=float),
        "params0": np.asarray(base_params, dtype=float),
    }


def _evaluate_custom_pole_residue_evaluation_data(evaluation_data: EvaluationData) -> MetricGroups:
    return evaluate_gradient_angle_agreement(
        np.asarray(evaluation_data["grad_fd"], dtype=float),
        np.asarray(evaluation_data["grad_adj"], dtype=float),
        angle_threshold_deg=ANGLE_TOL,
        metric_name="angle_deg",
    )


def _print_custom_pole_residue_summary(
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    case = POLE_RESIDUE_CASE
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    grad_adj = np.asarray(evaluation_data["grad_adj"], dtype=float)
    grad_fd = np.asarray(evaluation_data["grad_fd"], dtype=float)
    angle_deg = diagnostics["gradient_overlap_deg"]

    print(
        f"[custom-pole-residue-grad-test:{case['name']}] mode={mode_label}, "
        f"adjoint={grad_adj}, finite-difference={grad_fd}, angle_deg={angle_deg:.3f}",
        file=sys.stderr,
    )


@pytest.mark.numerical
def test_custom_pole_residue_grads_match_fd(
    request: pytest.FixtureRequest,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    tmp_path: Path,
    _enable_local_cache: None,
) -> None:
    case_identity = _case_identity(POLE_RESIDUE_CASE)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_custom_pole_residue_evaluation_data(
            numerical_case_dir, tmp_path
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_custom_pole_residue_evaluation_data(evaluation_data)
    )
    _print_custom_pole_residue_summary(evaluation_data, diagnostics, eval_only=numerical_eval_only)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "CustomPoleResidue gradient angle exceeds tolerance; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
