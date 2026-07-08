"""Numerical validation for multi-frequency custom dispersive medium gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import autograd.numpy as anp
import matplotlib.pyplot as plt
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
    case_identity_id,
    condition_metric,
    coords_for_bounds,
    evaluate_gradient_angle_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)
from .result_models import Metric


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


SIM_SIZE_SCALE = (3.0, 2.5, 3.0)
BOX_SIZE_SCALE = (0.8, 0.8, 0.8)
GRID_STEPS_PER_WVL = 40
RUN_TIME = 2e-13
FD_STEP = 5e-3
ANGLE_TOL = 5.0
PLOT_FD_STEP_SWEEP = False

FREQS = np.array([1.7e14, 2.4e14])
FREQ_WEIGHTS = np.array([1.0, 0.6])

PARAM_SHAPE_2D = (2, 2)
PARAM_SHAPE = (2, 2, 2)
FD_SWEEP_STEPS = np.logspace(-3, -1, num=7)

SELLMEIER_C_VAL = 0.6 * (td.C_0 / np.max(FREQS)) ** 2

TEST_CASES = [
    {
        "name": "lo1",  # keep names short, filenames get too long otherwise
        "kind": "lorentz",
        "eps_inf": 1.6,
        "param0": 0.5,
        "f0": 2.6e14,
        "delta": 0.2e14,
    },
    {
        "name": "lo2",
        "kind": "lorentz",
        "eps_inf": 2.3,
        "param0": 0.7,
        "f0": 2.3e14,
        "delta": 0.2e14,
    },
    {
        "name": "lo3",
        "kind": "lorentz",
        "eps_inf": 1.9,
        "param0": 0.35,
        "f0": 3.0e14,
        "delta": 0.2e14,
    },
    {
        "name": "sl",
        "kind": "sellmeier",
        "param0": 0.6,
        "c_val": SELLMEIER_C_VAL,
    },
    {
        "name": "dd",
        "kind": "drude",
        "eps_inf": 1.6,
        "param0": 0.5,
        "param_scale": 2.0e14,
        "delta": 0.3e14,
    },
    {
        "name": "db",
        "kind": "debye",
        "eps_inf": 2.5,
        "param0": 0.5,
        "tau": 0.4e-14,
    },
    {
        "name": "pr",
        "kind": "pole_residue",
        "eps_inf": 1.6,
        "param0": 0.5,
        "param_scale": 1.0e14,
        "a_val": -1.2e14,
    },
]


class CustomDispersiveCaseIdentity(BaseModel):
    """Semantic identity for one custom dispersive finite-difference case."""

    name: str
    kind: str
    eps_inf: float | None = None
    param0: float
    f0: float | None = None
    delta: float | None = None
    c_val: float | None = None
    param_scale: float | None = None
    tau: float | None = None
    a_val: float | None = None
    freqs: tuple[float, ...]
    freq_weights: tuple[float, ...]
    param_shape_2d: tuple[int, int]
    param_shape: tuple[int, int, int]
    fd_step: float
    angle_tol: float


class CustomDispersiveStepSweepCaseIdentity(CustomDispersiveCaseIdentity):
    """Semantic identity for the CustomLorentz finite-difference step sweep."""

    fd_sweep_steps: tuple[float, ...]


def _case_identity(case: dict[str, object]) -> CustomDispersiveCaseIdentity:
    return CustomDispersiveCaseIdentity(
        name=str(case["name"]),
        kind=str(case["kind"]),
        eps_inf=_optional_float(case.get("eps_inf")),
        param0=float(case["param0"]),
        f0=_optional_float(case.get("f0")),
        delta=_optional_float(case.get("delta")),
        c_val=_optional_float(case.get("c_val")),
        param_scale=_optional_float(case.get("param_scale")),
        tau=_optional_float(case.get("tau")),
        a_val=_optional_float(case.get("a_val")),
        freqs=tuple(float(value) for value in FREQS),
        freq_weights=tuple(float(value) for value in FREQ_WEIGHTS),
        param_shape_2d=PARAM_SHAPE_2D,
        param_shape=PARAM_SHAPE,
        fd_step=FD_STEP,
        angle_tol=ANGLE_TOL,
    )


def _step_sweep_case_identity(case: dict[str, object]) -> CustomDispersiveStepSweepCaseIdentity:
    return CustomDispersiveStepSweepCaseIdentity(
        **_case_identity(case).model_dump(),
        fd_sweep_steps=tuple(float(step) for step in FD_SWEEP_STEPS),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _build_base_sim(freqs: np.ndarray) -> tuple[td.Simulation, str, float]:
    wavelength_min = td.C_0 / np.max(freqs)
    sim_size = tuple(scale * wavelength_min for scale in SIM_SIZE_SCALE)

    freq0 = float(np.mean(freqs))
    fwidth = float(max(freqs.max() - freqs.min(), 0.4 * freq0))

    src = td.PlaneWave(
        center=(0.0, 0.0, -0.75 * sim_size[2] / 2),
        size=(sim_size[0], sim_size[1], 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
        pol_angle=0.0,
    )

    monitor_name = "field_monitor"
    monitor = td.FieldMonitor(
        center=(0.0, 0.0, sim_size[2] / 2 * 0.6),
        size=(sim_size[0], sim_size[1], 0.0),
        freqs=list(freqs),
        name=monitor_name,
        colocate=False,
    )

    sim = td.Simulation(
        size=sim_size,
        center=(0.0, 0.0, 0.0),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=GRID_STEPS_PER_WVL,
            wavelength=wavelength_min,
        ),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[src],
        monitors=[monitor],
        structures=[],
        run_time=RUN_TIME,
    )
    return sim, monitor_name, wavelength_min


def _box_geometry(wavelength_min: float) -> td.Box:
    size = tuple(scale * wavelength_min for scale in BOX_SIZE_SCALE)
    return td.Box(size=size, center=(0.0, 0.0, 0.0))


def _custom_medium(case, param_vals: anp.ndarray, box_geom: td.Box):
    bounds = box_geom.bounds
    coords = coords_for_bounds(bounds, param_vals.shape)
    kind = case["kind"]
    param_scale = case.get("param_scale", 1.0)
    scaled = param_scale * param_vals

    if kind == "lorentz":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        de = td.SpatialDataArray(scaled, coords=coords)
        f0 = td.SpatialDataArray(np.full(param_vals.shape, case["f0"]), coords=coords)
        delta = td.SpatialDataArray(np.full(param_vals.shape, case["delta"]), coords=coords)
        return td.CustomLorentz(eps_inf=eps_inf, coeffs=[(de, f0, delta)])
    if kind == "sellmeier":
        b = td.SpatialDataArray(scaled, coords=coords)
        c = td.SpatialDataArray(np.full(param_vals.shape, case["c_val"]), coords=coords)
        return td.CustomSellmeier(coeffs=[(b, c)])
    if kind == "drude":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        fp = td.SpatialDataArray(scaled, coords=coords)
        delta = td.SpatialDataArray(np.full(param_vals.shape, case["delta"]), coords=coords)
        return td.CustomDrude(eps_inf=eps_inf, coeffs=[(fp, delta)])
    if kind == "debye":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        de = td.SpatialDataArray(scaled, coords=coords)
        tau = td.SpatialDataArray(np.full(param_vals.shape, case["tau"]), coords=coords)
        return td.CustomDebye(eps_inf=eps_inf, coeffs=[(de, tau)])
    if kind == "pole_residue":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        a_val = td.SpatialDataArray(np.full(param_vals.shape, case["a_val"]), coords=coords)
        c_val = td.SpatialDataArray(scaled, coords=coords)
        return td.CustomPoleResidue(eps_inf=eps_inf, poles=[(a_val, c_val)])
    raise ValueError(f"Unsupported medium kind: {kind}")


def _add_medium(
    sim: td.Simulation, box_geom: td.Box, case, param_vals: anp.ndarray
) -> td.Simulation:
    medium = _custom_medium(case, param_vals, box_geom)
    structure = td.Structure(geometry=box_geom, medium=medium)
    return sim.updated_copy(structures=[structure])


def _base_fixed_grid_spec(
    base_sim: td.Simulation,
    box_geom: td.Box,
    case: dict[str, object],
    params0: anp.ndarray,
) -> td.GridSpec:
    """Use the unperturbed medium grid for all finite-difference simulations."""
    base_param_vals = _expand_params(np.asarray(params0, dtype=float))
    base_param_sim = _add_medium(base_sim, box_geom, case, base_param_vals)
    return td.GridSpec.from_grid(base_param_sim.grid)


def _metric_value(dataset) -> float:
    ex_vals = dataset.Ex.values
    ey_vals = dataset.Ey.values
    ez_vals = dataset.Ez.values
    intensity = anp.abs(ex_vals) ** 2 + anp.abs(ey_vals) ** 2 + anp.abs(ez_vals) ** 2
    weighted = intensity * anp.asarray(FREQ_WEIGHTS)
    return anp.real(anp.mean(weighted))


def _expand_params(params: anp.ndarray) -> anp.ndarray:
    vals_2d = anp.reshape(params, PARAM_SHAPE_2D)
    return anp.repeat(vals_2d[..., None], PARAM_SHAPE[2], axis=2)


def _run_simulation(
    sim: td.Simulation,
    monitor_name: str,
    tmp_path,
    label: str,
    local_gradient: bool,
) -> float:
    sim_data = web.run(
        sim,
        task_name=f"custom_disp_{label}",
        local_gradient=local_gradient,
        verbose=False,
        path=str(tmp_path / f"custom_disp_{label}.hdf5"),
    )
    return _metric_value(sim_data[monitor_name])


def _collect_custom_dispersive_evaluation_data(
    case: dict[str, object],
    numerical_case_dir: Path,
    tmp_path: Path,
) -> EvaluationData:
    base_sim, monitor_name, wavelength_min = _build_base_sim(FREQS)
    box_geom = _box_geometry(wavelength_min)

    params0 = anp.full(PARAM_SHAPE_2D, case["param0"]).reshape(-1)
    fixed_grid_spec = _base_fixed_grid_spec(base_sim, box_geom, case, params0)
    base_sim_fixed = base_sim.updated_copy(grid_spec=fixed_grid_spec, validate=True)

    def objective(param_vec):
        param_vals = _expand_params(param_vec)
        sim = _add_medium(base_sim_fixed, box_geom, case, param_vals)
        return _run_simulation(
            sim=sim,
            monitor_name=monitor_name,
            tmp_path=tmp_path,
            label="adjoint",
            local_gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = np.asarray(get_static(grad_adj), dtype=float).reshape(-1)

    fd_sims: dict[str, td.Simulation] = {}
    for idx in range(params0.size):
        delta = np.zeros_like(params0)
        delta[idx] = FD_STEP
        plus_vals = _expand_params(params0 + delta)
        minus_vals = _expand_params(params0 - delta)
        fd_sims[f"plus_{idx}"] = _add_medium(base_sim_fixed, box_geom, case, plus_vals)
        fd_sims[f"minus_{idx}"] = _add_medium(base_sim_fixed, box_geom, case, minus_vals)

    fd_results = web.run_async(
        fd_sims,
        path_dir=str(numerical_case_dir / f"{case['name']}"),
        local_gradient=False,
        verbose=False,
        lazy=False,
    )

    grad_fd = np.zeros_like(grad_adj)
    for idx in range(params0.size):
        val_plus = _metric_value(fd_results[f"plus_{idx}"][monitor_name])
        val_minus = _metric_value(fd_results[f"minus_{idx}"][monitor_name])
        grad_fd[idx] = (val_plus - val_minus) / (2.0 * FD_STEP)

    return {
        "grad_adj": np.asarray(grad_adj, dtype=float),
        "grad_fd": np.asarray(grad_fd, dtype=float),
        "params0": np.asarray(params0, dtype=float),
    }


def _evaluate_custom_dispersive_evaluation_data(evaluation_data: EvaluationData) -> MetricGroups:
    return evaluate_gradient_angle_agreement(
        np.asarray(evaluation_data["grad_fd"], dtype=float),
        np.asarray(evaluation_data["grad_adj"], dtype=float),
        angle_threshold_deg=ANGLE_TOL,
        metric_name="angle_deg",
    )


def _print_custom_dispersive_summary(
    case: dict[str, object],
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    grad_adj = np.asarray(evaluation_data["grad_adj"], dtype=float)
    grad_fd = np.asarray(evaluation_data["grad_fd"], dtype=float)
    angle_deg = diagnostics["gradient_overlap_deg"]
    print(
        (
            f"[custom-dispersive-multifreq:{case['name']}] mode={mode_label}, "
            f"adjoint={grad_adj}, "
            f"finite-difference={grad_fd}, angle_deg={angle_deg:.3f}"
        ),
        file=sys.stderr,
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=lambda case: case_identity_id(_case_identity(case), prefix="custom-disp"),
)
def test_custom_dispersive_multifreq_grad_matches_fd(
    request: pytest.FixtureRequest,
    case: dict[str, object],
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    tmp_path: Path,
    _enable_local_cache: None,
) -> None:
    case_identity = _case_identity(case)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_custom_dispersive_evaluation_data(
            case, numerical_case_dir, tmp_path
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_custom_dispersive_evaluation_data(evaluation_data)
    )
    _print_custom_dispersive_summary(
        case, evaluation_data, diagnostics, eval_only=numerical_eval_only
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Multi-frequency CustomDispersive gradient angle exceeds tolerance; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


def _collect_custom_lorentz_step_sweep_evaluation_data(
    numerical_case_dir: Path,
    tmp_path: Path,
) -> EvaluationData:
    base_sim, monitor_name, wavelength_min = _build_base_sim(FREQS)
    box_geom = _box_geometry(wavelength_min)

    case = TEST_CASES[0]
    params0 = anp.full(PARAM_SHAPE_2D, case["param0"]).reshape(-1)
    fixed_grid_spec = _base_fixed_grid_spec(base_sim, box_geom, case, params0)
    base_sim_fixed = base_sim.updated_copy(grid_spec=fixed_grid_spec, validate=True)

    def objective(de_params):
        de_vals = _expand_params(de_params)
        sim = _add_medium(base_sim_fixed, box_geom, case, de_vals)
        return _run_simulation(
            sim=sim,
            monitor_name=monitor_name,
            tmp_path=tmp_path,
            label="adjoint_sweep",
            local_gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = np.asarray(get_static(grad_adj), dtype=float).reshape(-1)

    sweep_runs: dict[str, td.Simulation] = {}
    step_labels = [f"{step:.3e}" for step in FD_SWEEP_STEPS]
    for step_label, step in zip(step_labels, FD_SWEEP_STEPS):
        plus_vals = _expand_params(params0 + step)
        minus_vals = _expand_params(params0 - step)
        sweep_runs[f"step_{step_label}_plus"] = _add_medium(
            base_sim_fixed, box_geom, case, plus_vals
        )
        sweep_runs[f"step_{step_label}_minus"] = _add_medium(
            base_sim_fixed, box_geom, case, minus_vals
        )

    sweep_results = web.run_async(
        sweep_runs,
        path_dir=str(numerical_case_dir / f"fd_sweep_{case['name']}"),
        local_gradient=False,
        verbose=False,
        lazy=False,
    )

    fd_sweep = []
    for step_label, step in zip(step_labels, FD_SWEEP_STEPS):
        plus_key = f"step_{step_label}_plus"
        minus_key = f"step_{step_label}_minus"
        plus_val = _metric_value(sweep_results[plus_key][monitor_name])
        minus_val = _metric_value(sweep_results[minus_key][monitor_name])
        fd_sweep.append((plus_val - minus_val) / (2.0 * step))

    fd_sweep = np.array(fd_sweep, dtype=float)

    return {
        "grad_adj": np.asarray(grad_adj, dtype=float),
        "fd_sweep": fd_sweep,
        "fd_sweep_steps": np.asarray(FD_SWEEP_STEPS, dtype=float),
        "params0": np.asarray(params0, dtype=float),
    }


def _evaluate_custom_lorentz_step_sweep_evaluation_data(
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric], dict[str, float]]:
    grad_adj = np.asarray(evaluation_data["grad_adj"], dtype=float)
    fd_sweep = np.asarray(evaluation_data["fd_sweep"], dtype=float)
    fd_min = float(np.min(fd_sweep))
    fd_max = float(np.max(fd_sweep))
    grad_adj_mean = float(np.mean(grad_adj))
    fd_sweep_span = float(fd_max - fd_min)
    fd_sweep_abs_max = float(np.max(np.abs(fd_sweep)))
    grad_adj_abs_mean = float(np.mean(np.abs(grad_adj)))

    regression_metrics = [
        condition_metric("adjoint_gradient_finite", np.all(np.isfinite(grad_adj))),
        condition_metric("fd_step_sweep_finite", np.all(np.isfinite(fd_sweep))),
    ]
    observation_metrics = [
        Metric(name="fd_sweep_span", observed=fd_sweep_span, expected=0.0, comparator="gte"),
        Metric(name="fd_sweep_abs_max", observed=fd_sweep_abs_max, expected=0.0, comparator="gte"),
        Metric(
            name="adjoint_gradient_abs_mean",
            observed=grad_adj_abs_mean,
            expected=0.0,
            comparator="gte",
        ),
    ]
    diagnostics = {
        "fd_min": fd_min,
        "fd_max": fd_max,
        "grad_adj_mean": grad_adj_mean,
    }
    return regression_metrics, observation_metrics, diagnostics


def _plot_custom_lorentz_step_sweep(
    evaluation_data: EvaluationData, numerical_case_dir: Path
) -> None:
    grad_adj = np.asarray(evaluation_data["grad_adj"], dtype=float)
    fd_sweep = np.asarray(evaluation_data["fd_sweep"], dtype=float)
    fd_sweep_steps = np.asarray(evaluation_data["fd_sweep_steps"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fd_sweep_steps, fd_sweep, marker="o", label="FD")
    ax.axhline(
        np.mean(grad_adj),
        color=ax.get_lines()[-1].get_color(),
        linestyle="--",
        alpha=0.7,
        label="Adjoint (mean)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step")
    ax.set_ylabel("Gradient value")
    ax.set_title("CustomLorentz FD sweep")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig_path = numerical_case_dir / "custom_lorentz_fd_step_sweep.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def _print_custom_lorentz_step_sweep_summary(
    evaluation_data: EvaluationData,
    diagnostics: dict[str, float],
    *,
    eval_only: bool,
) -> None:
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    print(
        (
            f"[custom-dispersive-fd-sweep] mode={mode_label}, "
            f"grad_adj={np.asarray(evaluation_data['grad_adj'], dtype=float)} "
            f"fd_grad[min,max]=({diagnostics['fd_min']:.6e},{diagnostics['fd_max']:.6e})"
        ),
        file=sys.stderr,
    )


@pytest.mark.numerical
def test_custom_lorentz_fd_step_sweep(
    request: pytest.FixtureRequest,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    tmp_path: Path,
    _enable_local_cache: None,
) -> None:
    case_identity = _step_sweep_case_identity(TEST_CASES[0])
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_custom_lorentz_step_sweep_evaluation_data(
            numerical_case_dir, tmp_path
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_custom_lorentz_step_sweep_evaluation_data(evaluation_data)
    )
    _print_custom_lorentz_step_sweep_summary(
        evaluation_data, diagnostics, eval_only=numerical_eval_only
    )

    if PLOT_FD_STEP_SWEEP:
        _plot_custom_lorentz_step_sweep(evaluation_data, numerical_case_dir)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "CustomLorentz finite-difference step sweep produced nonfinite data; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
