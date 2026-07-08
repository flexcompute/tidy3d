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
    finalize_result,
    load_or_collect_evaluation_data,
    scale_length_by_wavelength,
)
from .result_models import Metric


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


SIM_SIZE_SCALE = (4, 3, 4)
BOX_SIZE_SCALE = (1, 1, 1)
GRID_STEPS_PER_WVL = 30
RUN_TIME = 2e-12
ANGLE_TOL = 10.0
FD_STEP = 5e-2

TEST_CASES = [
    {
        "name": "opt_flux_iso",
        "wavelength": 1.0,
        "permittivities": (2.2, 2.2, 2.2),
        "objective_kind": "flux",
        "monitor_size": (np.inf, np.inf, 0.0),
        "polarization": 0.0,
        "medium_type": "isotropic",
    },
    {
        "name": "mw_intensity_iso",
        "wavelength": 1.6,
        "permittivities": (1.8, 1.8, 1.8),
        "objective_kind": "intensity",
        "monitor_size": (0.4, 0.4, 0.0),
        "polarization": np.pi / 5,
        "medium_type": "isotropic",
    },
    {
        "name": "opt_flux_custom_iso",
        "wavelength": 1.3,
        "permittivities": (2.0, 2.0, 2.0),
        "objective_kind": "flux",
        "monitor_size": (np.inf, np.inf, 0.0),
        "polarization": 0.0,
        "medium_type": "custom",
    },
    {
        "name": "mw_int_custom_iso",
        "wavelength": 1.1,
        "permittivities": (1.6, 1.6, 1.6),
        "objective_kind": "intensity",
        "monitor_size": (0.3, 0.3, 0.0),
        "polarization": np.pi / 3,
        "medium_type": "custom",
    },
    {
        "name": "opt_flux_aniso",
        "wavelength": 1.0,
        "permittivities": (2.1, 2.4, 2.7),
        "objective_kind": "flux",
        "monitor_size": (np.inf, np.inf, 0.0),
        "polarization": 0.0,
        "medium_type": "anisotropic",
    },
    {
        "name": "opt_int_point_aniso",
        "wavelength": 1.55,
        "permittivities": (1.6, 2.0, 2.5),
        "objective_kind": "intensity",
        "monitor_size": (0.0, 0.0, 0.0),
        "polarization": np.pi / 2,
        "medium_type": "anisotropic",
    },
    {
        "name": "mw_flux_aniso",
        "wavelength": 0.8,
        "permittivities": (2.2, 2.4, 1.5),
        "objective_kind": "flux",
        "monitor_size": (np.inf, np.inf, 0.0),
        "polarization": np.pi / 8,
        "medium_type": "anisotropic",
    },
    {
        "name": "mw_int_plane_aniso",
        "wavelength": 2.1,
        "permittivities": (2.6, 2.0, 3.1),
        "objective_kind": "intensity",
        "monitor_size": (0.2, 0.2, 0.0),
        "polarization": np.pi / 4,
        "medium_type": "anisotropic",
    },
    {
        "name": "opt_flux_custom_aniso",
        "wavelength": 1.2,
        "permittivities": (1.8, 2.3, 2.9),
        "objective_kind": "flux",
        "monitor_size": (np.inf, np.inf, 0.0),
        "polarization": 0.0,
        "medium_type": "custom_anisotropic",
    },
    {
        "name": "mw_int_custom_aniso",
        "wavelength": 1.9,
        "permittivities": (1.4, 2.0, 1.7),
        "objective_kind": "intensity",
        "monitor_size": (0.5, 0.5, 0.0),
        "polarization": np.pi / 6,
        "medium_type": "custom_anisotropic",
    },
]


class MediumGradientCaseIdentity(BaseModel):
    """Semantic identity for one medium-gradient numerical case."""

    name: str
    wavelength: float
    permittivities: tuple[float, float, float]
    objective_kind: str
    monitor_size: tuple[float, float, float]
    polarization: float
    medium_type: str


def _medium_case_identity(case) -> MediumGradientCaseIdentity:
    """Build the semantic case identity used for eval-only replay validation."""
    return MediumGradientCaseIdentity(
        name=case["name"],
        wavelength=case["wavelength"],
        permittivities=case["permittivities"],
        objective_kind=case["objective_kind"],
        monitor_size=case["monitor_size"],
        polarization=case["polarization"],
        medium_type=case["medium_type"],
    )


def _box_geometry(case) -> td.Box:
    size = tuple(scale * case["wavelength"] for scale in BOX_SIZE_SCALE)
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


def _add_medium(case, base_sim: td.Simulation, box_geom: td.Box, eps_vals) -> td.Simulation:
    medium_type = case["medium_type"]

    coords = None
    factor = None
    if medium_type in ("custom_anisotropic", "custom"):
        coords = {
            "x": np.linspace(-box_geom.size[0] / 2, box_geom.size[0] / 2, 4),
            "y": np.linspace(-box_geom.size[1] / 2, box_geom.size[1] / 2, 5),
            "z": np.linspace(-box_geom.size[2] / 2, box_geom.size[2] / 2, 3),
        }
        _cx, _cy, _cz = np.meshgrid(coords["x"], coords["y"], coords["z"], indexing="ij")
        factor = 1 + 0.2 * (_cx + _cy + _cz) / 3.0

    if medium_type == "custom_anisotropic":

        def _custom_medium(val):
            values = factor * val
            data = td.SpatialDataArray(values, coords=coords)
            return td.CustomMedium(permittivity=data)

        medium = td.CustomAnisotropicMedium(
            xx=_custom_medium(eps_vals[0]),
            yy=_custom_medium(eps_vals[1]),
            zz=_custom_medium(eps_vals[2]),
        )
    elif medium_type == "custom":

        def _custom_isotropic(val):
            values = factor * val
            data = td.SpatialDataArray(values, coords=coords)
            return td.CustomMedium(permittivity=data)

        medium = _custom_isotropic(eps_vals[0])
    elif medium_type == "isotropic":
        # use first entry; others are identical by construction
        medium = td.Medium(permittivity=eps_vals[0])
    elif medium_type == "anisotropic":
        medium = td.AnisotropicMedium(
            xx=td.Medium(permittivity=eps_vals[0]),
            yy=td.Medium(permittivity=eps_vals[1]),
            zz=td.Medium(permittivity=eps_vals[2]),
        )
    else:
        raise ValueError(
            "Medium type has to be one of 'custom', 'isotropic', 'anisotropic' or 'custom_anisotropic'"
        )

    structure = td.Structure(geometry=box_geom, medium=medium)
    return base_sim.updated_copy(structures=[structure])


def _metric_value(case, dataset, freq0):
    if case["objective_kind"] == "flux":
        return dataset.flux.values.item()
    ex_vals = dataset.Ex.values
    ey_vals = dataset.Ey.values
    ez_vals = dataset.Ez.values
    intensity = np.abs(ex_vals) ** 2 + np.abs(ey_vals) ** 2 + np.abs(ez_vals) ** 2
    return anp.real(anp.mean(intensity))


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    cos_theta = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _run_simulation(
    case, base_sim, box_geom, eps_vals, label, tmp_path, monitor_name, freq0, gradient
):
    sim = _add_medium(case, base_sim, box_geom, eps_vals)
    sim_data = web.run(
        sim,
        task_name=f"medium_grad_{case['name']}_{label}",
        local_gradient=gradient,
        verbose=False,
        path=str(tmp_path / f"{case['name']}_{label}.hdf5"),
    )
    return _metric_value(case, sim_data[monitor_name], freq0)


def _collect_medium_gradient_evaluation_data(
    case,
    numerical_case_dir: Path,
    tmp_path: Path,
) -> EvaluationData:
    """Collect adjoint and finite-difference gradients for one medium case."""
    base_sim, monitor_name, freq0 = _build_base_sim(case)
    box_geom = _box_geometry(case)
    params0 = anp.array(case["permittivities"])

    def objective(eps_vals):
        return _run_simulation(
            case,
            base_sim,
            box_geom,
            eps_vals,
            label="adjoint",
            tmp_path=tmp_path,
            monitor_name=monitor_name,
            freq0=freq0,
            gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = get_static(grad_adj)

    fd_sims = {}
    base_params = get_static(params0)
    for axis in range(3):
        delta = np.zeros_like(base_params)
        delta[axis] = FD_STEP
        fd_sims[f"fd_plus_{axis}"] = _add_medium(case, base_sim, box_geom, base_params + delta)
        fd_sims[f"fd_minus_{axis}"] = _add_medium(case, base_sim, box_geom, base_params - delta)

    if case["medium_type"] in ("custom", "custom_anisotropic"):
        fd_results = web.Batch(simulations=fd_sims, verbose=False).run(
            path_dir=str(numerical_case_dir / f"fd_batch_{case['name']}")
        )
    else:
        fd_results = web.run_async(
            fd_sims,
            path_dir=str(numerical_case_dir / f"fd_batch_{case['name']}"),
            local_gradient=False,
            verbose=False,
        )

    grad_fd = np.zeros_like(grad_adj)
    for axis in range(3):
        plus = _metric_value(case, fd_results[f"fd_plus_{axis}"][monitor_name], freq0)
        minus = _metric_value(case, fd_results[f"fd_minus_{axis}"][monitor_name], freq0)
        grad_fd[axis] = (plus - minus) / (2.0 * FD_STEP)

    return {
        "grad_adj": np.asarray(grad_adj, dtype=float),
        "grad_fd": np.asarray(grad_fd, dtype=float),
    }


def _evaluate_medium_gradient_evaluation_data(
    case, evaluation_data: EvaluationData
) -> MetricGroups:
    """Evaluate saved-or-fresh medium-gradient data into RFC-style metrics."""
    grad_adj = np.asarray(evaluation_data["grad_adj"], dtype=float)
    grad_fd = np.asarray(evaluation_data["grad_fd"], dtype=float)
    angle_deg = _angle_deg(grad_adj, grad_fd)
    angle_tol = case.get("angle_tol_deg", ANGLE_TOL)
    angle_passes = np.isnan(angle_deg) or angle_deg <= angle_tol

    if np.isnan(angle_deg):
        regression_metrics = [condition_metric("angle_deg_lte_or_nan", angle_passes)]
    else:
        regression_metrics = [
            Metric(
                name="angle_deg",
                observed=float(angle_deg),
                expected=float(angle_tol),
                comparator="lte",
            )
        ]

    diagnostics = {
        "angle_deg": float(angle_deg),
        "angle_tol": float(angle_tol),
        "adj_mag": float(np.linalg.norm(grad_adj)),
        "fd_mag": float(np.linalg.norm(grad_fd)),
    }
    return regression_metrics, [], diagnostics


def _print_medium_gradient_summary(
    case,
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    """Print the existing medium-gradient comparison summary."""
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    print(
        f"[medium-grad-test:{case['name']}] mode={mode_label}, "
        f"adjoint={evaluation_data['grad_adj']}, "
        f"finite-difference={evaluation_data['grad_fd']}, "
        f"angle_deg={diagnostics['angle_deg']:.3f}",
        file=sys.stderr,
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=lambda case: case_identity_id(_medium_case_identity(case), prefix="medium"),
)
def test_medium_grads_match_fd(
    request: pytest.FixtureRequest,
    case,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    tmp_path: Path,
    _enable_local_cache,
):
    case_identity = _medium_case_identity(case)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_medium_gradient_evaluation_data(
            case, numerical_case_dir, tmp_path
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_medium_gradient_evaluation_data(case, evaluation_data)
    )
    _print_medium_gradient_summary(
        case,
        evaluation_data,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Gradient angle deviation exceeds tolerance; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


@pytest.mark.skip
@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=lambda case: case_identity_id(_medium_case_identity(case), prefix="medium"),
)
def test_medium_fd_step_sweep(case, numerical_case_dir, tmp_path, _enable_local_cache):
    base_sim, monitor_name, freq0 = _build_base_sim(case)
    box_geom = _box_geometry(case)
    params0 = anp.array(case["permittivities"])

    def objective(eps_vals):
        return _run_simulation(
            case,
            base_sim,
            box_geom,
            eps_vals,
            label="adjoint_sweep",
            tmp_path=tmp_path,
            monitor_name=monitor_name,
            freq0=freq0,
            gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = get_static(grad_adj)
    base_params = get_static(params0)

    sweep_steps = np.logspace(-4, -1, num=9)
    step_labels = [f"{step:.3e}" for step in sweep_steps]

    sweep_runs: dict[str, td.Simulation] = {}
    for step_label, step in zip(step_labels, sweep_steps):
        for axis in range(base_params.size):
            delta = np.zeros_like(base_params)
            delta[axis] = step
            key_base = f"{case['name']}_axis{axis}_{step_label}"
            sweep_runs[f"{key_base}_plus"] = _add_medium(
                case,
                base_sim,
                box_geom,
                base_params + delta,
            )
            sweep_runs[f"{key_base}_minus"] = _add_medium(
                case,
                base_sim,
                box_geom,
                base_params - delta,
            )

    if case["medium_type"] in ("custom", "custom_anisotropic"):
        sweep_results = web.Batch(simulations=sweep_runs, verbose=False).run(
            path_dir=str(numerical_case_dir / f"fd_sweep_{case['name']}")
        )
    else:
        sweep_results = web.run_async(
            sweep_runs,
            path_dir=str(numerical_case_dir / f"fd_sweep_{case['name']}"),
            local_gradient=False,
            verbose=False,
        )

    fd_sweep_matrix = np.zeros((len(sweep_steps), base_params.size), dtype=float)
    for step_idx, (step_label, step) in enumerate(zip(step_labels, sweep_steps)):
        for axis in range(base_params.size):
            plus_key = f"{case['name']}_axis{axis}_{step_label}_plus"
            minus_key = f"{case['name']}_axis{axis}_{step_label}_minus"
            plus_val = _metric_value(case, sweep_results[plus_key][monitor_name], freq0)
            minus_val = _metric_value(case, sweep_results[minus_key][monitor_name], freq0)
            fd_sweep_matrix[step_idx, axis] = (plus_val - minus_val) / (2.0 * step)

    labels = ["xx", "yy", "zz"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for axis, label in enumerate(labels[: base_params.size]):
        ax.plot(sweep_steps, fd_sweep_matrix[:, axis], marker="o", label=f"{label} (FD)")
        color = ax.get_lines()[-1].get_color()
        ax.axhline(
            grad_adj[axis],
            color=color,
            linestyle="--",
            alpha=0.7,
            label=f"{label} (autograd)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step")
    ax.set_ylabel("Gradient value")
    ax.set_title(f"FD gradients vs. step size ({case['name']})")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig_path = numerical_case_dir / f"medium_fd_step_sweep_{case['name']}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # FD gradient extrema per parameter (across all step sizes)
    fd_min_per_param = fd_sweep_matrix.min(axis=0)
    fd_max_per_param = fd_sweep_matrix.max(axis=0)

    print(
        (
            f"[medium-fd-sweep:{case['name']}] "
            f"grad_adj={np.array2string(grad_adj, precision=6, separator=', ')} "
            f"fd_grad_per_param[min,max]="
            f"{[(f'({mn:.3e},{mx:.3e})') for mn, mx in zip(fd_min_per_param, fd_max_per_param)]}"
        ),
        file=sys.stderr,
    )
