from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import autograd as ag
import autograd.numpy as anp
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


def angled_overlap_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if np.isclose(norm_v1, 0.0) or np.isclose(norm_v2, 0.0):
        if not (np.isclose(norm_v1, 0.0) and np.isclose(norm_v2, 0.0)):
            return np.inf
        return 0.0
    dot = np.sum((v1 / norm_v1) * (v2 / norm_v2))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.arccos(dot) * 180.0 / np.pi)


@dataclass(frozen=True)
class GradientMetrics:
    grad_adjoint: np.ndarray
    grad_fd_half: np.ndarray
    grad_fd: np.ndarray
    grad_fd_double: np.ndarray
    fd_rel_half_vs_nominal: float
    fd_rel_double_vs_nominal: float
    angle_deg: float
    adjoint_norm: float
    fd_norm: float


@dataclass(frozen=True)
class BeamParamCase:
    source_type: str
    param_path: tuple[Any, ...]
    base_value: float
    delta: float
    num_freqs: int
    objective_mode: str
    source_position_case: str

    @property
    def case_name(self) -> str:
        if len(self.param_path) == 1:
            param_tag = str(self.param_path[0])
        else:
            param_tag = f"{self.param_path[0]}_{self.param_path[1]}"
        return (
            f"{self.source_type}_{param_tag}_{self.num_freqs}f_"
            f"{self.objective_mode}_{self.source_position_case}"
        )


SIM_RUN_TIME = 1e-12
NUMERICAL_RESULTS_SUBDIR = "numerical_results"
GAUSS_CASE_DATA_FILENAME = "gaussian_source_gradients_case_data.npz"
GAUSS_CASE_METRICS_FILENAME = "gaussian_source_gradients_case_metrics.json"
GAUSS_SUMMARY_NDJSON_FILENAME = "gaussian_source_gradients_summary.ndjson"
GAUSS_SUMMARY_JSON_FILENAME = "gaussian_source_gradients_summary.json"

GAUSS_NORMAL_AXIS = 2
GAUSS_BASE_FREQ0 = 2.1e14
GAUSS_BASE_FWIDTH = 2.0e13
GAUSS_MIN_STEPS = 26
GAUSS_WVL0 = td.C_0 / GAUSS_BASE_FREQ0
GAUSS_SIM_BUFFER_IN_WVLS = 3.0
GAUSS_MONITOR_CENTER_LATERAL = (0.0, 0.0)
GAUSS_MONITOR_LATERAL_SIZE = (1.2, 1.2)
GAUSS_MONITOR_NAME = "beam_grad_monitor"
GAUSS_ANGLE_LIMIT_DEG = 12.0
GAUSS_NORM_RTOL = 0.3
GAUSS_NORM_ATOL = 1e-6
GAUSS_FD_STABILITY_REL_ERR_MAX = 0.05
GAUSS_DISTANCE_IN_WVLS = 1.0
GAUSS_EQ_MONITOR_DISTANCE = 0.2
GAUSS_EQ_DOT_ATOL = 5e-2
GAUSS_EQ_NORM_RTOL = 8e-2
GAUSS_EQ_FIELD_GRAD_DOT_ATOL = 1e-1
GAUSS_EQ_FIELD_GRAD_NORM_RTOL = 2e-1
GAUSS_CENTER_OBJECTIVE_TARGET = (-0.5 * GAUSS_MONITOR_LATERAL_SIZE[0], -0.5 * GAUSS_WVL0)

GAUSS_MIN_SIM_AXIS = 2.0 * GAUSS_SIM_BUFFER_IN_WVLS * GAUSS_WVL0
GAUSS_BASE_WAIST_RADIUS = 1.5
GAUSS_BASE_WAIST_DISTANCE = -0.25
GAUSS_BASE_WAIST_SIZES = (1.5, 2.0)
GAUSS_BASE_WAIST_DISTANCES = (0.2, -0.18)

GAUSS_ANGLE_THETA_BASE = 0.15
GAUSS_ANGLE_PHI_BASE = 0.2
GAUSS_POL_ANGLE_BASE = 0.35
GAUSS_SIM_BG_INDICES = (1.0, 2.0)


_BASE_BEAM_PARAM_CASES = (
    ("gaussian", ("waist_radius",), GAUSS_BASE_WAIST_RADIUS, 0.025, 1, "single_point"),
    ("gaussian", ("waist_distance",), GAUSS_BASE_WAIST_DISTANCE, 0.025, 1, "single_point"),
    ("gaussian", ("angle_theta",), GAUSS_ANGLE_THETA_BASE, 2 * np.pi / 180.0, 3, "multi_freq_sum"),
    ("gaussian", ("angle_phi",), GAUSS_ANGLE_PHI_BASE, 2 * np.pi / 180.0, 1, "pol_sum"),
    ("gaussian", ("pol_angle",), GAUSS_POL_ANGLE_BASE, 2 * np.pi / 180.0, 1, "pol_sum"),
    ("gaussian", ("center", 1), GAUSS_MONITOR_CENTER_LATERAL[1], 0.02, 1, "center_weighted_single"),
    ("gaussian", ("center", 0), GAUSS_MONITOR_CENTER_LATERAL[0], 0.02, 3, "center_weighted_multi"),
    ("astigmatic", ("waist_sizes", 0), GAUSS_BASE_WAIST_SIZES[0], 0.025, 1, "single_point"),
    ("astigmatic", ("waist_sizes", 1), GAUSS_BASE_WAIST_SIZES[1], 0.025, 3, "multi_freq_sum"),
    ("astigmatic", ("waist_distances", 0), GAUSS_BASE_WAIST_DISTANCES[0], 0.15, 1, "single_point"),
    (
        "astigmatic",
        ("waist_distances", 1),
        GAUSS_BASE_WAIST_DISTANCES[1],
        0.15,
        3,
        "multi_freq_sum",
    ),
    ("astigmatic", ("angle_theta",), GAUSS_ANGLE_THETA_BASE, 2 * np.pi / 180.0, 1, "single_point"),
    ("astigmatic", ("angle_phi",), GAUSS_ANGLE_PHI_BASE, 2 * np.pi / 180.0, 3, "pol_sum"),
    ("astigmatic", ("pol_angle",), GAUSS_POL_ANGLE_BASE, 2 * np.pi / 180.0, 1, "pol_sum"),
    (
        "astigmatic",
        ("center", 1),
        GAUSS_MONITOR_CENTER_LATERAL[1],
        0.02,
        1,
        "center_weighted_single",
    ),
    (
        "astigmatic",
        ("center", 0),
        GAUSS_MONITOR_CENTER_LATERAL[0],
        0.02,
        3,
        "center_weighted_multi",
    ),
)

BEAM_PARAM_CASES = tuple(
    BeamParamCase(*case_values, source_position_case=position_case)
    for case_values in _BASE_BEAM_PARAM_CASES
    for position_case in ("source_before_monitor", "source_after_monitor")
)


def _objective_freqs(freq0: float, mode: str) -> list[float]:
    if mode in ("multi_freq_sum", "center_weighted_multi"):
        return [0.95 * freq0, freq0, 1.05 * freq0]
    return [freq0]


def _apply_beam_param(source: td.Source, param_path: tuple[Any, ...], value: float) -> td.Source:
    root = param_path[0]
    if root == "center":
        center = list(source.center)
        center[param_path[1]] = value
        return source.updated_copy(center=tuple(center), validate=False)
    if root in ("waist_sizes", "waist_distances"):
        vals = list(getattr(source, root))
        vals[param_path[1]] = value
        return source.updated_copy(validate=False, **{root: tuple(vals)})
    return source.updated_copy(validate=False, **{root: value})


def _beam_geometry(
    source_position_case: str, wvl0: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    """Return (source_center, monitor_center, beam_direction) satisfying geometry constraints."""
    if source_position_case not in ("source_before_monitor", "source_after_monitor"):
        raise ValueError(f"Unsupported source position case '{source_position_case}'.")

    distance = GAUSS_DISTANCE_IN_WVLS * wvl0
    monitor_center = [0.0, 0.0, 0.0]
    source_center = [0.0, 0.0, 0.0]

    lateral_axes = [ax for ax in range(3) if ax != GAUSS_NORMAL_AXIS]
    monitor_center[lateral_axes[0]] = GAUSS_MONITOR_CENTER_LATERAL[0]
    monitor_center[lateral_axes[1]] = GAUSS_MONITOR_CENTER_LATERAL[1]
    source_center[lateral_axes[0]] = GAUSS_MONITOR_CENTER_LATERAL[0]
    source_center[lateral_axes[1]] = GAUSS_MONITOR_CENTER_LATERAL[1]

    if source_position_case == "source_before_monitor":
        source_center[GAUSS_NORMAL_AXIS] = monitor_center[GAUSS_NORMAL_AXIS] - distance
    else:
        source_center[GAUSS_NORMAL_AXIS] = monitor_center[GAUSS_NORMAL_AXIS] + distance

    beam_direction = (
        "-" if source_center[GAUSS_NORMAL_AXIS] > monitor_center[GAUSS_NORMAL_AXIS] else "+"
    )
    return tuple(source_center), tuple(monitor_center), beam_direction


def _source_and_monitor_sizes() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return source and monitor sizes with matching normal axis."""
    source_size = [td.inf, td.inf, td.inf]
    source_size[GAUSS_NORMAL_AXIS] = 0.0
    monitor_size = [0.0, 0.0, 0.0]
    lateral_axes = [ax for ax in range(3) if ax != GAUSS_NORMAL_AXIS]
    monitor_size[lateral_axes[0]] = GAUSS_MONITOR_LATERAL_SIZE[0]
    monitor_size[lateral_axes[1]] = GAUSS_MONITOR_LATERAL_SIZE[1]
    return tuple(source_size), tuple(monitor_size)


def _simulation_domain_from_positions(
    source_center: tuple[float, float, float],
    monitor_center: tuple[float, float, float],
    wvl0: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build sim center and size from source/monitor positions with +/- 1 wavelength buffer."""
    buffer = GAUSS_SIM_BUFFER_IN_WVLS * wvl0
    sim_center = []
    sim_size = []
    for axis in range(3):
        pos_min = min(source_center[axis], monitor_center[axis]) - buffer
        pos_max = max(source_center[axis], monitor_center[axis]) + buffer
        sim_center.append(0.5 * (pos_min + pos_max))
        sim_size.append(pos_max - pos_min)
    return tuple(sim_center), tuple(sim_size)


def _make_gaussian_beam_source(case: BeamParamCase, value: float) -> td.Source:
    pulse = td.GaussianPulse(freq0=GAUSS_BASE_FREQ0, fwidth=GAUSS_BASE_FWIDTH)
    wvl0 = td.C_0 / GAUSS_BASE_FREQ0
    source_center, _monitor_center, beam_direction = _beam_geometry(case.source_position_case, wvl0)
    source_size, _monitor_size = _source_and_monitor_sizes()

    if case.source_type == "gaussian":
        source = td.GaussianBeam(
            center=source_center,
            size=source_size,
            source_time=pulse,
            direction=beam_direction,
            angle_theta=0.12,
            angle_phi=0.2,
            pol_angle=0.3,
            waist_radius=GAUSS_BASE_WAIST_RADIUS,
            waist_distance=0.2,
            num_freqs=case.num_freqs,
        )
    elif case.source_type == "astigmatic":
        source = td.AstigmaticGaussianBeam(
            center=source_center,
            size=source_size,
            source_time=pulse,
            direction=beam_direction,
            angle_theta=0.12,
            angle_phi=0.2,
            pol_angle=0.3,
            waist_sizes=GAUSS_BASE_WAIST_SIZES,
            waist_distances=GAUSS_BASE_WAIST_DISTANCES,
            num_freqs=case.num_freqs,
        )
    else:
        raise ValueError(f"Unsupported beam source type '{case.source_type}'.")

    return _apply_beam_param(source, case.param_path, value)


def _make_beam_grad_sim(
    source: td.Source,
    objective_mode: str,
    sim_bg_index: float,
    *,
    fixed_monitor_center: tuple[float, float, float] | None = None,
    fixed_sim_center: tuple[float, float, float] | None = None,
    fixed_sim_size: tuple[float, float, float] | None = None,
) -> td.Simulation:
    freqs = _objective_freqs(GAUSS_BASE_FREQ0, objective_mode)
    wvl0 = td.C_0 / GAUSS_BASE_FREQ0
    distance = GAUSS_DISTANCE_IN_WVLS * wvl0
    if fixed_monitor_center is None:
        monitor_center = [0.0, 0.0, 0.0]
        lateral_axes = [ax for ax in range(3) if ax != GAUSS_NORMAL_AXIS]
        monitor_center[lateral_axes[0]] = GAUSS_MONITOR_CENTER_LATERAL[0]
        monitor_center[lateral_axes[1]] = GAUSS_MONITOR_CENTER_LATERAL[1]
        sign = 1.0 if source.direction == "+" else -1.0
        monitor_center[GAUSS_NORMAL_AXIS] = source.center[GAUSS_NORMAL_AXIS] + sign * distance
        monitor_center = tuple(monitor_center)
    else:
        monitor_center = tuple(fixed_monitor_center)
    _source_size, monitor_size = _source_and_monitor_sizes()
    # Ensure monitor/source share normal axis (both planar in the same direction).
    if monitor_size.index(0.0) != source.size.index(0.0):
        raise ValueError("Monitor and Gaussian source must share the same normal axis.")
    if fixed_sim_center is None or fixed_sim_size is None:
        sim_center, sim_size = _simulation_domain_from_positions(
            source_center=tuple(source.center),
            monitor_center=monitor_center,
            wvl0=wvl0,
        )
    else:
        sim_center, sim_size = tuple(fixed_sim_center), tuple(fixed_sim_size)
    monitor = td.FieldMonitor(
        name=GAUSS_MONITOR_NAME,
        center=monitor_center,
        size=monitor_size,
        freqs=freqs,
    )
    return td.Simulation(
        center=sim_center,
        size=sim_size,
        run_time=SIM_RUN_TIME,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=GAUSS_MIN_STEPS, wavelength=td.C_0 / GAUSS_BASE_FREQ0
        ),
        medium=td.Medium(permittivity=sim_bg_index**2),
        structures=[],
        sources=[source],
        monitors=[monitor],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )


def _eval_beam_objective(sim_data: td.SimulationData, objective_mode: str) -> float:
    field_data = sim_data.load_field_monitor(GAUSS_MONITOR_NAME)
    if objective_mode == "pol_sum":
        ex_values = np.asarray(field_data.Ex.values)
        ey_values = np.asarray(field_data.Ey.values)
        # Polarization-sensitive objective to improve observability of angle_phi/pol_angle.
        return np.sum(np.abs(ex_values) ** 2 - 0.5 * np.abs(ey_values) ** 2)
    if objective_mode in ("center_weighted_single", "center_weighted_multi"):
        intensity_da = field_data.intensity
        # intensity_vals = anp.asarray(intensity_da.values)
        intensity_vals = intensity_da.values
        dim_to_axis = {dim: idx for idx, dim in enumerate(intensity_da.dims)}
        if "x" not in dim_to_axis or "y" not in dim_to_axis:
            raise ValueError(
                "Center-weighted objective requires 'x' and 'y' intensity coordinates."
            )

        ax_x = dim_to_axis["x"]
        ax_y = dim_to_axis["y"]
        intensity_xyfirst = anp.moveaxis(intensity_vals, (ax_x, ax_y), (0, 1))
        if intensity_xyfirst.ndim > 2:
            weight_xy = anp.sum(intensity_xyfirst, axis=tuple(range(2, intensity_xyfirst.ndim)))
        else:
            weight_xy = intensity_xyfirst

        x_coords = anp.asarray(intensity_da.coords["x"].data)  # , dtype=float)
        y_coords = anp.asarray(intensity_da.coords["y"].data)  # , dtype=float)
        total_weight = anp.sum(weight_xy)
        denom = anp.maximum(total_weight, 1e-30)
        x_center = anp.sum(weight_xy * x_coords[:, None]) / denom
        y_center = anp.sum(weight_xy * y_coords[None, :]) / denom

        target_x, target_y = GAUSS_CENTER_OBJECTIVE_TARGET
        return anp.sqrt((x_center - target_x) ** 2 + (y_center - target_y) ** 2)
    if objective_mode in ("single_point", "multi_freq_sum"):
        return np.sum(np.asarray(field_data.intensity.values))
    raise ValueError(f"Unsupported objective_mode '{objective_mode}'.")


def _print_beam_case_summary(
    case: BeamParamCase,
    source: td.Source,
    sim: td.Simulation,
    metrics: GradientMetrics,
    *,
    sim_bg_index: float,
    status: str,
    assertion_error: str = "",
) -> None:
    """Print one end-of-test summary block with setup and error metrics."""
    monitor = sim.monitors[0]
    separation = monitor.center[GAUSS_NORMAL_AXIS] - source.center[GAUSS_NORMAL_AXIS]
    separation_wvls = separation / GAUSS_WVL0
    adjoint_fd_rel_norm_error = np.nan
    if not np.isclose(metrics.fd_norm, 0.0):
        adjoint_fd_rel_norm_error = (metrics.adjoint_norm - metrics.fd_norm) / metrics.fd_norm

    case_lines = [
        "",
        "=" * 92,
        f"Beam Gradient Case: {case.case_name} | sim_bg_index={sim_bg_index} | STATUS: {status}",
        "-" * 92,
        "Setup",
        f"  source_type={case.source_type} | objective_mode={case.objective_mode} | num_freqs={case.num_freqs}",
        f"  traced_param={case.param_path} | base={case.base_value} | delta={case.delta}",
        f"  source_position_case={case.source_position_case} | source_direction={source.direction}",
        f"  source_center={source.center} | source_size={source.size}",
        f"  source_angles(theta, phi, pol)=({source.angle_theta}, {source.angle_phi}, {source.pol_angle})",
    ]
    if case.source_type == "gaussian":
        case_lines.extend(
            [
                f"  source_waist_radius={source.waist_radius} | source_waist_distance={source.waist_distance}",
            ]
        )
    else:
        case_lines.extend(
            [
                f"  source_waist_sizes={source.waist_sizes} | source_waist_distances={source.waist_distances}",
            ]
        )
    case_lines.extend(
        [
            f"  monitor_center={monitor.center} | monitor_size={monitor.size}",
            f"  monitor_freqs={monitor.freqs}",
            f"  source-monitor separation (normal)={separation} | in wavelengths={separation_wvls}",
            f"  sim_center={sim.center} | sim_size={sim.size} | normal_axis={GAUSS_NORMAL_AXIS} | sim_bg_index={sim_bg_index}",
            "Results",
            f"  grad_adjoint={float(metrics.grad_adjoint[0])}",
            f"  grad_fd_half={float(metrics.grad_fd_half[0])}",
            f"  grad_fd_nominal={float(metrics.grad_fd[0])}",
            f"  grad_fd_double={float(metrics.grad_fd_double[0])}",
            f"  fd_rel_err_half_vs_nominal={metrics.fd_rel_half_vs_nominal}",
            f"  fd_rel_err_double_vs_nominal={metrics.fd_rel_double_vs_nominal}",
            f"  adjoint_vs_fd_angle_deg={metrics.angle_deg}",
            f"  adjoint_norm={metrics.adjoint_norm} | fd_norm={metrics.fd_norm}",
            f"  adjoint_vs_fd_rel_norm_error={adjoint_fd_rel_norm_error}",
        ]
    )
    if assertion_error:
        case_lines.extend(
            [
                "Assertion Error",
                f"  {assertion_error}",
            ]
        )
    case_lines.append("=" * 92)
    print("\n".join(case_lines), file=sys.stderr)


def _beam_case_metrics_record(
    case: BeamParamCase,
    source: td.Source,
    sim: td.Simulation,
    metrics: GradientMetrics,
    *,
    sim_bg_index: float,
    status: str,
    assertion_error: str,
) -> dict[str, Any]:
    """Build a compact serializable metrics record for one beam-gradient test case."""
    grad_adjoint = float(metrics.grad_adjoint[0])
    grad_fd_half = float(metrics.grad_fd_half[0])
    grad_fd_nominal = float(metrics.grad_fd[0])
    grad_fd_double = float(metrics.grad_fd_double[0])
    case_name = f"{case.case_name}_n{sim_bg_index:g}"
    return {
        "case_name": case_name,
        "base_case_name": case.case_name,
        "status": status,  # keep pass/fail context for summary parsing
        "assertion_error": assertion_error,
        "sim_bg_index": float(sim_bg_index),
        "grad_adjoint": grad_adjoint,
        "grad_fd_half": grad_fd_half,
        "grad_fd_nominal": grad_fd_nominal,
        "grad_fd_double": grad_fd_double,
    }


def _write_beam_case_artifacts(
    numerical_case_dir: Path,
    numerical_artifact_root: Path,
    *,
    metrics_record: dict[str, Any],
    metrics: GradientMetrics,
) -> None:
    """Persist per-case FD/adjoint data and update summary files."""
    results_dir = numerical_case_dir / NUMERICAL_RESULTS_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / GAUSS_CASE_DATA_FILENAME,
        grad_adjoint=metrics.grad_adjoint,
        grad_fd_half=metrics.grad_fd_half,
        grad_fd_nominal=metrics.grad_fd,
        grad_fd_double=metrics.grad_fd_double,
    )
    (results_dir / GAUSS_CASE_METRICS_FILENAME).write_text(
        json.dumps(metrics_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_ndjson_path = numerical_artifact_root / GAUSS_SUMMARY_NDJSON_FILENAME
    line = json.dumps(metrics_record, sort_keys=True) + "\n"
    fd = os.open(summary_ndjson_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)

    all_records = []
    with summary_ndjson_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                all_records.append(json.loads(raw_line))
            except json.JSONDecodeError:
                # Ignore a partial line if another process is appending concurrently.
                continue
    summary_json_path = numerical_artifact_root / GAUSS_SUMMARY_JSON_FILENAME
    summary_json_path.write_text(
        json.dumps(all_records, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_beam_param_gradient_case(
    tmp_path, case: BeamParamCase, sim_bg_index: float
) -> GradientMetrics:
    case_id = f"{case.case_name}_n{sim_bg_index:g}"
    fixed_monitor_center = None
    fixed_sim_center = None
    fixed_sim_size = None
    if case.param_path and case.param_path[0] == "center":
        source_base = _make_gaussian_beam_source(case, case.base_value)
        sim_base = _make_beam_grad_sim(source_base, case.objective_mode, sim_bg_index)
        fixed_monitor_center = tuple(sim_base.monitors[0].center)
        fixed_sim_center = tuple(sim_base.center)
        fixed_sim_size = tuple(sim_base.size)

    def objective_adj(value: float) -> float:
        source = _make_gaussian_beam_source(case, value)
        sim = _make_beam_grad_sim(
            source,
            case.objective_mode,
            sim_bg_index,
            fixed_monitor_center=fixed_monitor_center,
            fixed_sim_center=fixed_sim_center,
            fixed_sim_size=fixed_sim_size,
        )
        sim_data = web.run(
            sim,
            task_name=f"{case_id}_adj",
            path=tmp_path / f"{case_id}_adj.hdf5",
            local_gradient=True,
            verbose=False,
        )
        return _eval_beam_objective(sim_data, case.objective_mode)

    grad_adjoint = float(ag.grad(objective_adj)(case.base_value))

    delta_half = 0.5 * case.delta
    delta_nominal = case.delta
    delta_double = 2.0 * case.delta
    delta_map = {"half": delta_half, "nominal": delta_nominal, "double": delta_double}

    sims = {}
    for delta_label, delta_value in delta_map.items():
        source_plus = _make_gaussian_beam_source(case, case.base_value + delta_value)
        source_minus = _make_gaussian_beam_source(case, case.base_value - delta_value)
        sims[f"{case_id}_{delta_label}_plus"] = _make_beam_grad_sim(
            source_plus,
            case.objective_mode,
            sim_bg_index,
            fixed_monitor_center=fixed_monitor_center,
            fixed_sim_center=fixed_sim_center,
            fixed_sim_size=fixed_sim_size,
        )
        sims[f"{case_id}_{delta_label}_minus"] = _make_beam_grad_sim(
            source_minus,
            case.objective_mode,
            sim_bg_index,
            fixed_monitor_center=fixed_monitor_center,
            fixed_sim_center=fixed_sim_center,
            fixed_sim_size=fixed_sim_size,
        )
    sim_data_map = web.run_async(
        sims,
        path_dir=tmp_path,
        local_gradient=False,
        verbose=False,
    )
    grad_fd_map = {}
    for delta_label, delta_value in delta_map.items():
        obj_plus = _eval_beam_objective(
            sim_data_map[f"{case_id}_{delta_label}_plus"], case.objective_mode
        )
        obj_minus = _eval_beam_objective(
            sim_data_map[f"{case_id}_{delta_label}_minus"], case.objective_mode
        )
        grad_fd_map[delta_label] = float((obj_plus - obj_minus) / (2.0 * delta_value))
    grad_fd_half = grad_fd_map["half"]
    grad_fd = grad_fd_map["nominal"]
    grad_fd_double = grad_fd_map["double"]

    grad_adjoint_vec = np.asarray([grad_adjoint], dtype=float)
    grad_fd_vec = np.asarray([grad_fd], dtype=float)
    angle_deg = angled_overlap_deg(grad_adjoint_vec, grad_fd_vec)
    adjoint_norm = float(np.linalg.norm(grad_adjoint_vec))
    fd_norm = float(np.linalg.norm(grad_fd_vec))

    fd_rel_half_vs_nominal = np.nan
    if not np.isclose(grad_fd, 0.0):
        fd_rel_half_vs_nominal = (grad_fd_half - grad_fd) / grad_fd
    fd_rel_double_vs_nominal = np.nan
    if not np.isclose(grad_fd, 0.0):
        fd_rel_double_vs_nominal = (grad_fd_double - grad_fd) / grad_fd

    return GradientMetrics(
        grad_adjoint=grad_adjoint_vec,
        grad_fd_half=np.asarray([grad_fd_half], dtype=float),
        grad_fd=grad_fd_vec,
        grad_fd_double=np.asarray([grad_fd_double], dtype=float),
        fd_rel_half_vs_nominal=float(fd_rel_half_vs_nominal),
        fd_rel_double_vs_nominal=float(fd_rel_double_vs_nominal),
        angle_deg=angle_deg,
        adjoint_norm=adjoint_norm,
        fd_norm=fd_norm,
    )


def _assert_beam_fd_agreement(metrics: GradientMetrics, *, label: str) -> None:
    assert metrics.angle_deg < GAUSS_ANGLE_LIMIT_DEG, label
    assert np.isfinite(metrics.adjoint_norm), label
    assert np.isfinite(metrics.fd_norm), label
    np.testing.assert_allclose(
        metrics.adjoint_norm,
        metrics.fd_norm,
        rtol=GAUSS_NORM_RTOL,
        atol=GAUSS_NORM_ATOL,
        err_msg=label,
    )
    # Require central-difference stability when the nominal FD estimate is
    # large enough to define a meaningful relative error.
    for rel_err in (metrics.fd_rel_half_vs_nominal, metrics.fd_rel_double_vs_nominal):
        if np.isfinite(rel_err):
            assert abs(rel_err) <= GAUSS_FD_STABILITY_REL_ERR_MAX, (
                f"{label}: unstable FD estimate (rel_err={rel_err:.6g})"
            )


@pytest.mark.numerical
@pytest.mark.parametrize("case", BEAM_PARAM_CASES, ids=lambda case: case.case_name)
@pytest.mark.parametrize("sim_bg_index", GAUSS_SIM_BG_INDICES, ids=lambda n: f"n{n:g}")
def test_gaussian_beam_parameter_gradients_fd_vs_autograd(
    _enable_local_cache,
    tmp_path,
    numerical_case_dir,
    numerical_artifact_root,
    case,
    sim_bg_index,
    redirect_stdout_to_stderr,
):
    """Finite-difference validation for Gaussian and astigmatic source parameter gradients."""
    source_base = _make_gaussian_beam_source(case, case.base_value)
    sim_base = _make_beam_grad_sim(source_base, case.objective_mode, sim_bg_index)
    metrics = _run_beam_param_gradient_case(tmp_path, case, sim_bg_index)
    status = "PASS"
    assertion_error = ""
    try:
        _assert_beam_fd_agreement(metrics, label=f"{case.case_name}_n{sim_bg_index:g}")
    except AssertionError as exc:
        status = "FAIL"
        assertion_error = str(exc)
        raise
    finally:
        metrics_record = _beam_case_metrics_record(
            case,
            source_base,
            sim_base,
            metrics,
            sim_bg_index=sim_bg_index,
            status=status,
            assertion_error=assertion_error,
        )
        _write_beam_case_artifacts(
            numerical_case_dir,
            numerical_artifact_root,
            metrics_record=metrics_record,
            metrics=metrics,
        )
        _print_beam_case_summary(
            case,
            source_base,
            sim_base,
            metrics,
            sim_bg_index=sim_bg_index,
            status=status,
            assertion_error=assertion_error,
        )
