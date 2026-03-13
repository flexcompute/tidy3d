from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Callable

import autograd as ag
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


CENTER_ANGLE_LIMIT_DEG = 5.0
CENTER_NORM_RTOL = 0.18
CENTER_NORM_ATOL = 1e-6
DATASET_ANGLE_LIMIT_DEG = 5.0
DATASET_NORM_RTOL = 0.15
DATASET_NORM_ATOL = 1e-6

BASE_WVL0 = 2.0
BASE_MIN_STEPS_PER_WVL = 40
BASE_DATASET_SPACING = 0.125
BASE_SOURCE_SIZE = (1.0, 1.0, 0.0)
BASE_SOURCE_CENTER = (0.1, 0.4, -0.2)
BASE_PARAM_AMPLITUDES = (1.0, -0.5, 0.25)

SIM_RUN_TIME = 1e-12
MONITOR_CENTER = (-0.3, 0.1, 0.2)
MONITOR_SIZE = (0.5, 0.5, 0.0)
FLUX_MONITOR_NAME = "flux_monitor"
FAR_CORNER_BLOCK_SIZE = (0.1, 0.1, 0.1)
FAR_CORNER_BLOCK_PERMITTIVITY = 100.0
FAR_CORNER_BLOCK_OFFSET_FACTOR = 1.35

DATASET_SPACING_VALUES = (0.25, 0.125, 0.0625)
MIN_STEPS_PER_WVL_VALUES = (40, 60, 80)
WVL0_VALUES = (0.5, 1.0, 2.0)
SOURCE_SIZE_XY_VALUES = (0.5, 1.0, 2.0)
AMPLITUDE_SCALE_VALUES = (0.25, 0.5, 1.0)
# Tuple format is (background_permittivity, source_structure_permittivity).
# ``None`` means "no enclosing structure around the source", so this sweep varies
# only the simulation background medium.
PERMITTIVITY_VALUES = (
    (1.0, None),
    (2.0, None),
    (4.0, None),
)
SIM_DIMS_VALUES = (3, 2)
OBJECTIVE_3D_MODE = (
    "flux"  # or "intensity" # flux turned out to be more stable in finite difference gradient
)
OFF_CENTER_MONITOR_FREQ_SCALE = 0.95
TWO_FREQ_MONITOR_SCALES = (1.0, OFF_CENTER_MONITOR_FREQ_SCALE)
DERIVATIVE_TARGETS = ("dataset", "center")


def _axis_coords(size: float, spacing: float) -> np.ndarray:
    size = float(get_static(size))
    if size <= 0:
        return np.array([0.0], dtype=float)
    n = max(2, int(np.round(size / spacing)))
    return np.linspace(-size / 2, size / 2, n)


def _make_coords(
    source_size: tuple[float, float, float], dataset_spacing: float, freq0: float
) -> dict[str, object]:
    return {
        "x": get_static(_axis_coords(source_size[0], dataset_spacing)),
        "y": get_static(_axis_coords(source_size[1], dataset_spacing)),
        "z": get_static(_axis_coords(source_size[2], dataset_spacing)),
        "f": [float(get_static(freq0))],
    }


def _make_field_dataset(
    field_prefix: str,
    amplitudes: tuple[float, float, float],
    coords: dict[str, object],
) -> td.FieldDataset:
    x = np.asarray(coords["x"], dtype=float)
    y = np.asarray(coords["y"], dtype=float)
    z = np.asarray(coords["z"], dtype=float)
    shape = (
        len(x),
        len(y),
        len(z),
        1,
    )

    def _axis_scale(values: np.ndarray) -> np.ndarray:
        if values.size <= 1:
            return np.ones_like(values, dtype=float)
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if np.isclose(v_min, v_max):
            return np.ones_like(values, dtype=float)
        t = (values - v_min) / (v_max - v_min)
        return 0.5 + 1.5 * t

    sx = _axis_scale(x)[:, None, None]
    sy = _axis_scale(y)[None, :, None]
    sz = _axis_scale(z)[None, None, :]
    profile = np.zeros((len(x), len(y), len(z)), dtype=float)
    n_active_axes = 0
    if len(x) > 1:
        profile += sx
        n_active_axes += 1
    if len(y) > 1:
        profile += sy
        n_active_axes += 1
    if len(z) > 1:
        profile += sz
        n_active_axes += 1
    if n_active_axes == 0:
        profile = np.ones((len(x), len(y), len(z)), dtype=float)
    else:
        profile = profile / n_active_axes

    components = {}
    for amp, axis in zip(amplitudes, "xyz"):
        data = (amp * (1 + 0.5j) * profile).reshape(shape)
        components[f"{field_prefix}{axis}"] = td.ScalarFieldDataArray(data, coords=coords)
    return td.FieldDataset(**components)


def _scaled_amplitudes(scale: float) -> tuple[float, float, float]:
    return tuple(scale * value for value in BASE_PARAM_AMPLITUDES)


def _to_static_float_tuple(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(get_static(value)) for value in values)


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
class SweepConfig:
    dataset_spacing: float = BASE_DATASET_SPACING
    min_steps_per_wvl: int = BASE_MIN_STEPS_PER_WVL
    wvl0: float = BASE_WVL0
    source_size: tuple[float, float, float] = BASE_SOURCE_SIZE
    sim_dims: int = 3
    objective_3d: str = OBJECTIVE_3D_MODE
    monitor_freq_scale: float = 1.0
    monitor_freq_scales: tuple[float, ...] | None = None
    amplitude_scale: float = 1.0
    background_permittivity: float = 1.0
    source_structure_permittivity: float | None = None
    source_structure_custom_medium: bool = False
    add_far_corner_structure: bool = False


@dataclass(frozen=True)
class SourceCase:
    name: str
    monitor_components: tuple[str, str, str]
    source_kind: str
    field_prefix: str
    source_size_mask: tuple[bool, bool, bool]
    dataset_delta_fd: float = 1e-4


@dataclass(frozen=True)
class GradientMetrics:
    grad_adjoint: np.ndarray
    grad_fd: np.ndarray
    angle_deg: float
    adjoint_norm: float
    fd_norm: float


def _resolve_derivative_target(case: SourceCase, derivative_target: str) -> str:
    if derivative_target == "dataset":
        return "field_dataset" if case.source_kind == "field" else "current_dataset"
    return derivative_target


SOURCE_CASES = (
    SourceCase(
        name="custom_field_vec_e",
        monitor_components=("Ex", "Ey", "Ez"),
        source_kind="field",
        field_prefix="E",
        source_size_mask=(True, True, False),
    ),
    SourceCase(
        name="custom_field_vec_h",
        monitor_components=("Hx", "Hy", "Hz"),
        source_kind="field",
        field_prefix="H",
        source_size_mask=(True, True, False),
    ),
    SourceCase(
        name="custom_current_vec_e",
        monitor_components=("Ex", "Ey", "Ez"),
        source_kind="current",
        field_prefix="E",
        source_size_mask=(True, True, True),
    ),
    SourceCase(
        name="custom_current_vec_h",
        monitor_components=("Hx", "Hy", "Hz"),
        source_kind="current",
        field_prefix="H",
        source_size_mask=(True, True, True),
    ),
)


def _source_size_for_case(
    case: SourceCase, source_size: tuple[float, float, float], sim_dims: int
) -> tuple[float, float, float]:
    masked_size = tuple(
        source_size[axis] if case.source_size_mask[axis] else 0.0 for axis in range(3)
    )
    if sim_dims == 2:
        if case.source_kind == "field":
            # Keep CustomFieldSource planar in 2D: collapse y and use the second
            # in-plane extent on z.
            return (masked_size[0], 0.0, masked_size[1])
        return (masked_size[0], 0.0, masked_size[2])
    return masked_size


def _collapse_y_size(size: tuple[float, float, float], sim_dims: int) -> tuple[float, float, float]:
    if sim_dims == 2:
        return (size[0], 0.0, size[2])
    return size


def _collapse_y_center(
    center: tuple[float, float, float], sim_dims: int
) -> tuple[float, float, float]:
    if sim_dims == 2:
        return (center[0], 0.0, center[2])
    return center


def _make_source(
    case: SourceCase,
    amplitudes: tuple[float, float, float],
    config: SweepConfig,
    freq0: float,
    pulse: td.GaussianPulse,
    source_size_override: tuple[float, float, float] | None = None,
    source_center_override: tuple[float, float, float] | None = None,
    dataset_size_override: tuple[float, float, float] | None = None,
) -> td.Source:
    source_size = _source_size_for_case(
        case,
        source_size_override if source_size_override is not None else config.source_size,
        config.sim_dims,
    )
    dataset_size = _source_size_for_case(
        case,
        dataset_size_override if dataset_size_override is not None else config.source_size,
        config.sim_dims,
    )
    source_center = _collapse_y_center(
        source_center_override if source_center_override is not None else BASE_SOURCE_CENTER,
        config.sim_dims,
    )

    if case.source_kind == "field":
        coords = _make_coords(dataset_size, config.dataset_spacing, freq0)
        field_dataset = _make_field_dataset(case.field_prefix, amplitudes, coords)
        return td.CustomFieldSource(
            center=source_center,
            size=source_size,
            source_time=pulse,
            field_dataset=field_dataset,
        )

    if case.source_kind == "current":
        coords = _make_coords(dataset_size, config.dataset_spacing, freq0)
        current_dataset = _make_field_dataset(case.field_prefix, amplitudes, coords)
        return td.CustomCurrentSource(
            center=source_center,
            size=source_size,
            source_time=pulse,
            current_dataset=current_dataset,
        )

    raise ValueError(f"Unsupported source_kind: {case.source_kind}")


def _source_host_structure(
    source_center: tuple[float, float, float],
    source_size: tuple[float, float, float],
    source_structure_permittivity: float,
    sim_dims: int,
) -> td.Structure:
    """Create the enclosing structure ("host") that contains the source."""
    source_size = _to_static_float_tuple(source_size)
    host_size = _collapse_y_size(
        (
            source_size[0] + 0.2,
            source_size[1] + 0.2,
            max(source_size[2] + 0.2, 0.3),
        ),
        sim_dims,
    )
    host_center = _collapse_y_center(_to_static_float_tuple(source_center), sim_dims)
    return td.Structure(
        geometry=td.Box(center=host_center, size=host_size),
        medium=td.Medium(permittivity=source_structure_permittivity),
    )


def _source_host_custom_medium(
    source_center: tuple[float, float, float],
    source_size: tuple[float, float, float],
    sim_dims: int,
) -> td.Structure:
    """Create a nonuniform custom-medium host structure that encloses the source."""
    source_size = _to_static_float_tuple(source_size)
    host_size = _collapse_y_size(
        (
            source_size[0] + 0.2,
            source_size[1] + 0.2,
            max(source_size[2] + 0.2, 0.3),
        ),
        sim_dims,
    )
    host_center = _collapse_y_center(_to_static_float_tuple(source_center), sim_dims)
    host_bounds_min = [c - 0.5 * s for c, s in zip(host_center, host_size)]
    host_bounds_max = [c + 0.5 * s for c, s in zip(host_center, host_size)]

    x = np.linspace(host_bounds_min[0], host_bounds_max[0], 9)
    if sim_dims == 2:
        y = np.array([host_center[1]], dtype=float)
    else:
        y = np.linspace(host_bounds_min[1], host_bounds_max[1], 9)
    z = np.linspace(host_bounds_min[2], host_bounds_max[2], 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    eps = 2.5 + 0.2 * (X - host_center[0]) + 0.15 * (Y - host_center[1]) ** 2
    eps += 0.05 * (Z - host_center[2])

    return td.Structure(
        geometry=td.Box(center=host_center, size=host_size),
        medium=td.CustomMedium(
            permittivity=td.SpatialDataArray(
                eps,
                coords={"x": x, "y": y, "z": z},
            )
        ),
    )


def _far_corner_structure(wvl0: float, sim_dims: int) -> td.Structure:
    """Create a tiny high-index block near a far corner of the simulation domain."""
    center_val = FAR_CORNER_BLOCK_OFFSET_FACTOR * wvl0
    center = _collapse_y_center((center_val, center_val, center_val), sim_dims)
    size = _collapse_y_size(FAR_CORNER_BLOCK_SIZE, sim_dims)
    return td.Structure(
        geometry=td.Box(
            center=center,
            size=size,
        ),
        medium=td.Medium(permittivity=FAR_CORNER_BLOCK_PERMITTIVITY),
    )


def _make_sim(
    source: td.Source,
    config: SweepConfig,
) -> td.Simulation:
    freq0 = td.C_0 / config.wvl0
    if config.monitor_freq_scales is None:
        monitor_freqs = [config.monitor_freq_scale * freq0]
    else:
        monitor_freqs = [scale * freq0 for scale in config.monitor_freq_scales]
    monitor_center = _collapse_y_center(MONITOR_CENTER, config.sim_dims)
    monitor_size = _collapse_y_size(MONITOR_SIZE, config.sim_dims)
    monitor = td.FieldMonitor(
        name=FLUX_MONITOR_NAME,
        center=monitor_center,
        size=monitor_size,
        freqs=monitor_freqs,
    )
    structures = []
    if config.source_structure_permittivity is not None and config.source_structure_custom_medium:
        raise ValueError(
            "Only one source host type is allowed: set either "
            "'source_structure_permittivity' or 'source_structure_custom_medium'."
        )

    host_center = _collapse_y_center(_to_static_float_tuple(BASE_SOURCE_CENTER), config.sim_dims)
    host_size = _collapse_y_size(_to_static_float_tuple(config.source_size), config.sim_dims)

    if config.source_structure_permittivity is not None:
        structures.append(
            _source_host_structure(
                source_center=host_center,
                source_size=host_size,
                source_structure_permittivity=config.source_structure_permittivity,
                sim_dims=config.sim_dims,
            )
        )
    if config.source_structure_custom_medium:
        structures.append(
            _source_host_custom_medium(
                source_center=host_center,
                source_size=host_size,
                sim_dims=config.sim_dims,
            )
        )
    if config.add_far_corner_structure:
        structures.append(_far_corner_structure(config.wvl0, config.sim_dims))

    sim_size = _collapse_y_size(
        (3 * config.wvl0, 3 * config.wvl0, 3 * config.wvl0), config.sim_dims
    )
    return td.Simulation(
        size=sim_size,
        run_time=SIM_RUN_TIME,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=config.min_steps_per_wvl,
            wavelength=config.wvl0,
        ),
        medium=td.Medium(permittivity=config.background_permittivity),
        structures=structures,
        sources=[source],
        monitors=[monitor],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )


def _eval_objective(
    sim_data: td.SimulationData, sim_dims: int, objective_3d: str, freq0: float
) -> float:
    field_data = sim_data.load_field_monitor(FLUX_MONITOR_NAME)
    if sim_dims == 2:
        return np.sum(field_data.intensity.values)
    if objective_3d == "flux":
        return field_data.flux.values
    if objective_3d == "flux_difference":
        flux = np.asarray(field_data.flux.values).reshape(-1)
        freqs = np.asarray(field_data.flux.coords["f"].data, dtype=float).reshape(-1)
        if freqs.size < 2:
            raise ValueError(
                "flux_difference objective requires at least two monitor frequencies, "
                f"got {freqs.size}."
            )
        idx_center = int(np.argmin(np.abs(freqs - freq0)))
        idx_off = int(np.argmin(np.abs(freqs - (OFF_CENTER_MONITOR_FREQ_SCALE * freq0))))
        if idx_center == idx_off:
            raise ValueError(
                "flux_difference objective requires distinct bins for f0 and off-center frequency."
            )
        return flux[idx_center] - flux[idx_off]
    if objective_3d == "intensity":
        return np.sum(field_data.intensity.values)
    raise ValueError(f"Unsupported 3D objective mode: {objective_3d!r}")


def _active_axes_for_derivative(
    case: SourceCase, config: SweepConfig, derivative_target: str
) -> tuple[int, ...]:
    if derivative_target in ("field_dataset", "current_dataset"):
        return (0, 1, 2)

    if derivative_target == "center":
        effective_size = _source_size_for_case(case, config.source_size, config.sim_dims)
        active_mask = [
            effective_size[axis] > _finite_difference_delta(case, config, derivative_target)
            for axis in range(3)
        ]
        if config.sim_dims == 2:
            active_mask[1] = False
        return tuple(axis for axis, is_active in enumerate(active_mask) if is_active)

    raise ValueError(f"Unsupported derivative target: {derivative_target!r}")


def _finite_difference_delta(
    case: SourceCase, config: SweepConfig, derivative_target: str
) -> float:
    if derivative_target in ("field_dataset", "current_dataset"):
        return case.dataset_delta_fd
    if derivative_target == "center":
        return config.wvl0 / config.min_steps_per_wvl
    raise ValueError(f"Unsupported derivative target: {derivative_target!r}")


def _fd_agreement_tolerances(derivative_target: str) -> tuple[float, float, float]:
    if derivative_target in ("field_dataset", "current_dataset"):
        return DATASET_ANGLE_LIMIT_DEG, DATASET_NORM_RTOL, DATASET_NORM_ATOL
    if derivative_target == "center":
        return CENTER_ANGLE_LIMIT_DEG, CENTER_NORM_RTOL, CENTER_NORM_ATOL
    raise ValueError(f"Unsupported derivative target for agreement check: {derivative_target!r}")


def _run_gradient_case(
    tmp_path,
    case: SourceCase,
    config: SweepConfig,
    derivative_target: str,
    *,
    label: str,
) -> GradientMetrics:
    resolved_derivative_target = _resolve_derivative_target(case, derivative_target)
    freq0 = td.C_0 / config.wvl0
    pulse = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    amplitudes = _scaled_amplitudes(config.amplitude_scale)
    if resolved_derivative_target in ("field_dataset", "current_dataset"):
        params = amplitudes
    elif resolved_derivative_target == "center":
        params = BASE_SOURCE_CENTER
    else:
        raise ValueError(f"Unsupported derivative target: {derivative_target!r}")

    active_axes = _active_axes_for_derivative(case, config, resolved_derivative_target)
    if not active_axes:
        pytest.skip(
            f"No active axes for derivative target {derivative_target!r} in "
            f"{case.name} ({config.sim_dims}D)."
        )
    delta = _finite_difference_delta(case, config, resolved_derivative_target)

    def make_source(derivative_params: tuple[float, float, float]) -> td.Source:
        if resolved_derivative_target in ("field_dataset", "current_dataset"):
            return _make_source(case, derivative_params, config, freq0, pulse)
        if resolved_derivative_target == "center":
            return _make_source(
                case,
                amplitudes,
                config,
                freq0,
                pulse,
                source_center_override=derivative_params,
                dataset_size_override=config.source_size,
            )
        raise ValueError(f"Unsupported derivative target: {resolved_derivative_target!r}")

    def objective_adj(p0: float, p1: float, p2: float) -> float:
        sim = _make_sim(make_source((p0, p1, p2)), config)
        sim_data = web.run(
            sim,
            task_name=f"{label}_adj",
            path=tmp_path / f"{label}_adj.hdf5",
            local_gradient=True,
            verbose=False,
        )
        return _eval_objective(sim_data, config.sim_dims, config.objective_3d, freq0)

    grad_adjoint = np.zeros(3, dtype=float)
    for axis in active_axes:
        grad_adjoint[axis] = float(ag.grad(objective_adj, axis)(*params))

    sims = {}
    for idx in active_axes:
        axis = "xyz"[idx]
        params_plus = list(params)
        params_plus[idx] += delta
        params_minus = list(params)
        params_minus[idx] -= delta
        sims[f"{label}_fd_{axis}_plus"] = _make_sim(make_source(tuple(params_plus)), config)
        sims[f"{label}_fd_{axis}_minus"] = _make_sim(make_source(tuple(params_minus)), config)

    sim_data_map = web.run_async(
        sims,
        path_dir=tmp_path,
        local_gradient=False,
        verbose=False,
    )

    grad_fd = np.zeros(3, dtype=float)
    for idx in active_axes:
        axis = "xyz"[idx]
        obj_plus = float(
            np.asarray(
                _eval_objective(
                    sim_data_map[f"{label}_fd_{axis}_plus"],
                    config.sim_dims,
                    config.objective_3d,
                    freq0,
                )
            ).squeeze()
        )
        obj_minus = float(
            np.asarray(
                _eval_objective(
                    sim_data_map[f"{label}_fd_{axis}_minus"],
                    config.sim_dims,
                    config.objective_3d,
                    freq0,
                )
            ).squeeze()
        )
        grad_fd[idx] = (obj_plus - obj_minus) / (2 * delta)

    angle_deg = angled_overlap_deg(grad_adjoint, grad_fd)
    adjoint_norm = float(np.linalg.norm(grad_adjoint))
    fd_norm = float(np.linalg.norm(grad_fd))

    print(f"[{label}] grad_adjoint = {grad_adjoint}", file=sys.stderr)
    print(f"[{label}] grad_fd      = {grad_fd}", file=sys.stderr)
    print(f"[{label}] angle_deg    = {angle_deg}", file=sys.stderr)
    print(f"[{label}] adjoint_norm = {adjoint_norm}", file=sys.stderr)
    print(f"[{label}] fd_norm      = {fd_norm}", file=sys.stderr)

    return GradientMetrics(
        grad_adjoint=grad_adjoint,
        grad_fd=grad_fd,
        angle_deg=angle_deg,
        adjoint_norm=adjoint_norm,
        fd_norm=fd_norm,
    )


def _run_adjoint_only_gradient_case(
    tmp_path,
    case: SourceCase,
    config: SweepConfig,
    *,
    label: str,
    params: tuple[complex, complex, complex],
) -> np.ndarray:
    """Compute source-parameter adjoint gradients without finite differences."""
    freq0 = td.C_0 / config.wvl0
    pulse = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)

    def objective_adj(ax: complex, ay: complex, az: complex):
        source = _make_source(case, (ax, ay, az), config, freq0, pulse)
        sim = _make_sim(source, config)
        sim_data = web.run(
            sim,
            task_name=f"{label}_adj_only",
            path=tmp_path / f"{label}_adj_only.hdf5",
            local_gradient=True,
            verbose=False,
        )
        return _eval_objective(sim_data, config.sim_dims, config.objective_3d, freq0)

    return np.array(
        [
            ag.grad(objective_adj, 0)(*params),
            ag.grad(objective_adj, 1)(*params),
            ag.grad(objective_adj, 2)(*params),
        ],
        dtype=complex,
    )


def _assert_fd_agreement(
    metrics: GradientMetrics, case: SourceCase, derivative_target: str, *, label: str
) -> None:
    resolved_derivative_target = _resolve_derivative_target(case, derivative_target)
    angle_limit_deg, norm_rtol, norm_atol = _fd_agreement_tolerances(resolved_derivative_target)
    context = (
        f"{label}: angle_deg={metrics.angle_deg:.6g}, "
        f"grad_adjoint={metrics.grad_adjoint}, grad_fd={metrics.grad_fd}, "
        f"adjoint_norm={metrics.adjoint_norm:.6g}, fd_norm={metrics.fd_norm:.6g}"
    )
    assert metrics.angle_deg < angle_limit_deg, context
    assert np.isfinite(metrics.adjoint_norm), context
    assert np.isfinite(metrics.fd_norm), context
    np.testing.assert_allclose(
        metrics.adjoint_norm,
        metrics.fd_norm,
        rtol=norm_rtol,
        atol=norm_atol,
        err_msg=context,
    )


def _run_variation_sweep(
    tmp_path,
    case: SourceCase,
    sim_dims: int,
    derivative_target: str,
    variation_name: str,
    values: tuple,
    update_config: Callable[[SweepConfig, object], SweepConfig],
) -> None:
    base_config = replace(SweepConfig(), sim_dims=sim_dims)
    resolved_target = _resolve_derivative_target(case, derivative_target)
    for value in values:
        config = update_config(base_config, value)
        label = f"{case.name}_{resolved_target}_{variation_name}_{value}_{sim_dims}d"
        metrics = _run_gradient_case(tmp_path, case, config, derivative_target, label=label)
        _assert_fd_agreement(metrics, case, derivative_target, label=label)


def _skip_2d_field_cases(sim_dims: int, case: SourceCase) -> None:
    if sim_dims == 2 and case.source_kind == "field":
        pytest.skip("2D variation sweeps are only run for CustomCurrentSource cases.")


@pytest.mark.numerical
@pytest.mark.parametrize("sim_dims", SIM_DIMS_VALUES, ids=lambda dims: f"{dims}d")
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_dataset_spacing(
    _enable_local_cache, tmp_path, case, derivative_target, sim_dims
):
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        tmp_path,
        case,
        sim_dims,
        derivative_target,
        "dataset_spacing",
        DATASET_SPACING_VALUES,
        lambda base, value: replace(base, dataset_spacing=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("sim_dims", SIM_DIMS_VALUES, ids=lambda dims: f"{dims}d")
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_grid_resolution(
    _enable_local_cache, tmp_path, case, derivative_target, sim_dims
):
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        tmp_path,
        case,
        sim_dims,
        derivative_target,
        "min_steps_per_wvl",
        MIN_STEPS_PER_WVL_VALUES,
        lambda base, value: replace(base, min_steps_per_wvl=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("sim_dims", SIM_DIMS_VALUES, ids=lambda dims: f"{dims}d")
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_source_size(
    _enable_local_cache, tmp_path, case, derivative_target, sim_dims
):
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        tmp_path,
        case,
        sim_dims,
        derivative_target,
        "source_size_xy",
        SOURCE_SIZE_XY_VALUES,
        lambda base, value: replace(base, source_size=(value, value, 0.0)),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("sim_dims", SIM_DIMS_VALUES, ids=lambda dims: f"{dims}d")
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_amplitude(
    _enable_local_cache, tmp_path, case, derivative_target, sim_dims
):
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        tmp_path,
        case,
        sim_dims,
        derivative_target,
        "amplitude_scale",
        AMPLITUDE_SCALE_VALUES,
        lambda base, value: replace(base, amplitude_scale=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("sim_dims", SIM_DIMS_VALUES, ids=lambda dims: f"{dims}d")
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_permittivity(
    _enable_local_cache, tmp_path, case, derivative_target, sim_dims
):
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        tmp_path,
        case,
        sim_dims,
        derivative_target,
        "permittivity",
        PERMITTIVITY_VALUES,
        lambda base, value: replace(
            base,
            background_permittivity=value[0],
            source_structure_permittivity=value[1],
        ),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("sim_dims", SIM_DIMS_VALUES, ids=lambda dims: f"{dims}d")
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_wavelength(
    _enable_local_cache, tmp_path, case, derivative_target, sim_dims
):
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        tmp_path,
        case,
        sim_dims,
        derivative_target,
        "wvl0",
        WVL0_VALUES,
        lambda base, value: replace(base, wvl0=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case_name", ("custom_field_vec_e", "custom_current_vec_e"))
def test_custom_source_gradient_off_center_monitor_frequency(
    _enable_local_cache, tmp_path, case_name, derivative_target
):
    """Check source gradients when objective frequency is offset from source center frequency."""
    case = next(candidate for candidate in SOURCE_CASES if candidate.name == case_name)
    resolved_target = _resolve_derivative_target(case, derivative_target)
    config = replace(
        SweepConfig(sim_dims=3),
        monitor_freq_scale=OFF_CENTER_MONITOR_FREQ_SCALE,
    )
    label = f"{case.name}_{resolved_target}_monitor_freq_scale_{OFF_CENTER_MONITOR_FREQ_SCALE}_3d"
    metrics = _run_gradient_case(
        tmp_path,
        case,
        config,
        derivative_target,
        label=label,
    )
    _assert_fd_agreement(metrics, case, derivative_target, label=label)


@pytest.mark.numerical
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case_name", ("custom_field_vec_e", "custom_current_vec_e"))
def test_custom_source_gradient_two_frequency_flux_difference(
    _enable_local_cache, tmp_path, case_name, derivative_target
):
    """Check source gradients with objective flux(f0) - flux(0.95*f0) in 3D."""
    case = next(candidate for candidate in SOURCE_CASES if candidate.name == case_name)
    resolved_target = _resolve_derivative_target(case, derivative_target)
    config = replace(
        SweepConfig(sim_dims=3),
        objective_3d="flux_difference",
        monitor_freq_scales=TWO_FREQ_MONITOR_SCALES,
    )
    label = f"{case.name}_{resolved_target}_flux_difference_3d"
    metrics = _run_gradient_case(
        tmp_path,
        case,
        config,
        derivative_target,
        label=label,
    )
    _assert_fd_agreement(metrics, case, derivative_target, label=label)


@pytest.mark.numerical
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_inside_structure(
    _enable_local_cache, tmp_path, case, derivative_target
):
    resolved_target = _resolve_derivative_target(case, derivative_target)
    structure_config = replace(SweepConfig(), source_structure_permittivity=4.0)
    structure_metrics = _run_gradient_case(
        tmp_path,
        case,
        structure_config,
        derivative_target,
        label=f"{case.name}_{resolved_target}_source_in_structure_eps_4",
    )
    _assert_fd_agreement(
        structure_metrics,
        case,
        derivative_target,
        label=f"{case.name}_{resolved_target}_source_in_structure_eps_4",
    )


@pytest.mark.numerical
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
def test_custom_current_source_gradient_cube_size(_enable_local_cache, tmp_path, derivative_target):
    current_case = next(case for case in SOURCE_CASES if case.name == "custom_current_vec_e")
    resolved_target = _resolve_derivative_target(current_case, derivative_target)
    cube_config = replace(SweepConfig(), source_size=(0.5, 0.5, 0.5))
    cube_metrics = _run_gradient_case(
        tmp_path,
        current_case,
        cube_config,
        derivative_target,
        label=f"{current_case.name}_{resolved_target}_source_size_xyz_0.5",
    )
    _assert_fd_agreement(
        cube_metrics,
        current_case,
        derivative_target,
        label=f"{current_case.name}_{resolved_target}_source_size_xyz_0.5",
    )


@pytest.mark.numerical
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_in_nonuniform_custom_medium(
    _enable_local_cache, tmp_path, case, derivative_target
):
    resolved_target = _resolve_derivative_target(case, derivative_target)
    custom_medium_config = replace(SweepConfig(), source_structure_custom_medium=True)
    custom_medium_metrics = _run_gradient_case(
        tmp_path,
        case,
        custom_medium_config,
        derivative_target,
        label=f"{case.name}_{resolved_target}_source_in_custom_medium",
    )
    _assert_fd_agreement(
        custom_medium_metrics,
        case,
        derivative_target,
        label=f"{case.name}_{resolved_target}_source_in_custom_medium",
    )


@pytest.mark.numerical
@pytest.mark.parametrize("derivative_target", DERIVATIVE_TARGETS)
def test_custom_source_gradient_stable_with_remote_dt_constraint(
    _enable_local_cache, tmp_path, derivative_target
):
    """Source gradients should stay stable when unrelated remote cells constrain global dt."""
    case = next(candidate for candidate in SOURCE_CASES if candidate.name == "custom_current_vec_h")
    resolved_target = _resolve_derivative_target(case, derivative_target)
    base_config = replace(SweepConfig(), source_structure_permittivity=4.0)
    constrained_config = replace(base_config, add_far_corner_structure=True)

    base_metrics = _run_gradient_case(
        tmp_path,
        case,
        base_config,
        derivative_target,
        label=f"{case.name}_{resolved_target}_inside_structure_base",
    )
    constrained_metrics = _run_gradient_case(
        tmp_path,
        case,
        constrained_config,
        derivative_target,
        label=f"{case.name}_{resolved_target}_inside_structure_remote_dt",
    )

    _assert_fd_agreement(
        base_metrics,
        case,
        derivative_target,
        label=f"{case.name}_{resolved_target}_inside_structure_base",
    )
    _assert_fd_agreement(
        constrained_metrics,
        case,
        derivative_target,
        label=f"{case.name}_{resolved_target}_inside_structure_remote_dt",
    )
    resolved_derivative_target = _resolve_derivative_target(case, derivative_target)
    _, norm_rtol, norm_atol = _fd_agreement_tolerances(resolved_derivative_target)

    np.testing.assert_allclose(
        constrained_metrics.fd_norm,
        base_metrics.fd_norm,
        rtol=norm_rtol,
        atol=norm_atol,
    )
    np.testing.assert_allclose(
        constrained_metrics.adjoint_norm,
        base_metrics.adjoint_norm,
        rtol=norm_rtol,
        atol=norm_atol,
    )


@pytest.mark.numerical
@pytest.mark.parametrize("case_name", ("custom_field_vec_e", "custom_current_vec_e"))
def test_custom_source_intensity_gradient_global_phase_equivariance(
    _enable_local_cache, tmp_path, case_name, redirect_stdout_to_stderr
):
    """Intensity objective gradients should rotate with a global ``1j`` source phase."""
    case = next(candidate for candidate in SOURCE_CASES if candidate.name == case_name)
    config = replace(SweepConfig(sim_dims=3), objective_3d="intensity")

    base_params = tuple(np.asarray(BASE_PARAM_AMPLITUDES, dtype=complex))
    phased_params = tuple(1j * np.asarray(BASE_PARAM_AMPLITUDES, dtype=complex))

    grad_base = _run_adjoint_only_gradient_case(
        tmp_path,
        case,
        config,
        label=f"{case.name}_intensity_phase_base",
        params=base_params,
    )
    grad_phased = _run_adjoint_only_gradient_case(
        tmp_path,
        case,
        config,
        label=f"{case.name}_intensity_phase_j",
        params=phased_params,
    )

    assert not np.allclose(grad_base.real, 0.0, rtol=0.0, atol=1e-12)
    assert not np.allclose(grad_base.imag, 0.0, rtol=0.0, atol=1e-12)
    # This test differentiates a real-valued intensity objective through the full adjoint chain,
    # not just the direct source VJP map. Under this convention, a global ``1j`` phase on the
    # traced source parameters rotates the gradient by ``-1j``.
    np.testing.assert_allclose(grad_phased, -1j * grad_base, rtol=1e-2, atol=1e-6)
