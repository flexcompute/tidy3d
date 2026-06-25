from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace

import autograd as ag
import numpy as np
import pytest
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static

from .numerical_test_helpers import (
    EvaluationData,
    case_identity_id,
    condition_metric,
    evaluate_allclose_agreement,
    finalize_result,
    gradient_angle_deg,
    load_or_collect_evaluation_data,
)
from .result_models import Metric


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


def _dataset_spacing_sweep_config(base_config: SweepConfig, value: float) -> SweepConfig:
    return replace(base_config, dataset_spacing=value)


def _grid_resolution_sweep_config(base_config: SweepConfig, value: int) -> SweepConfig:
    return replace(base_config, min_steps_per_wvl=value)


def _source_size_sweep_config(base_config: SweepConfig, value: float) -> SweepConfig:
    return replace(base_config, source_size=(value, value, 0.0))


def _amplitude_sweep_config(base_config: SweepConfig, value: float) -> SweepConfig:
    return replace(base_config, amplitude_scale=value)


def _permittivity_sweep_config(
    base_config: SweepConfig, value: tuple[float, float | None]
) -> SweepConfig:
    return replace(
        base_config,
        background_permittivity=value[0],
        source_structure_permittivity=value[1],
    )


def _wavelength_sweep_config(base_config: SweepConfig, value: float) -> SweepConfig:
    return replace(base_config, wvl0=value)


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


@dataclass(frozen=True)
class SourceGradientTestParameters:
    case_name: str
    derivative_target: str
    sim_dims: int = 3


class SourceGradientCaseIdentity(BaseModel):
    """Semantic identity for one custom-source gradient artifact."""

    test_name: str
    case_name: str
    derivative_target: str
    resolved_derivative_target: str
    dataset_spacing: float
    min_steps_per_wvl: int
    wvl0: float
    source_size: tuple[float, float, float]
    sim_dims: int
    objective_3d: str
    monitor_freq_scale: float
    monitor_freq_scales: tuple[float, ...] | None
    amplitude_scale: float
    background_permittivity: float
    source_structure_permittivity: float | None
    source_structure_custom_medium: bool
    add_far_corner_structure: bool
    variation_name: str | None = None
    variation_values: tuple[str, ...] = ()


class SourceVariationSweepCaseIdentity(SourceGradientCaseIdentity):
    """Semantic identity for a custom-source gradient variation sweep artifact."""

    realized_sweep_config_hash: str


class SourcePhaseCaseIdentity(BaseModel):
    """Semantic identity for one custom-source global-phase equivariance artifact."""

    test_name: str
    case_name: str
    dataset_spacing: float
    min_steps_per_wvl: int
    wvl0: float
    source_size: tuple[float, float, float]
    sim_dims: int
    objective_3d: str
    monitor_freq_scale: float
    monitor_freq_scales: tuple[float, ...] | None
    amplitude_scale: float
    background_permittivity: float
    source_structure_permittivity: float | None
    source_structure_custom_medium: bool
    add_far_corner_structure: bool


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


def _source_case_from_name(case_name: str) -> SourceCase:
    return next(case for case in SOURCE_CASES if case.name == case_name)


def _source_gradient_parameters(
    *,
    case_names: tuple[str, ...] | None = None,
    sim_dims_values: tuple[int, ...] = (3,),
    derivative_targets: tuple[str, ...] = DERIVATIVE_TARGETS,
) -> tuple[SourceGradientTestParameters, ...]:
    selected_case_names = case_names or tuple(case.name for case in SOURCE_CASES)
    return tuple(
        SourceGradientTestParameters(
            case_name=case_name,
            derivative_target=derivative_target,
            sim_dims=sim_dims,
        )
        for sim_dims in sim_dims_values
        for derivative_target in derivative_targets
        for case_name in selected_case_names
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


def _sweep_config_identity_payload(config: SweepConfig) -> dict[str, object]:
    return {
        "dataset_spacing": float(config.dataset_spacing),
        "min_steps_per_wvl": int(config.min_steps_per_wvl),
        "wvl0": float(config.wvl0),
        "source_size": tuple(float(value) for value in config.source_size),
        "sim_dims": int(config.sim_dims),
        "objective_3d": config.objective_3d,
        "monitor_freq_scale": float(config.monitor_freq_scale),
        "monitor_freq_scales": (
            None
            if config.monitor_freq_scales is None
            else tuple(float(value) for value in config.monitor_freq_scales)
        ),
        "amplitude_scale": float(config.amplitude_scale),
        "background_permittivity": float(config.background_permittivity),
        "source_structure_permittivity": (
            None
            if config.source_structure_permittivity is None
            else float(config.source_structure_permittivity)
        ),
        "source_structure_custom_medium": bool(config.source_structure_custom_medium),
        "add_far_corner_structure": bool(config.add_far_corner_structure),
    }


def _source_gradient_case_identity(
    *,
    test_name: str,
    case: SourceCase,
    derivative_target: str,
    config: SweepConfig,
    variation_name: str | None = None,
    variation_values: tuple = (),
) -> SourceGradientCaseIdentity:
    return SourceGradientCaseIdentity(
        test_name=test_name,
        case_name=case.name,
        derivative_target=derivative_target,
        resolved_derivative_target=_resolve_derivative_target(case, derivative_target),
        **_sweep_config_identity_payload(config),
        variation_name=variation_name,
        variation_values=tuple(str(value) for value in variation_values),
    )


def _phase_equivariance_config() -> SweepConfig:
    return replace(SweepConfig(sim_dims=3), objective_3d="intensity")


def _source_phase_case_identity(
    *,
    test_name: str,
    case: SourceCase,
    config: SweepConfig,
) -> SourcePhaseCaseIdentity:
    return SourcePhaseCaseIdentity(
        test_name=test_name,
        case_name=case.name,
        **_sweep_config_identity_payload(config),
    )


def _source_phase_case_id(case_name: str) -> str:
    return case_identity_id(
        _source_phase_case_identity(
            test_name="custom_source_intensity_gradient_global_phase_equivariance",
            case=_source_case_from_name(case_name),
            config=_phase_equivariance_config(),
        ),
        prefix="source-phase",
    )


def _realized_sweep_configs(
    base_config: SweepConfig,
    values: tuple,
    update_config: Callable[[SweepConfig, object], SweepConfig],
) -> tuple[SweepConfig, ...]:
    return tuple(update_config(base_config, value) for value in values)


def _realized_sweep_config_hash(configs: tuple[SweepConfig, ...]) -> str:
    config_json = json.dumps(
        [_sweep_config_identity_payload(config) for config in configs],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _source_variation_sweep_case_identity(
    *,
    test_name: str,
    case: SourceCase,
    derivative_target: str,
    base_config: SweepConfig,
    variation_name: str,
    variation_values: tuple,
    realized_configs: tuple[SweepConfig, ...],
) -> SourceVariationSweepCaseIdentity:
    base_identity = _source_gradient_case_identity(
        test_name=test_name,
        case=case,
        derivative_target=derivative_target,
        config=base_config,
        variation_name=variation_name,
        variation_values=variation_values,
    )
    return SourceVariationSweepCaseIdentity(
        **base_identity.model_dump(),
        realized_sweep_config_hash=_realized_sweep_config_hash(realized_configs),
    )


def _source_gradient_case_id(
    *,
    test_name: str,
    params: SourceGradientTestParameters,
    config: SweepConfig,
    variation_name: str | None = None,
    variation_values: tuple = (),
) -> str:
    case = _source_case_from_name(params.case_name)
    return case_identity_id(
        _source_gradient_case_identity(
            test_name=test_name,
            case=case,
            derivative_target=params.derivative_target,
            config=config,
            variation_name=variation_name,
            variation_values=variation_values,
        ),
        prefix="source",
    )


def _source_variation_sweep_case_id(
    *,
    test_name: str,
    params: SourceGradientTestParameters,
    base_config: SweepConfig,
    variation_name: str,
    variation_values: tuple,
    update_config: Callable[[SweepConfig, object], SweepConfig],
) -> str:
    case = _source_case_from_name(params.case_name)
    realized_configs = _realized_sweep_configs(base_config, variation_values, update_config)
    return case_identity_id(
        _source_variation_sweep_case_identity(
            test_name=test_name,
            case=case,
            derivative_target=params.derivative_target,
            base_config=base_config,
            variation_name=variation_name,
            variation_values=variation_values,
            realized_configs=realized_configs,
        ),
        prefix="source",
    )


def _gradient_metrics_to_evaluation_data(metrics: GradientMetrics) -> EvaluationData:
    return {
        "grad_adjoint": np.asarray(metrics.grad_adjoint, dtype=float),
        "grad_fd": np.asarray(metrics.grad_fd, dtype=float),
    }


def _gradient_metrics_from_gradients(
    grad_adjoint: np.ndarray,
    grad_fd: np.ndarray,
) -> GradientMetrics:
    grad_adjoint = np.asarray(grad_adjoint, dtype=float)
    grad_fd = np.asarray(grad_fd, dtype=float)
    return GradientMetrics(
        grad_adjoint=grad_adjoint,
        grad_fd=grad_fd,
        angle_deg=gradient_angle_deg(grad_adjoint, grad_fd),
        adjoint_norm=float(np.linalg.norm(grad_adjoint)),
        fd_norm=float(np.linalg.norm(grad_fd)),
    )


def _gradient_metrics_from_evaluation_data(
    evaluation_data: EvaluationData,
    *,
    index: int | None = None,
) -> GradientMetrics:
    def value(name: str) -> np.ndarray:
        data = np.asarray(evaluation_data[name], dtype=float)
        return data if index is None else data[index]

    grad_adjoint = value("grad_adjoint")
    grad_fd = value("grad_fd")
    return _gradient_metrics_from_gradients(grad_adjoint, grad_fd)


def _evaluate_source_gradient_metrics(
    metrics: GradientMetrics,
    case: SourceCase,
    derivative_target: str,
    *,
    metric_prefix: str,
) -> tuple[list[Metric], list[Metric]]:
    resolved_derivative_target = _resolve_derivative_target(case, derivative_target)
    angle_limit_deg, norm_rtol, norm_atol = _fd_agreement_tolerances(resolved_derivative_target)
    regression_metrics = [
        Metric(
            name=f"{metric_prefix}_angle_deg",
            observed=metrics.angle_deg,
            expected=angle_limit_deg,
            comparator="lt",
        ),
        condition_metric(
            f"{metric_prefix}_adjoint_norm_finite",
            np.isfinite(metrics.adjoint_norm),
        ),
        condition_metric(
            f"{metric_prefix}_fd_norm_finite",
            np.isfinite(metrics.fd_norm),
        ),
    ]
    norm_metrics, norm_observations, _ = evaluate_allclose_agreement(
        np.asarray([metrics.adjoint_norm], dtype=float),
        np.asarray([metrics.fd_norm], dtype=float),
        rtol=norm_rtol,
        atol=norm_atol,
        metric_name=f"{metric_prefix}_norm_scaled_error",
    )
    regression_metrics.extend(norm_metrics)
    observation_metrics = [
        Metric(
            name=f"{metric_prefix}_adjoint_norm",
            observed=metrics.adjoint_norm,
            expected=0.0,
            comparator="gte",
        ),
        Metric(
            name=f"{metric_prefix}_fd_norm",
            observed=metrics.fd_norm,
            expected=0.0,
            comparator="gte",
        ),
    ]
    observation_metrics.extend(norm_observations)
    return regression_metrics, observation_metrics


def _collect_source_gradient_evaluation_data(
    numerical_case_dir,
    case: SourceCase,
    config: SweepConfig,
    derivative_target: str,
    *,
    label: str,
) -> EvaluationData:
    sim_path_dir = numerical_case_dir / "simulations"
    sim_path_dir.mkdir(parents=True, exist_ok=True)
    metrics = _run_gradient_case(
        sim_path_dir,
        case,
        config,
        derivative_target,
        label=label,
    )
    return _gradient_metrics_to_evaluation_data(metrics)


def _evaluate_source_gradient_evaluation_data(
    evaluation_data: EvaluationData,
    case: SourceCase,
    derivative_target: str,
    *,
    metric_prefix: str,
) -> tuple[list[Metric], list[Metric]]:
    metrics = _gradient_metrics_from_evaluation_data(evaluation_data)
    return _evaluate_source_gradient_metrics(
        metrics,
        case,
        derivative_target,
        metric_prefix=metric_prefix,
    )


def _finalize_source_gradient_result(
    *,
    request: pytest.FixtureRequest,
    numerical_case_dir,
    regression_metrics: list[Metric],
    observation_metrics: list[Metric],
) -> None:
    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Custom source adjoint and finite-difference gradients diverged; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )


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

    metrics = _gradient_metrics_from_gradients(grad_adjoint, grad_fd)

    print(f"[{label}] grad_adjoint = {grad_adjoint}", file=sys.stderr)
    print(f"[{label}] grad_fd      = {grad_fd}", file=sys.stderr)
    print(f"[{label}] angle_deg    = {metrics.angle_deg}", file=sys.stderr)
    print(f"[{label}] adjoint_norm = {metrics.adjoint_norm}", file=sys.stderr)
    print(f"[{label}] fd_norm      = {metrics.fd_norm}", file=sys.stderr)

    return metrics


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


def _collect_source_variation_sweep_evaluation_data(
    numerical_case_dir,
    case: SourceCase,
    derivative_target: str,
    variation_name: str,
    values: tuple,
    realized_configs: tuple[SweepConfig, ...],
) -> EvaluationData:
    resolved_target = _resolve_derivative_target(case, derivative_target)
    metrics_by_value = []
    sim_path_dir = numerical_case_dir / "simulations"
    sim_path_dir.mkdir(parents=True, exist_ok=True)
    for value, config in zip(values, realized_configs, strict=True):
        label = f"{case.name}_{resolved_target}_{variation_name}_{value}_{config.sim_dims}d"
        metrics_by_value.append(
            _run_gradient_case(sim_path_dir, case, config, derivative_target, label=label)
        )

    return {
        "grad_adjoint": np.stack([metrics.grad_adjoint for metrics in metrics_by_value]),
        "grad_fd": np.stack([metrics.grad_fd for metrics in metrics_by_value]),
    }


def _evaluate_source_variation_sweep_evaluation_data(
    evaluation_data: EvaluationData,
    case: SourceCase,
    derivative_target: str,
    variation_name: str,
    values: tuple,
) -> tuple[list[Metric], list[Metric]]:
    regression_metrics: list[Metric] = []
    observation_metrics: list[Metric] = []
    for idx, _value in enumerate(values):
        metrics = _gradient_metrics_from_evaluation_data(evaluation_data, index=idx)
        value_regression_metrics, value_observation_metrics = _evaluate_source_gradient_metrics(
            metrics,
            case,
            derivative_target,
            metric_prefix=f"{variation_name}_{idx}",
        )
        regression_metrics.extend(value_regression_metrics)
        observation_metrics.extend(value_observation_metrics)
    return regression_metrics, observation_metrics


def _run_variation_sweep(
    request: pytest.FixtureRequest,
    numerical_case_dir,
    numerical_eval_only: bool,
    case: SourceCase,
    sim_dims: int,
    derivative_target: str,
    variation_name: str,
    values: tuple,
    update_config: Callable[[SweepConfig, object], SweepConfig],
    *,
    test_name: str,
) -> None:
    base_config = replace(SweepConfig(), sim_dims=sim_dims)
    realized_configs = _realized_sweep_configs(base_config, values, update_config)
    case_identity = _source_variation_sweep_case_identity(
        test_name=test_name,
        case=case,
        derivative_target=derivative_target,
        base_config=base_config,
        variation_name=variation_name,
        variation_values=values,
        realized_configs=realized_configs,
    )
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_source_variation_sweep_evaluation_data(
            numerical_case_dir,
            case,
            derivative_target,
            variation_name,
            values,
            realized_configs,
        ),
    )
    regression_metrics, observation_metrics = _evaluate_source_variation_sweep_evaluation_data(
        evaluation_data,
        case,
        derivative_target,
        variation_name,
        values,
    )
    _finalize_source_gradient_result(
        request=request,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
    )


def _run_single_source_gradient_artifact(
    *,
    request: pytest.FixtureRequest,
    numerical_case_dir,
    numerical_eval_only: bool,
    test_name: str,
    case: SourceCase,
    config: SweepConfig,
    derivative_target: str,
    label: str,
    metric_prefix: str,
) -> None:
    case_identity = _source_gradient_case_identity(
        test_name=test_name,
        case=case,
        derivative_target=derivative_target,
        config=config,
    )
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_source_gradient_evaluation_data(
            numerical_case_dir,
            case,
            config,
            derivative_target,
            label=label,
        ),
    )
    regression_metrics, observation_metrics = _evaluate_source_gradient_evaluation_data(
        evaluation_data,
        case,
        derivative_target,
        metric_prefix=metric_prefix,
    )
    _finalize_source_gradient_result(
        request=request,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
    )


def _collect_phase_equivariance_evaluation_data(
    numerical_case_dir,
    case: SourceCase,
    config: SweepConfig,
) -> EvaluationData:
    sim_path_dir = numerical_case_dir / "simulations"
    sim_path_dir.mkdir(parents=True, exist_ok=True)
    base_params = tuple(np.asarray(BASE_PARAM_AMPLITUDES, dtype=complex))
    phased_params = tuple(1j * np.asarray(BASE_PARAM_AMPLITUDES, dtype=complex))

    grad_base = _run_adjoint_only_gradient_case(
        sim_path_dir,
        case,
        config,
        label=f"{case.name}_intensity_phase_base",
        params=base_params,
    )
    grad_phased = _run_adjoint_only_gradient_case(
        sim_path_dir,
        case,
        config,
        label=f"{case.name}_intensity_phase_j",
        params=phased_params,
    )
    return {
        "grad_base": np.asarray(grad_base, dtype=complex),
        "grad_phased": np.asarray(grad_phased, dtype=complex),
    }


def _evaluate_phase_equivariance_evaluation_data(
    evaluation_data: EvaluationData,
) -> tuple[list[Metric], list[Metric]]:
    grad_base = np.asarray(evaluation_data["grad_base"], dtype=complex)
    grad_phased = np.asarray(evaluation_data["grad_phased"], dtype=complex)
    regression_metrics = [
        condition_metric(
            "base_real_nonzero",
            not np.allclose(grad_base.real, 0.0, rtol=0.0, atol=1e-12),
        ),
        condition_metric(
            "base_imag_nonzero",
            not np.allclose(grad_base.imag, 0.0, rtol=0.0, atol=1e-12),
        ),
    ]
    phase_metrics, phase_observations, _ = evaluate_allclose_agreement(
        grad_phased,
        -1j * grad_base,
        rtol=1e-2,
        atol=1e-6,
        metric_name="phase_equivariance_scaled_error",
    )
    regression_metrics.extend(phase_metrics)
    return regression_metrics, phase_observations


def _skip_2d_field_cases(sim_dims: int, case: SourceCase) -> None:
    if sim_dims == 2 and case.source_kind == "field":
        pytest.skip("2D variation sweeps are only run for CustomCurrentSource cases.")


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(sim_dims_values=SIM_DIMS_VALUES),
    ids=lambda params: _source_variation_sweep_case_id(
        test_name="custom_source_gradient_vs_dataset_spacing",
        params=params,
        base_config=replace(SweepConfig(), sim_dims=params.sim_dims),
        variation_name="dataset_spacing",
        variation_values=DATASET_SPACING_VALUES,
        update_config=_dataset_spacing_sweep_config,
    ),
)
def test_custom_source_gradient_vs_dataset_spacing(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    sim_dims = params.sim_dims
    derivative_target = params.derivative_target
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        request,
        numerical_case_dir,
        numerical_eval_only,
        case,
        sim_dims,
        derivative_target,
        "dataset_spacing",
        DATASET_SPACING_VALUES,
        _dataset_spacing_sweep_config,
        test_name="custom_source_gradient_vs_dataset_spacing",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(sim_dims_values=SIM_DIMS_VALUES),
    ids=lambda params: _source_variation_sweep_case_id(
        test_name="custom_source_gradient_vs_grid_resolution",
        params=params,
        base_config=replace(SweepConfig(), sim_dims=params.sim_dims),
        variation_name="min_steps_per_wvl",
        variation_values=MIN_STEPS_PER_WVL_VALUES,
        update_config=_grid_resolution_sweep_config,
    ),
)
def test_custom_source_gradient_vs_grid_resolution(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    sim_dims = params.sim_dims
    derivative_target = params.derivative_target
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        request,
        numerical_case_dir,
        numerical_eval_only,
        case,
        sim_dims,
        derivative_target,
        "min_steps_per_wvl",
        MIN_STEPS_PER_WVL_VALUES,
        _grid_resolution_sweep_config,
        test_name="custom_source_gradient_vs_grid_resolution",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(sim_dims_values=SIM_DIMS_VALUES),
    ids=lambda params: _source_variation_sweep_case_id(
        test_name="custom_source_gradient_vs_source_size",
        params=params,
        base_config=replace(SweepConfig(), sim_dims=params.sim_dims),
        variation_name="source_size_xy",
        variation_values=SOURCE_SIZE_XY_VALUES,
        update_config=_source_size_sweep_config,
    ),
)
def test_custom_source_gradient_vs_source_size(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    sim_dims = params.sim_dims
    derivative_target = params.derivative_target
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        request,
        numerical_case_dir,
        numerical_eval_only,
        case,
        sim_dims,
        derivative_target,
        "source_size_xy",
        SOURCE_SIZE_XY_VALUES,
        _source_size_sweep_config,
        test_name="custom_source_gradient_vs_source_size",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(sim_dims_values=SIM_DIMS_VALUES),
    ids=lambda params: _source_variation_sweep_case_id(
        test_name="custom_source_gradient_vs_amplitude",
        params=params,
        base_config=replace(SweepConfig(), sim_dims=params.sim_dims),
        variation_name="amplitude_scale",
        variation_values=AMPLITUDE_SCALE_VALUES,
        update_config=_amplitude_sweep_config,
    ),
)
def test_custom_source_gradient_vs_amplitude(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    sim_dims = params.sim_dims
    derivative_target = params.derivative_target
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        request,
        numerical_case_dir,
        numerical_eval_only,
        case,
        sim_dims,
        derivative_target,
        "amplitude_scale",
        AMPLITUDE_SCALE_VALUES,
        _amplitude_sweep_config,
        test_name="custom_source_gradient_vs_amplitude",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(sim_dims_values=SIM_DIMS_VALUES),
    ids=lambda params: _source_variation_sweep_case_id(
        test_name="custom_source_gradient_vs_permittivity",
        params=params,
        base_config=replace(SweepConfig(), sim_dims=params.sim_dims),
        variation_name="permittivity",
        variation_values=PERMITTIVITY_VALUES,
        update_config=_permittivity_sweep_config,
    ),
)
def test_custom_source_gradient_vs_permittivity(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    sim_dims = params.sim_dims
    derivative_target = params.derivative_target
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        request,
        numerical_case_dir,
        numerical_eval_only,
        case,
        sim_dims,
        derivative_target,
        "permittivity",
        PERMITTIVITY_VALUES,
        _permittivity_sweep_config,
        test_name="custom_source_gradient_vs_permittivity",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(sim_dims_values=SIM_DIMS_VALUES),
    ids=lambda params: _source_variation_sweep_case_id(
        test_name="custom_source_gradient_vs_wavelength",
        params=params,
        base_config=replace(SweepConfig(), sim_dims=params.sim_dims),
        variation_name="wvl0",
        variation_values=WVL0_VALUES,
        update_config=_wavelength_sweep_config,
    ),
)
def test_custom_source_gradient_vs_wavelength(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    sim_dims = params.sim_dims
    derivative_target = params.derivative_target
    _skip_2d_field_cases(sim_dims, case)
    _run_variation_sweep(
        request,
        numerical_case_dir,
        numerical_eval_only,
        case,
        sim_dims,
        derivative_target,
        "wvl0",
        WVL0_VALUES,
        _wavelength_sweep_config,
        test_name="custom_source_gradient_vs_wavelength",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(
        case_names=("custom_field_vec_e", "custom_current_vec_e"),
    ),
    ids=lambda params: _source_gradient_case_id(
        test_name="custom_source_gradient_off_center_monitor_frequency",
        params=params,
        config=replace(
            SweepConfig(sim_dims=3),
            monitor_freq_scale=OFF_CENTER_MONITOR_FREQ_SCALE,
        ),
    ),
)
def test_custom_source_gradient_off_center_monitor_frequency(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    """Check source gradients when objective frequency is offset from source center frequency."""
    case = _source_case_from_name(params.case_name)
    derivative_target = params.derivative_target
    resolved_target = _resolve_derivative_target(case, derivative_target)
    config = replace(
        SweepConfig(sim_dims=3),
        monitor_freq_scale=OFF_CENTER_MONITOR_FREQ_SCALE,
    )
    label = f"{case.name}_{resolved_target}_monitor_freq_scale_{OFF_CENTER_MONITOR_FREQ_SCALE}_3d"
    _run_single_source_gradient_artifact(
        request=request,
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        test_name="custom_source_gradient_off_center_monitor_frequency",
        case=case,
        config=config,
        derivative_target=derivative_target,
        label=label,
        metric_prefix="off_center_monitor_frequency",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(
        case_names=("custom_field_vec_e", "custom_current_vec_e"),
    ),
    ids=lambda params: _source_gradient_case_id(
        test_name="custom_source_gradient_two_frequency_flux_difference",
        params=params,
        config=replace(
            SweepConfig(sim_dims=3),
            objective_3d="flux_difference",
            monitor_freq_scales=TWO_FREQ_MONITOR_SCALES,
        ),
    ),
)
def test_custom_source_gradient_two_frequency_flux_difference(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    """Check source gradients with objective flux(f0) - flux(0.95*f0) in 3D."""
    case = _source_case_from_name(params.case_name)
    derivative_target = params.derivative_target
    resolved_target = _resolve_derivative_target(case, derivative_target)
    config = replace(
        SweepConfig(sim_dims=3),
        objective_3d="flux_difference",
        monitor_freq_scales=TWO_FREQ_MONITOR_SCALES,
    )
    label = f"{case.name}_{resolved_target}_flux_difference_3d"
    _run_single_source_gradient_artifact(
        request=request,
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        test_name="custom_source_gradient_two_frequency_flux_difference",
        case=case,
        config=config,
        derivative_target=derivative_target,
        label=label,
        metric_prefix="two_frequency_flux_difference",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(),
    ids=lambda params: _source_gradient_case_id(
        test_name="custom_source_gradient_inside_structure",
        params=params,
        config=replace(SweepConfig(), source_structure_permittivity=4.0),
    ),
)
def test_custom_source_gradient_inside_structure(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    derivative_target = params.derivative_target
    resolved_target = _resolve_derivative_target(case, derivative_target)
    structure_config = replace(SweepConfig(), source_structure_permittivity=4.0)
    _run_single_source_gradient_artifact(
        request=request,
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        test_name="custom_source_gradient_inside_structure",
        case=case,
        config=structure_config,
        derivative_target=derivative_target,
        label=f"{case.name}_{resolved_target}_source_in_structure_eps_4",
        metric_prefix="inside_structure",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(case_names=("custom_current_vec_e",)),
    ids=lambda params: _source_gradient_case_id(
        test_name="custom_current_source_gradient_cube_size",
        params=params,
        config=replace(SweepConfig(), source_size=(0.5, 0.5, 0.5)),
    ),
)
def test_custom_current_source_gradient_cube_size(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    current_case = _source_case_from_name(params.case_name)
    derivative_target = params.derivative_target
    resolved_target = _resolve_derivative_target(current_case, derivative_target)
    cube_config = replace(SweepConfig(), source_size=(0.5, 0.5, 0.5))
    _run_single_source_gradient_artifact(
        request=request,
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        test_name="custom_current_source_gradient_cube_size",
        case=current_case,
        config=cube_config,
        derivative_target=derivative_target,
        label=f"{current_case.name}_{resolved_target}_source_size_xyz_0.5",
        metric_prefix="cube_size",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params",
    _source_gradient_parameters(),
    ids=lambda params: _source_gradient_case_id(
        test_name="custom_source_gradient_in_nonuniform_custom_medium",
        params=params,
        config=replace(SweepConfig(), source_structure_custom_medium=True),
    ),
)
def test_custom_source_gradient_in_nonuniform_custom_medium(
    request: pytest.FixtureRequest,
    params: SourceGradientTestParameters,
    numerical_case_dir,
    numerical_eval_only: bool,
):
    case = _source_case_from_name(params.case_name)
    derivative_target = params.derivative_target
    resolved_target = _resolve_derivative_target(case, derivative_target)
    custom_medium_config = replace(SweepConfig(), source_structure_custom_medium=True)
    _run_single_source_gradient_artifact(
        request=request,
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        test_name="custom_source_gradient_in_nonuniform_custom_medium",
        case=case,
        config=custom_medium_config,
        derivative_target=derivative_target,
        label=f"{case.name}_{resolved_target}_source_in_custom_medium",
        metric_prefix="nonuniform_custom_medium",
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "case_name",
    ("custom_field_vec_e", "custom_current_vec_e"),
    ids=_source_phase_case_id,
)
def test_custom_source_intensity_gradient_global_phase_equivariance(
    request: pytest.FixtureRequest,
    case_name,
    numerical_case_dir,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    """Intensity objective gradients should rotate with a global ``1j`` source phase."""
    case = _source_case_from_name(case_name)
    config = _phase_equivariance_config()

    case_identity = _source_phase_case_identity(
        test_name="custom_source_intensity_gradient_global_phase_equivariance",
        case=case,
        config=config,
    )
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_phase_equivariance_evaluation_data(
            numerical_case_dir,
            case,
            config,
        ),
    )
    regression_metrics, observation_metrics = _evaluate_phase_equivariance_evaluation_data(
        evaluation_data
    )

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Custom source global-phase gradient equivariance failed; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
