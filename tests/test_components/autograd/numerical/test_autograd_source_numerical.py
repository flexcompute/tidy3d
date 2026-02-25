from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Callable

import autograd as ag
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


ANGLE_LIMIT_DEG = 5.0
NORM_RTOL = 0.1
NORM_ATOL = 1e-6

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


def _axis_coords(size: float, spacing: float) -> np.ndarray:
    if size <= 0:
        return np.array([0.0], dtype=float)
    n = max(2, int(np.round(size / spacing)))
    return np.linspace(-size / 2, size / 2, n)


def _make_coords(
    source_size: tuple[float, float, float], dataset_spacing: float, freq0: float
) -> dict[str, object]:
    return {
        "x": _axis_coords(source_size[0], dataset_spacing),
        "y": _axis_coords(source_size[1], dataset_spacing),
        "z": _axis_coords(source_size[2], dataset_spacing),
        "f": [freq0],
    }


def _make_field_dataset(
    field_prefix: str,
    amplitudes: tuple[float, float, float],
    coords: dict[str, object],
) -> td.FieldDataset:
    shape = (
        len(np.asarray(coords["x"])),
        len(np.asarray(coords["y"])),
        len(np.asarray(coords["z"])),
        1,
    )
    components = {}
    for amp, axis in zip(amplitudes, "xyz"):
        data = amp * np.ones(shape, dtype=float)
        components[f"{field_prefix}{axis}"] = td.ScalarFieldDataArray(data, coords=coords)
    return td.FieldDataset(**components)


def _scaled_amplitudes(scale: float) -> tuple[float, float, float]:
    return tuple(scale * value for value in BASE_PARAM_AMPLITUDES)


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
    amplitude_scale: float = 1.0
    background_permittivity: float = 1.0
    source_structure_permittivity: float | None = None
    source_structure_custom_medium: bool = False


@dataclass(frozen=True)
class SourceCase:
    name: str
    monitor_components: tuple[str, str, str]
    source_kind: str
    field_prefix: str
    delta: float = 1e-4


@dataclass(frozen=True)
class GradientMetrics:
    grad_adjoint: np.ndarray
    grad_fd: np.ndarray
    angle_deg: float
    adjoint_norm: float
    fd_norm: float


SOURCE_CASES = (
    SourceCase(
        name="custom_field_vec_e",
        monitor_components=("Ex", "Ey", "Ez"),
        source_kind="field",
        field_prefix="E",
    ),
    SourceCase(
        name="custom_field_vec_h",
        monitor_components=("Hx", "Hy", "Hz"),
        source_kind="field",
        field_prefix="H",
    ),
    SourceCase(
        name="custom_current_vec_e",
        monitor_components=("Ex", "Ey", "Ez"),
        source_kind="current",
        field_prefix="E",
    ),
    SourceCase(
        name="custom_current_vec_h",
        monitor_components=("Hx", "Hy", "Hz"),
        source_kind="current",
        field_prefix="H",
    ),
)


def _make_source(
    case: SourceCase,
    amplitudes: tuple[float, float, float],
    config: SweepConfig,
    freq0: float,
    pulse: td.GaussianPulse,
) -> td.Source:
    if case.source_kind == "field":
        source_size = (config.source_size[0], config.source_size[1], 0.0)
        coords = _make_coords(source_size, config.dataset_spacing, freq0)
        field_dataset = _make_field_dataset(case.field_prefix, amplitudes, coords)
        return td.CustomFieldSource(
            center=BASE_SOURCE_CENTER,
            size=source_size,
            source_time=pulse,
            field_dataset=field_dataset,
        )

    if case.source_kind == "current":
        coords = _make_coords(config.source_size, config.dataset_spacing, freq0)
        current_dataset = _make_field_dataset(case.field_prefix, amplitudes, coords)
        return td.CustomCurrentSource(
            center=BASE_SOURCE_CENTER,
            size=config.source_size,
            source_time=pulse,
            current_dataset=current_dataset,
        )

    raise ValueError(f"Unsupported source_kind: {case.source_kind}")


def _source_host_structure(
    source: td.Source,
    source_structure_permittivity: float,
) -> td.Structure:
    """Create the enclosing structure ("host") that contains the source."""
    source_size = tuple(float(value) for value in source.size)
    host_size = (
        source_size[0] + 0.2,
        source_size[1] + 0.2,
        max(source_size[2] + 0.2, 0.3),
    )
    return td.Structure(
        geometry=td.Box(center=source.center, size=host_size),
        medium=td.Medium(permittivity=source_structure_permittivity),
    )


def _source_host_custom_medium(source: td.Source) -> td.Structure:
    """Create a nonuniform custom-medium host structure that encloses the source."""
    source_size = tuple(float(value) for value in source.size)
    host_size = (
        source_size[0] + 0.2,
        source_size[1] + 0.2,
        max(source_size[2] + 0.2, 0.3),
    )
    host_center = tuple(float(value) for value in source.center)
    host_bounds_min = [c - 0.5 * s for c, s in zip(host_center, host_size)]
    host_bounds_max = [c + 0.5 * s for c, s in zip(host_center, host_size)]

    x = np.linspace(host_bounds_min[0], host_bounds_max[0], 9)
    y = np.linspace(host_bounds_min[1], host_bounds_max[1], 9)
    z = np.linspace(host_bounds_min[2], host_bounds_max[2], 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    eps = 2.5 + 0.2 * (X - host_center[0]) + 0.15 * (Y - host_center[1]) ** 2
    eps += 0.05 * (Z - host_center[2])

    return td.Structure(
        geometry=td.Box(center=source.center, size=host_size),
        medium=td.CustomMedium(
            permittivity=td.SpatialDataArray(
                eps,
                coords={"x": x, "y": y, "z": z},
            )
        ),
    )


def _make_sim(
    source: td.Source,
    config: SweepConfig,
) -> td.Simulation:
    freq0 = td.C_0 / config.wvl0
    monitor = td.FieldMonitor(
        name=FLUX_MONITOR_NAME,
        center=MONITOR_CENTER,
        size=MONITOR_SIZE,
        freqs=[freq0],
    )
    structures = []
    if config.source_structure_permittivity is not None and config.source_structure_custom_medium:
        raise ValueError(
            "Only one source host type is allowed: set either "
            "'source_structure_permittivity' or 'source_structure_custom_medium'."
        )

    if config.source_structure_permittivity is not None:
        structures.append(_source_host_structure(source, config.source_structure_permittivity))
    if config.source_structure_custom_medium:
        structures.append(_source_host_custom_medium(source))

    sim_size = (3 * config.wvl0, 3 * config.wvl0, 3 * config.wvl0)
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


def _eval_objective_flux(sim_data: td.SimulationData) -> float:
    field_data = sim_data.load_field_monitor(FLUX_MONITOR_NAME)
    return field_data.flux.values


def _run_gradient_case(
    tmp_path,
    case: SourceCase,
    config: SweepConfig,
    *,
    label: str,
) -> GradientMetrics:
    freq0 = td.C_0 / config.wvl0
    pulse = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    params = _scaled_amplitudes(config.amplitude_scale)

    def make_source(amps: tuple[float, float, float]) -> td.Source:
        return _make_source(case, amps, config, freq0, pulse)

    def objective_adj(ax: float, ay: float, az: float) -> float:
        sim = _make_sim(make_source((ax, ay, az)), config)
        sim_data = web.run(
            sim,
            task_name=f"{label}_adj",
            path=tmp_path / f"{label}_adj.hdf5",
            local_gradient=True,
            verbose=False,
        )
        return _eval_objective_flux(sim_data)

    grad_adjoint = np.array(
        [
            ag.grad(objective_adj, 0)(*params),
            ag.grad(objective_adj, 1)(*params),
            ag.grad(objective_adj, 2)(*params),
        ],
        dtype=float,
    )

    sims = {}
    for idx, axis in enumerate("xyz"):
        params_plus = list(params)
        params_plus[idx] += case.delta
        params_minus = list(params)
        params_minus[idx] -= case.delta
        sims[f"{label}_fd_{axis}_plus"] = _make_sim(make_source(tuple(params_plus)), config)
        sims[f"{label}_fd_{axis}_minus"] = _make_sim(make_source(tuple(params_minus)), config)

    sim_data_map = web.run_async(
        sims,
        path_dir=tmp_path,
        local_gradient=False,
        verbose=False,
    )

    grad_fd = np.zeros(3, dtype=float)
    for idx, axis in enumerate("xyz"):
        obj_plus = float(
            np.asarray(_eval_objective_flux(sim_data_map[f"{label}_fd_{axis}_plus"])).squeeze()
        )
        obj_minus = float(
            np.asarray(_eval_objective_flux(sim_data_map[f"{label}_fd_{axis}_minus"])).squeeze()
        )
        grad_fd[idx] = (obj_plus - obj_minus) / (2 * case.delta)

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


def _assert_fd_agreement(metrics: GradientMetrics, *, label: str) -> None:
    assert metrics.angle_deg < ANGLE_LIMIT_DEG, label
    assert np.isfinite(metrics.adjoint_norm), label
    assert np.isfinite(metrics.fd_norm), label
    np.testing.assert_allclose(
        metrics.adjoint_norm,
        metrics.fd_norm,
        rtol=NORM_RTOL,
        atol=NORM_ATOL,
        err_msg=label,
    )


def _run_variation_sweep(
    tmp_path,
    case: SourceCase,
    variation_name: str,
    values: tuple,
    update_config: Callable[[SweepConfig, object], SweepConfig],
) -> None:
    base_config = SweepConfig()
    for value in values:
        config = update_config(base_config, value)
        label = f"{case.name}_{variation_name}_{value}"
        metrics = _run_gradient_case(tmp_path, case, config, label=label)
        _assert_fd_agreement(metrics, label=label)


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_dataset_spacing(_enable_local_cache, tmp_path, case):
    _run_variation_sweep(
        tmp_path,
        case,
        "dataset_spacing",
        DATASET_SPACING_VALUES,
        lambda base, value: replace(base, dataset_spacing=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_grid_resolution(_enable_local_cache, tmp_path, case):
    _run_variation_sweep(
        tmp_path,
        case,
        "min_steps_per_wvl",
        MIN_STEPS_PER_WVL_VALUES,
        lambda base, value: replace(base, min_steps_per_wvl=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_source_size(_enable_local_cache, tmp_path, case):
    _run_variation_sweep(
        tmp_path,
        case,
        "source_size_xy",
        SOURCE_SIZE_XY_VALUES,
        lambda base, value: replace(base, source_size=(value, value, 0.0)),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_amplitude(_enable_local_cache, tmp_path, case):
    _run_variation_sweep(
        tmp_path,
        case,
        "amplitude_scale",
        AMPLITUDE_SCALE_VALUES,
        lambda base, value: replace(base, amplitude_scale=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_permittivity(_enable_local_cache, tmp_path, case):
    _run_variation_sweep(
        tmp_path,
        case,
        "permittivity",
        PERMITTIVITY_VALUES,
        lambda base, value: replace(
            base,
            background_permittivity=value[0],
            source_structure_permittivity=value[1],
        ),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_vs_wavelength(_enable_local_cache, tmp_path, case):
    _run_variation_sweep(
        tmp_path,
        case,
        "wvl0",
        WVL0_VALUES,
        lambda base, value: replace(base, wvl0=value),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_inside_structure(_enable_local_cache, tmp_path, case):
    structure_config = replace(SweepConfig(), source_structure_permittivity=4.0)
    structure_metrics = _run_gradient_case(
        tmp_path,
        case,
        structure_config,
        label=f"{case.name}_source_in_structure_eps_4",
    )
    _assert_fd_agreement(structure_metrics, label=f"{case.name}_source_in_structure_eps_4")


@pytest.mark.numerical
def test_custom_current_source_gradient_cube_size(_enable_local_cache, tmp_path):
    current_case = next(case for case in SOURCE_CASES if case.name == "custom_current_vec_e")
    cube_config = replace(SweepConfig(), source_size=(0.5, 0.5, 0.5))
    cube_metrics = _run_gradient_case(
        tmp_path,
        current_case,
        cube_config,
        label=f"{current_case.name}_source_size_xyz_0.5",
    )
    _assert_fd_agreement(cube_metrics, label=f"{current_case.name}_source_size_xyz_0.5")


@pytest.mark.numerical
@pytest.mark.parametrize("case", SOURCE_CASES, ids=lambda case: case.name)
def test_custom_source_gradient_in_nonuniform_custom_medium(_enable_local_cache, tmp_path, case):
    custom_medium_config = replace(SweepConfig(), source_structure_custom_medium=True)
    custom_medium_metrics = _run_gradient_case(
        tmp_path,
        case,
        custom_medium_config,
        label=f"{case.name}_source_in_custom_medium",
    )
    _assert_fd_agreement(custom_medium_metrics, label=f"{case.name}_source_in_custom_medium")
