from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import autograd.numpy as anp
import numpy as np
import pytest
from autograd.core import make_vjp

import tidy3d as td

RESULTS_SUBDIR = "field_projection_benchmarks"


@dataclass(frozen=True)
class ProjectionBenchmarkConfig:
    name: str
    projection_type: str
    far_field_approx: bool
    src_n: int
    obs_n: int
    size: tuple[float, float, float]
    sim_size: tuple[float, float, float]
    freq: float
    proj_distance: float
    pts_per_wavelength: int | None = 10
    smooth_phase: bool = False


def _make_projection_monitor(
    config: ProjectionBenchmarkConfig, center: tuple[float, float, float]
) -> td.AbstractFieldProjectionMonitor:
    if config.projection_type == "angular":
        theta = np.linspace(np.pi / 8, np.pi - np.pi / 8, config.obs_n)
        phi = np.linspace(0, 2 * np.pi, config.obs_n)
        return td.FieldProjectionAngleMonitor(
            center=center,
            size=config.size,
            freqs=[config.freq],
            name=f"{config.name}_far",
            custom_origin=center,
            theta=list(theta),
            phi=list(phi),
            proj_distance=config.proj_distance,
            normal_dir="+",
            far_field_approx=config.far_field_approx,
        )

    if config.projection_type == "cartesian":
        xy = np.linspace(-config.proj_distance / 10, config.proj_distance / 10, config.obs_n)
        return td.FieldProjectionCartesianMonitor(
            center=center,
            size=config.size,
            freqs=[config.freq],
            name=f"{config.name}_far",
            custom_origin=center,
            x=list(xy),
            y=list(xy),
            proj_axis=0,
            proj_distance=config.proj_distance,
            normal_dir="+",
            far_field_approx=config.far_field_approx,
        )

    if config.projection_type == "kspace":
        uv = np.linspace(-0.4, 0.4, config.obs_n)
        return td.FieldProjectionKSpaceMonitor(
            center=center,
            size=config.size,
            freqs=[config.freq],
            name=f"{config.name}_far",
            custom_origin=center,
            ux=list(uv),
            uy=list(uv),
            proj_axis=0,
            proj_distance=config.proj_distance,
            normal_dir="+",
            far_field_approx=config.far_field_approx,
        )

    raise ValueError(f"Unsupported projection type: {config.projection_type}")


def _make_field_values(
    x: np.ndarray, y: np.ndarray, scale: float, *, smooth_phase: bool
) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if smooth_phase:
        pattern = np.exp(1j * 2 * np.pi * (0.03 * xx + 0.02 * yy))
    else:
        rng = np.random.default_rng(0)
        pattern = (1 + 1j) * rng.random((len(x), len(y)))
    return scale * pattern[..., None, None]


def _build_projector_and_monitor(
    config: ProjectionBenchmarkConfig, scale: float = 1.0
) -> tuple[td.FieldProjector, td.AbstractFieldProjectionMonitor]:
    center = (0.0, 0.0, 0.0)
    monitor = td.FieldMonitor(
        size=config.size,
        center=center,
        freqs=[config.freq],
        name=f"{config.name}_near",
        colocate=False,
    )

    sim = td.Simulation(
        size=config.sim_size,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / config.freq),
        monitors=(monitor,),
        run_time=1e-12,
    )

    x = np.linspace(-config.size[0] / 2, config.size[0] / 2, config.src_n)
    y = np.linspace(-config.size[1] / 2, config.size[1] / 2, config.src_n)
    z = np.array([0.0])
    coords = {"x": x, "y": y, "z": z, "f": [config.freq]}

    scalar_field = td.ScalarFieldDataArray(
        _make_field_values(x, y, scale, smooth_phase=config.smooth_phase),
        coords=coords,
    )

    data = td.FieldData(
        monitor=monitor,
        Ex=scalar_field,
        Ey=scalar_field,
        Ez=scalar_field,
        Hx=scalar_field,
        Hy=scalar_field,
        Hz=scalar_field,
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(monitor),
    )

    sim_data = td.SimulationData(simulation=sim, data=(data,))
    projector = td.FieldProjector.from_near_field_monitors(
        sim_data=sim_data,
        near_monitors=[monitor],
        normal_dirs=["+"],
        pts_per_wavelength=config.pts_per_wavelength,
    )
    proj_monitor = _make_projection_monitor(config, center)
    return projector, proj_monitor


def _projection_objective(config: ProjectionBenchmarkConfig, scale: float):
    projector, proj_monitor = _build_projector_and_monitor(config, scale=scale)
    projected = projector.project_fields(proj_monitor, verbose=False)
    fields = projected.Etheta.data, projected.Ephi.data
    total = sum(anp.conj(field) * field for field in fields)
    return anp.real(anp.sum(total))


def _measure_callable(func, *, repeat: int = 3, warmup: int = 1) -> dict[str, Any]:
    for _ in range(warmup):
        func()

    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        samples.append(time.perf_counter() - t0)

    return {
        "samples_s": samples,
        "mean_s": float(np.mean(samples)),
        "median_s": float(np.median(samples)),
    }


def _measure_vjp(config: ProjectionBenchmarkConfig, *, repeat: int = 2) -> dict[str, Any]:
    trace_samples = []
    backward_samples = []
    objective_values = []
    gradients = []

    for _ in range(repeat):
        t0 = time.perf_counter()
        vjp, ans = make_vjp(lambda scale: _projection_objective(config, scale), 1.0)
        t1 = time.perf_counter()
        grad = vjp(1.0)
        t2 = time.perf_counter()

        trace_samples.append(t1 - t0)
        backward_samples.append(t2 - t1)
        objective_values.append(float(ans))
        gradients.append(float(grad))

    return {
        "trace_forward_samples_s": trace_samples,
        "trace_forward_mean_s": float(np.mean(trace_samples)),
        "trace_forward_median_s": float(np.median(trace_samples)),
        "backward_samples_s": backward_samples,
        "backward_mean_s": float(np.mean(backward_samples)),
        "backward_median_s": float(np.median(backward_samples)),
        "objective_value": float(np.mean(objective_values)),
        "gradient_value": float(np.mean(gradients)),
    }


def _write_benchmark_artifacts(
    results: list[dict[str, Any]], output_dir: Path, stem: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf8")

    numeric_keys = [
        key
        for key in results[0].keys()
        if isinstance(results[0][key], (int, float, np.integer, np.floating))
    ]
    npz_path = output_dir / f"{stem}.npz"
    np.savez(
        npz_path,
        case_names=np.array([result["name"] for result in results], dtype=object),
        **{key: np.array([result[key] for result in results], dtype=float) for key in numeric_keys},
    )

    return json_path, npz_path


@pytest.mark.perf
def test_field_projection_projection_type_benchmarks(
    performance_case_dir: Path, redirect_stdout_to_stderr
):
    """Benchmark local field-projection forward and autodiff cost across projection types."""
    configs = (
        ProjectionBenchmarkConfig(
            name="angular_approx",
            projection_type="angular",
            far_field_approx=True,
            src_n=32,
            obs_n=21,
            size=(2.0, 2.0, 0.0),
            sim_size=(5.0, 5.0, 5.0),
            freq=1e13,
            proj_distance=50.0,
        ),
        ProjectionBenchmarkConfig(
            name="cartesian_approx",
            projection_type="cartesian",
            far_field_approx=True,
            src_n=32,
            obs_n=31,
            size=(2.0, 2.0, 0.0),
            sim_size=(5.0, 5.0, 5.0),
            freq=1e13,
            proj_distance=50.0,
        ),
        ProjectionBenchmarkConfig(
            name="kspace_approx",
            projection_type="kspace",
            far_field_approx=True,
            src_n=32,
            obs_n=31,
            size=(2.0, 2.0, 0.0),
            sim_size=(5.0, 5.0, 5.0),
            freq=1e13,
            proj_distance=50.0,
        ),
        ProjectionBenchmarkConfig(
            name="cartesian_exact",
            projection_type="cartesian",
            far_field_approx=False,
            src_n=16,
            obs_n=9,
            size=(2.0, 2.0, 0.0),
            sim_size=(5.0, 5.0, 5.0),
            freq=1e13,
            proj_distance=50.0,
        ),
    )

    print(f"Saving field projection benchmark artifacts in {performance_case_dir}")
    results = []
    for config in configs:
        projector, proj_monitor = _build_projector_and_monitor(config)
        project_fields_stats = _measure_callable(
            lambda proj=projector, mon=proj_monitor: proj.project_fields(mon, verbose=False),
            repeat=3,
            warmup=1,
        )
        vjp_stats = _measure_vjp(config, repeat=2)

        result = {
            **asdict(config),
            "project_fields_mean_s": project_fields_stats["mean_s"],
            "project_fields_median_s": project_fields_stats["median_s"],
            "trace_forward_mean_s": vjp_stats["trace_forward_mean_s"],
            "trace_forward_median_s": vjp_stats["trace_forward_median_s"],
            "backward_mean_s": vjp_stats["backward_mean_s"],
            "backward_median_s": vjp_stats["backward_median_s"],
            "objective_value": vjp_stats["objective_value"],
            "gradient_value": vjp_stats["gradient_value"],
            "project_fields_samples_s": project_fields_stats["samples_s"],
            "trace_forward_samples_s": vjp_stats["trace_forward_samples_s"],
            "backward_samples_s": vjp_stats["backward_samples_s"],
        }
        print(json.dumps(result, indent=2))
        results.append(result)

    output_dir = performance_case_dir / RESULTS_SUBDIR
    json_path, npz_path = _write_benchmark_artifacts(
        results, output_dir, stem="projection_type_benchmarks"
    )

    assert json_path.exists()
    assert npz_path.exists()
    assert all(result["project_fields_mean_s"] > 0 for result in results)
    assert all(result["trace_forward_mean_s"] > 0 for result in results)
    assert all(result["backward_mean_s"] > 0 for result in results)


@pytest.mark.perf
def test_field_projection_resampling_benchmarks(
    performance_case_dir: Path, redirect_stdout_to_stderr
):
    """Benchmark local approximate Cartesian projection with varying resampling density."""
    base_kwargs = {
        "projection_type": "cartesian",
        "far_field_approx": True,
        "src_n": 128,
        "obs_n": 21,
        "size": (20.0, 20.0, 0.0),
        "sim_size": (24.0, 24.0, 8.0),
        "freq": 3e14,
        "proj_distance": 100.0,
        "smooth_phase": True,
    }
    configs = (
        ProjectionBenchmarkConfig(name="cartesian_ppw4", pts_per_wavelength=4, **base_kwargs),
        ProjectionBenchmarkConfig(name="cartesian_ppw6", pts_per_wavelength=6, **base_kwargs),
        ProjectionBenchmarkConfig(name="cartesian_ppw10", pts_per_wavelength=10, **base_kwargs),
        ProjectionBenchmarkConfig(
            name="cartesian_raw_grid", pts_per_wavelength=None, **base_kwargs
        ),
    )

    print(f"Saving field projection resampling artifacts in {performance_case_dir}")
    results = []
    for config in configs:
        projector, proj_monitor = _build_projector_and_monitor(config)
        project_fields_stats = _measure_callable(
            lambda proj=projector, mon=proj_monitor: proj.project_fields(mon, verbose=False),
            repeat=2,
            warmup=1,
        )
        projected = projector.project_fields(proj_monitor, verbose=False)
        power = projected.power.data
        currents = projector.currents[projector.surfaces[0].monitor.name]

        result = {
            **asdict(config),
            "project_fields_mean_s": project_fields_stats["mean_s"],
            "project_fields_median_s": project_fields_stats["median_s"],
            "power_sum": float(np.real(np.sum(power))),
            "resampled_nx": len(currents["x"]),
            "resampled_ny": len(currents["y"]),
            "resampled_nz": len(currents["z"]),
            "project_fields_samples_s": project_fields_stats["samples_s"],
        }
        print(json.dumps(result, indent=2))
        results.append(result)

    output_dir = performance_case_dir / RESULTS_SUBDIR
    json_path, npz_path = _write_benchmark_artifacts(
        results, output_dir, stem="resampling_benchmarks"
    )

    assert json_path.exists()
    assert npz_path.exists()
    assert all(result["project_fields_mean_s"] > 0 for result in results)
    assert all(result["resampled_nx"] > 0 for result in results)
