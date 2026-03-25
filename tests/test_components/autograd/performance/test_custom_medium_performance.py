from __future__ import annotations

import cProfile
import os
import pstats
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import tidy3d as td
from tidy3d.components import medium as medium_module
from tidy3d.components.autograd import derivative_utils as derivative_utils_module
from tidy3d.components.data import sim_data as sim_data_module
from tidy3d.components.medium import AbstractCustomMedium, CustomMedium
from tidy3d.web.api.autograd import backward as backward_module
from tidy3d.web.api.autograd.backward import postprocess_adj

from .postprocess_adj_utils import (
    CUSTOM_MEDIUM_DATASET_ROOT,
    PostprocessAdjInputs,
    dataset_exists,
    generate_custom_medium_postprocess_adj_inputs,
    load_postprocess_adj_inputs,
    persist_postprocess_adj_inputs,
)

MARK_GENERATE_NAME = "generate_profile"

# Tune these values to stress-test specific memory/performance regimes.
CUSTOM_MEDIUM_BOX_SIZE = (12.0, 12.0, 2.0)
CUSTOM_MEDIUM_PIXEL_SIZE = 0.05
CUSTOM_MEDIUM_NUM_FREQS = 20
CUSTOM_MEDIUM_PERMITTIVITY = 2.25
CUSTOM_MEDIUM_WAVELENGTH = 1.55

# Run mode for comparison. Set TIDY3D_PROFILE_CHUNK_MODE to one of:
#   all, half, 2, 1, none
PROFILE_CHUNK_MODE = os.getenv("TIDY3D_PROFILE_CHUNK_MODE", "none").strip().lower()
PROFILE_AVAILABLE_MEMORY_GB = os.getenv("TIDY3D_PROFILE_AVAILABLE_MEMORY_GB", "").strip()


@dataclass
class ProfileArtifacts:
    cpu_profile_path: Path
    stats_text_path: Path
    peak_memory_bytes: int
    instrumentation_text_path: Path


@dataclass
class Probe:
    derivative_calls: int = 0
    transpose_calls: int = 0
    by_dim_counter: Counter | None = None
    by_component_counter: Counter | None = None

    get_adjoint_data_calls: int = 0
    get_adjoint_data_output_bytes: int = 0

    scale_calls: int = 0
    scale_input_bytes: int = 0
    scale_output_bytes: int = 0

    slice_calls: int = 0
    slice_input_bytes: int = 0
    slice_output_bytes: int = 0

    derivative_info_calls: int = 0
    derivative_info_input_bytes: int = 0

    compute_derivatives_calls: int = 0
    compute_derivatives_input_bytes: int = 0

    resolve_chunk_calls: int = 0
    resolve_chunk_inputs: list[dict] | None = None

    checkpoint_stats: dict | None = None

    def __post_init__(self) -> None:
        self.by_dim_counter = Counter()
        self.by_component_counter = Counter()
        self.resolve_chunk_inputs = []
        self.checkpoint_stats = {}


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.3f} {unit}"
        amount /= 1024.0
    return f"{amount:.3f} TiB"


def _estimate_nbytes(obj) -> int:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        return sum(_estimate_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_nbytes(v) for v in obj)

    field_components = getattr(obj, "field_components", None)
    if isinstance(field_components, dict):
        return sum(_estimate_nbytes(v) for v in field_components.values())

    values = getattr(obj, "values", None)
    if values is not None:
        try:
            return int(np.asarray(values).nbytes)
        except Exception:
            return 0
    return 0


def _record_checkpoint(probe: Probe, label: str, estimated_live_bytes: int = 0) -> None:
    try:
        import tracemalloc
    except ImportError:
        return
    if not tracemalloc.is_tracing():
        return

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    stats = probe.checkpoint_stats.setdefault(
        label,
        {
            "count": 0,
            "max_current_bytes": 0,
            "max_peak_bytes": 0,
            "max_estimated_live_bytes": 0,
        },
    )
    stats["count"] += 1
    stats["max_current_bytes"] = max(stats["max_current_bytes"], int(current_bytes))
    stats["max_peak_bytes"] = max(stats["max_peak_bytes"], int(peak_bytes))
    stats["max_estimated_live_bytes"] = max(
        stats["max_estimated_live_bytes"], int(estimated_live_bytes)
    )


def _infer_num_freqs(inputs: PostprocessAdjInputs) -> int:
    fld_adj = inputs.sim_data_adj._get_adjoint_data(0, data_type="fld")
    first = next(iter(fld_adj.field_components.values()))
    return len(first.coords["f"])


def _chunk_override(mode: str, n_freqs: int) -> int | None:
    if mode == "all":
        return n_freqs
    if mode == "half":
        return max(1, n_freqs // 2)
    if mode == "2":
        return 2
    if mode == "1":
        return 1
    if mode == "none":
        return None
    raise ValueError(f"Unsupported TIDY3D_PROFILE_CHUNK_MODE='{mode}'.")


@contextmanager
def _instrument_pipeline():
    probe = Probe()

    original_derivative_fn = AbstractCustomMedium._derivative_field_cmp_custom
    original_transpose_utils = derivative_utils_module.transpose_interp_axis
    original_transpose_medium = medium_module.transpose_interp_axis
    original_scale_fn = backward_module.scale_field_data
    original_slice_fn = backward_module._slice_field_data
    original_get_adj_fn = sim_data_module.SimulationData._get_adjoint_data
    original_derivative_info_ctor = backward_module.DerivativeInfo
    original_resolve_chunk_fn = backward_module._resolve_freq_chunk_size
    original_compute_derivatives_fn = CustomMedium._compute_derivatives

    def _wrapped_derivative(
        self,
        E_der_map,
        spatial_data,
        dim,
        bounds=None,
        component="real",
        interp_method=None,
        sum_over_freqs=True,
    ):
        probe.derivative_calls += 1
        probe.by_dim_counter[dim] += 1
        probe.by_component_counter[component] += 1
        _record_checkpoint(probe, "custom_derivative:start")
        out = original_derivative_fn(
            self=self,
            E_der_map=E_der_map,
            spatial_data=spatial_data,
            dim=dim,
            bounds=bounds,
            component=component,
            interp_method=interp_method,
            sum_over_freqs=sum_over_freqs,
        )
        _record_checkpoint(
            probe, "custom_derivative:end", estimated_live_bytes=_estimate_nbytes(out)
        )
        return out

    def _wrapped_transpose(
        field_values,
        field_coords_1d,
        param_coords_1d,
        *,
        method="linear",
        coordinate_tolerance=derivative_utils_module.AUTOGRAD_COORDINATE_TOLERANCE,
    ):
        probe.transpose_calls += 1
        _record_checkpoint(
            probe, "transpose:start", estimated_live_bytes=_estimate_nbytes(field_values)
        )
        out = original_transpose_utils(
            field_values=field_values,
            field_coords_1d=np.asarray(field_coords_1d),
            param_coords_1d=np.asarray(param_coords_1d),
            method=method,
            coordinate_tolerance=coordinate_tolerance,
        )
        _record_checkpoint(probe, "transpose:end", estimated_live_bytes=_estimate_nbytes(out))
        return out

    def _wrapped_get_adjoint_data(self, structure_index, data_type):
        _record_checkpoint(probe, "_get_adjoint_data:start")
        probe.get_adjoint_data_calls += 1
        out = original_get_adj_fn(self, structure_index, data_type)
        out_bytes = _estimate_nbytes(out)
        probe.get_adjoint_data_output_bytes += out_bytes
        _record_checkpoint(probe, "_get_adjoint_data:end", estimated_live_bytes=out_bytes)
        return out

    def _wrapped_scale_field_data(fld_data, scale):
        in_bytes = _estimate_nbytes(fld_data)
        probe.scale_calls += 1
        probe.scale_input_bytes += in_bytes
        _record_checkpoint(probe, "scale_field_data:start", estimated_live_bytes=in_bytes)
        out = original_scale_fn(fld_data=fld_data, scale=scale)
        out_bytes = _estimate_nbytes(out)
        probe.scale_output_bytes += out_bytes
        _record_checkpoint(probe, "scale_field_data:end", estimated_live_bytes=out_bytes)
        return out

    def _wrapped_slice_field_data(field_data, freq_indices, component_indicator=None):
        in_bytes = _estimate_nbytes(field_data)
        probe.slice_calls += 1
        probe.slice_input_bytes += in_bytes
        _record_checkpoint(probe, "slice_field_data:start", estimated_live_bytes=in_bytes)
        out = original_slice_fn(field_data, freq_indices, component_indicator)
        out_bytes = _estimate_nbytes(out)
        probe.slice_output_bytes += out_bytes
        _record_checkpoint(probe, "slice_field_data:end", estimated_live_bytes=out_bytes)
        return out

    def _wrapped_derivative_info_ctor(*args, **kwargs):
        in_bytes = (
            _estimate_nbytes(kwargs.get("E_der_map"))
            + _estimate_nbytes(kwargs.get("D_der_map"))
            + _estimate_nbytes(kwargs.get("E_fwd"))
            + _estimate_nbytes(kwargs.get("E_adj"))
            + _estimate_nbytes(kwargs.get("D_fwd"))
            + _estimate_nbytes(kwargs.get("D_adj"))
            + _estimate_nbytes(kwargs.get("eps_data"))
        )
        probe.derivative_info_calls += 1
        probe.derivative_info_input_bytes += in_bytes
        _record_checkpoint(probe, "DerivativeInfo:create:start", estimated_live_bytes=in_bytes)
        out = original_derivative_info_ctor(*args, **kwargs)
        _record_checkpoint(probe, "DerivativeInfo:create:end", estimated_live_bytes=in_bytes)
        return out

    def _wrapped_compute_derivatives(self, derivative_info, *args, **kwargs):
        in_bytes = (
            _estimate_nbytes(getattr(derivative_info, "E_der_map", None))
            + _estimate_nbytes(getattr(derivative_info, "D_der_map", None))
            + _estimate_nbytes(getattr(derivative_info, "E_fwd", None))
            + _estimate_nbytes(getattr(derivative_info, "E_adj", None))
            + _estimate_nbytes(getattr(derivative_info, "D_fwd", None))
            + _estimate_nbytes(getattr(derivative_info, "D_adj", None))
            + _estimate_nbytes(getattr(derivative_info, "eps_data", None))
        )
        probe.compute_derivatives_calls += 1
        probe.compute_derivatives_input_bytes += in_bytes
        _record_checkpoint(probe, "compute_derivatives:start", estimated_live_bytes=in_bytes)
        out = original_compute_derivatives_fn(self, derivative_info, *args, **kwargs)
        _record_checkpoint(probe, "compute_derivatives:end", estimated_live_bytes=in_bytes)
        return out

    def _wrapped_resolve_freq_chunk_size(n_freqs, max_freqs_from_budget, fallback_num_freqs=1):
        available_bytes = backward_module.system_utils.get_available_memory_bytes()
        in_bytes = (
            int(max_freqs_from_budget(available_bytes))
            if available_bytes > 0
            else int(fallback_num_freqs)
        )
        in_bytes = max(in_bytes, 0)
        probe.resolve_chunk_calls += 1
        _record_checkpoint(probe, "resolve_chunk:start", estimated_live_bytes=in_bytes)
        out = original_resolve_chunk_fn(
            n_freqs=n_freqs,
            max_freqs_from_budget=max_freqs_from_budget,
            fallback_num_freqs=fallback_num_freqs,
        )
        probe.resolve_chunk_inputs.append(
            {
                "n_freqs": int(n_freqs),
                "configured_chunk_size": fallback_num_freqs,
                "resolved_chunk_size": int(out),
                "dataset_bytes": int(in_bytes),
            }
        )
        _record_checkpoint(probe, "resolve_chunk:end", estimated_live_bytes=in_bytes)
        return out

    derivative_utils_module.transpose_interp_axis = _wrapped_transpose
    medium_module.transpose_interp_axis = _wrapped_transpose
    AbstractCustomMedium._derivative_field_cmp_custom = _wrapped_derivative
    sim_data_module.SimulationData._get_adjoint_data = _wrapped_get_adjoint_data
    backward_module.scale_field_data = _wrapped_scale_field_data
    backward_module._slice_field_data = _wrapped_slice_field_data
    backward_module.DerivativeInfo = _wrapped_derivative_info_ctor
    backward_module._resolve_freq_chunk_size = _wrapped_resolve_freq_chunk_size
    CustomMedium._compute_derivatives = _wrapped_compute_derivatives

    try:
        yield probe
    finally:
        derivative_utils_module.transpose_interp_axis = original_transpose_utils
        medium_module.transpose_interp_axis = original_transpose_medium
        AbstractCustomMedium._derivative_field_cmp_custom = original_derivative_fn
        sim_data_module.SimulationData._get_adjoint_data = original_get_adj_fn
        backward_module.scale_field_data = original_scale_fn
        backward_module._slice_field_data = original_slice_fn
        backward_module.DerivativeInfo = original_derivative_info_ctor
        backward_module._resolve_freq_chunk_size = original_resolve_chunk_fn
        CustomMedium._compute_derivatives = original_compute_derivatives_fn


def _write_instrumentation_report(
    report_path: Path, probe: Probe, peak_memory_bytes: int, mode: str
) -> None:
    rows = sorted(
        probe.checkpoint_stats.items(),
        key=lambda kv: kv[1]["max_peak_bytes"],
        reverse=True,
    )

    lines = [
        "CustomMedium postprocess_adj memory instrumentation",
        "===============================================",
        "",
        f"chunk mode: {mode}",
        f"postprocess_adj tracemalloc peak: {_format_bytes(peak_memory_bytes)}",
        "",
        "Chunk resolution",
        "---------------",
        f"resolve_chunk calls: {probe.resolve_chunk_calls}",
    ]
    if probe.resolve_chunk_inputs:
        for idx, data in enumerate(probe.resolve_chunk_inputs, start=1):
            lines.append(
                f"resolve[{idx}]: n_freqs={data['n_freqs']}, "
                f"configured={data['configured_chunk_size']}, "
                f"resolved={data['resolved_chunk_size']}, "
                f"dataset_bytes={_format_bytes(data['dataset_bytes'])}"
            )
    else:
        lines.append("No chunk-resolution data collected.")

    lines.extend(
        [
            "",
            "Call counts",
            "-----------",
            f"_get_adjoint_data calls: {probe.get_adjoint_data_calls}",
            f"_derivative_field_cmp_custom calls: {probe.derivative_calls}",
            f"transpose_interp_axis calls: {probe.transpose_calls}",
            f"_slice_field_data calls: {probe.slice_calls}",
            f"DerivativeInfo constructor calls: {probe.derivative_info_calls}",
            f"CustomMedium._compute_derivatives calls: {probe.compute_derivatives_calls}",
            f"by dim: {dict(probe.by_dim_counter)}",
            f"by component: {dict(probe.by_component_counter)}",
            "",
            "Estimated bytes by stage (accumulated)",
            "--------------------------------------",
            f"_get_adjoint_data output bytes: {_format_bytes(probe.get_adjoint_data_output_bytes)}",
            f"scale_field_data input bytes: {_format_bytes(probe.scale_input_bytes)}",
            f"scale_field_data output bytes: {_format_bytes(probe.scale_output_bytes)}",
            f"_slice_field_data input bytes: {_format_bytes(probe.slice_input_bytes)}",
            f"_slice_field_data output bytes: {_format_bytes(probe.slice_output_bytes)}",
            f"DerivativeInfo input bytes: {_format_bytes(probe.derivative_info_input_bytes)}",
            f"CustomMedium._compute_derivatives input bytes: {_format_bytes(probe.compute_derivatives_input_bytes)}",
            "",
            "Peak timeline checkpoints (max over occurrences)",
            "---------------------------------------------",
        ]
    )

    for label, stats in rows:
        lines.append(
            f"{label}: count={stats['count']}, "
            f"max_current={_format_bytes(stats['max_current_bytes'])}, "
            f"max_peak={_format_bytes(stats['max_peak_bytes'])}, "
            f"max_estimated_live={_format_bytes(stats['max_estimated_live_bytes'])}"
        )

    lines.extend(
        [
            "",
            "Notes",
            "-----",
            "- Byte totals are accumulated over calls, not simultaneous live memory.",
            "- max_peak reflects tracemalloc high-water mark captured at each checkpoint.",
            "- Compare reports across chunk modes (all/half/2/1) to see where peak shifts.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf8")


def _run_postprocess_adj(inputs: PostprocessAdjInputs) -> dict:
    return postprocess_adj(
        sim_data_adj=inputs.sim_data_adj,
        sim_data_orig=inputs.sim_data_orig,
        sim_data_fwd=inputs.sim_data_fwd,
        sim_fields_keys=inputs.sim_fields_keys,
    )


def _load_inputs_or_skip() -> PostprocessAdjInputs:
    if not dataset_exists(CUSTOM_MEDIUM_DATASET_ROOT):
        pytest.skip(
            f"Persisted CustomMedium postprocess_adj dataset missing at {CUSTOM_MEDIUM_DATASET_ROOT}. "
            f"Run with @{MARK_GENERATE_NAME} first."
        )
    return load_postprocess_adj_inputs(CUSTOM_MEDIUM_DATASET_ROOT)


def _profile_callable(func: Callable[[], dict], output_dir: Path, mode: str) -> ProfileArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tracemalloc
    except ImportError:
        tracemalloc = None

    if tracemalloc:
        tracemalloc.start()

    with _instrument_pipeline() as probe:
        _record_checkpoint(probe, "profile:before_postprocess")
        profile = cProfile.Profile()
        profile.enable()
        result = func()
        profile.disable()
        _record_checkpoint(probe, "profile:after_postprocess")

    cpu_profile_path = output_dir / f"postprocess_adj_custom_medium_{mode}.prof"
    profile.dump_stats(cpu_profile_path)

    peak_memory_bytes = -1
    if tracemalloc:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    stats_text_path = output_dir / f"postprocess_adj_custom_medium_profile_{mode}.txt"
    with stats_text_path.open("w", encoding="utf8") as stats_file:
        stats = pstats.Stats(profile, stream=stats_file)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        stats_file.write(f"\nResult keys: {sorted(result.keys())}\n")
        stats_file.write(f"\nPeak memory GB: {peak_memory_bytes / (1024.0 * 1024.0 * 1024.0)}\n")

    instrumentation_text_path = (
        output_dir / f"postprocess_adj_custom_medium_memory_breakdown_{mode}.txt"
    )
    _write_instrumentation_report(instrumentation_text_path, probe, peak_memory_bytes, mode)

    return ProfileArtifacts(
        cpu_profile_path=cpu_profile_path,
        stats_text_path=stats_text_path,
        peak_memory_bytes=peak_memory_bytes,
        instrumentation_text_path=instrumentation_text_path,
    )


@pytest.mark.generate_profile
def test_generate_custom_medium_postprocess_adj_dataset(tmp_path: Path, redirect_stdout_to_stderr):
    """Generate cached inputs for CustomMedium postprocess_adj profiling."""
    print(f"Profile data in {tmp_path}")
    print(
        "CustomMedium parameters: "
        f"box_size={CUSTOM_MEDIUM_BOX_SIZE}, "
        f"pixel_size={CUSTOM_MEDIUM_PIXEL_SIZE}, "
        f"num_freqs={CUSTOM_MEDIUM_NUM_FREQS}"
    )

    inputs = generate_custom_medium_postprocess_adj_inputs(
        output_dir=tmp_path,
        box_size=CUSTOM_MEDIUM_BOX_SIZE,
        pixel_size=CUSTOM_MEDIUM_PIXEL_SIZE,
        num_freqs=CUSTOM_MEDIUM_NUM_FREQS,
        permittivity=CUSTOM_MEDIUM_PERMITTIVITY,
        wavelength=CUSTOM_MEDIUM_WAVELENGTH,
    )

    persist_postprocess_adj_inputs(inputs, CUSTOM_MEDIUM_DATASET_ROOT)
    assert dataset_exists(CUSTOM_MEDIUM_DATASET_ROOT)


@pytest.mark.performance_profile
def test_custom_medium_postprocess_adj_profile(
    tmp_path: Path, redirect_stdout_to_stderr, monkeypatch
):
    """Profile CustomMedium ``postprocess_adj`` using cached simulation data."""
    print(f"Profile data in {tmp_path}")

    inputs = _load_inputs_or_skip()
    n_freqs = _infer_num_freqs(inputs)
    configured_chunk_size = _chunk_override(PROFILE_CHUNK_MODE, n_freqs)

    if PROFILE_AVAILABLE_MEMORY_GB:
        available_memory_bytes = int(float(PROFILE_AVAILABLE_MEMORY_GB) * 1024**3)
        monkeypatch.setattr(
            backward_module.system_utils,
            "get_available_memory_bytes",
            lambda: available_memory_bytes,
        )

    original_chunk_size = td.config.adjoint.solver_freq_chunk_size
    td.config.adjoint.solver_freq_chunk_size = configured_chunk_size
    try:
        mode_label = PROFILE_CHUNK_MODE
        artifacts = _profile_callable(
            lambda: _run_postprocess_adj(inputs),
            tmp_path / "profile",
            mode=mode_label,
        )
    finally:
        td.config.adjoint.solver_freq_chunk_size = original_chunk_size

    assert artifacts.cpu_profile_path.exists()
    assert artifacts.stats_text_path.exists()
    assert artifacts.instrumentation_text_path.exists()
