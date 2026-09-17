from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

import tidy3d as td
import tidy3d.system as system_utils
from tidy3d.components.autograd import get_static
from tidy3d.components.autograd.derivative_utils import (
    TRIANGLE_MESH_SURFACE_GRADIENT_MAX_BATCH_SIZE,
    DerivativeInfo,
    triangle_mesh_surface_gradient_batch_size,
)
from tidy3d.components.autograd.monitor_names import adjoint_monitor_name
from tidy3d.components.autograd.point_consumption import (
    adjoint_point_cloud_chunk_bounds,
    adjoint_point_cloud_chunk_tag,
    compute_point_integrands,
)
from tidy3d.components.autograd.utils import accumulate_field_map as _accumulate_field_map
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.geometry.bound_ops import bounds_contains, bounds_intersection
from tidy3d.components.source.adjoint_helpers import (
    collapse_source_adjoint_to_dataset_frequency,
)
from tidy3d.components.source.field import AbstractGaussianBeam
from tidy3d.config import config
from tidy3d.em.translate.sample_sets import (
    SamplingContext,
    shape_paths_by_structure,
    validate_sample_sets_coverage,
)
from tidy3d.exceptions import AdjointError
from tidy3d.log import log
from tidy3d.packaging import disable_local_subpixel

from .constants import SAMPLE_SETS_FILE
from .flux_monitor import expand_flux_monitor_vjps
from .utils import (
    E_to_D,
    custom_vjp_geometry_exclusions,
    expand_custom_vjp_configs,
    filter_vjp_map,
    get_derivative_maps,
    scale_field_data,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.spacing import SamplingResolution
    from tidy3d.components.data.data_array import ScalarFieldDataArray
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.monitor import Monitor
    from tidy3d.em.translate.sample_sets import GeometrySampleSets, StructureSampleSets

    from .context import AdjointPostprocessInputs
    from .types import DerivativeView, NumericalStructureConfig


# Scaling factor for chunk-dependent memory growth on top of the baseline estimate.
ADJOINT_MEMORY_MULTIPLIER = 6.0
# Baseline factor for the always-live forward/adjoint field and permittivity datasets.
ADJOINT_MEMORY_BASELINE_MULTIPLIER = 2.0


@dataclass(frozen=True)
class AdjointSetupResult:
    """Result of setting up adjoint simulations for a VJP map."""

    simulations: list[td.Simulation]
    all_sources_underflowed: bool = False


def make_adjoint_monitors(
    simulation: td.Simulation,
    sim_fields_keys: list[tuple],
    sample_sets: GeometrySampleSets | None = None,
) -> list[Monitor]:
    """Return the structure adjoint monitors used by adjoint simulations.

    Point-cloud permittivity monitors are forward-only (permittivity is identical in
    the adjoint simulation), so they are dropped here.
    """
    monitors_fld, monitors_eps = simulation._make_adjoint_monitors(
        sim_fields_keys, sample_sets=sample_sets
    )
    monitors_eps = [
        monitor
        for monitor in monitors_eps
        if not isinstance(monitor, td.PointCloudPermittivityMonitor)
    ]
    return [*monitors_fld, *monitors_eps]


def _resolve_freq_chunk_size(
    n_freqs: int, max_freqs_from_budget: Callable[[int], int], fallback_num_freqs: int = 1
) -> int:
    """Resolve frequency chunk size from available memory and a budget callback."""
    if n_freqs < 1:
        return 0
    fallback_chunk = min(n_freqs, fallback_num_freqs)

    available_bytes = system_utils.get_available_memory_bytes()
    if available_bytes < 0:
        return fallback_chunk

    max_freqs_by_budget = max_freqs_from_budget(available_bytes)
    if max_freqs_by_budget < 1:
        td.log.warning(
            "Adjoint postprocessing may exceed the available memory budget even at one "
            "frequency per chunk; forcing chunk size to 1.",
            log_once=True,
        )
    return max(1, min(n_freqs, max_freqs_by_budget))


def _resolve_triangle_mesh_surface_gradient_batch_size(num_freqs: int) -> int:
    """Resolve the runtime TriangleMesh batch size from the host memory budget."""
    available_bytes = system_utils.get_available_memory_bytes()
    memory_budget_bytes = (
        int(available_bytes * config.adjoint.memory_allotment_fraction)
        if available_bytes > 0
        else None
    )
    return triangle_mesh_surface_gradient_batch_size(
        num_samples=TRIANGLE_MESH_SURFACE_GRADIENT_MAX_BATCH_SIZE,
        num_freqs=num_freqs,
        memory_budget_bytes=memory_budget_bytes,
    )


def _sampling_resolution_fields(
    sampling_resolution: SamplingResolution | None, num_freqs: int
) -> dict[str, float | int | None]:
    """Return resolved ``DerivativeInfo`` sampling fields from optional precomputed data."""

    if sampling_resolution is None:
        return {
            "resolved_adaptive_vjp_spacing": None,
            "resolved_material_length_scale": None,
            "resolved_material_wavelength": None,
            "resolved_surface_gradient_batch_size": _resolve_triangle_mesh_surface_gradient_batch_size(
                num_freqs
            ),
        }
    return {
        "resolved_adaptive_vjp_spacing": sampling_resolution.spacing,
        "resolved_material_length_scale": sampling_resolution.material_length_scale,
        "resolved_material_wavelength": sampling_resolution.material_wavelength,
        "resolved_surface_gradient_batch_size": _resolve_triangle_mesh_surface_gradient_batch_size(
            num_freqs
        ),
    }


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_keys: list[tuple],
    max_num_adjoint_per_fwd: int,
    already_filtered: bool = False,
    sim_data_fwd: td.SimulationData | None = None,
    return_result: bool = False,
    sample_sets: GeometrySampleSets | None = None,
) -> list[td.Simulation] | AdjointSetupResult:
    """Construct an adjoint simulation from a set of data_fields for the VJP."""

    td.log.info("Running custom vjp (adjoint) pipeline.")

    if not already_filtered:
        data_fields_vjp = filter_vjp_map(data_fields_vjp)

    # if all entries are zero, there is no adjoint sim to run
    if not data_fields_vjp:
        result = AdjointSetupResult([])
        return result if return_result else result.simulations

    data_fields_vjp, sim_data_for_adj = expand_flux_monitor_vjps(
        data_fields_vjp=data_fields_vjp,
        sim_data_orig=sim_data_orig,
        sim_data_fwd=sim_data_fwd,
    )

    # start with the full simulation data structure and either zero out the fields
    # that have no tracer data for them or insert the tracer data
    full_sim_data_dict = sim_data_for_adj._strip_traced_fields(
        include_untraced_data_arrays=True, starting_paths=(("data",),)
    )
    for path, value in full_sim_data_dict.items():
        if path in data_fields_vjp:
            full_sim_data_dict[path] = data_fields_vjp[path]
        else:
            full_sim_data_dict[path] = 0 * value

    # insert the raw VJP data into the .data of the original SimulationData
    sim_data_vjp = sim_data_for_adj._insert_traced_fields(field_mapping=full_sim_data_dict)

    # make adjoint simulation from that SimulationData
    data_vjp_paths = set(data_fields_vjp.keys())

    adjoint_monitors = make_adjoint_monitors(
        sim_data_orig.simulation, sim_fields_keys, sample_sets=sample_sets
    )

    adjoint_setup_result = sim_data_vjp._make_adjoint_sims_with_result(
        data_vjp_paths=data_vjp_paths,
        adjoint_monitors=adjoint_monitors,
    )
    sims_adj = adjoint_setup_result.simulations

    if len(sims_adj) > max_num_adjoint_per_fwd:
        raise AdjointError(
            f"Number of adjoint simulations ({len(sims_adj)}) exceeds the maximum allowed "
            f"({max_num_adjoint_per_fwd}) per forward simulation. This typically means that "
            "there are many frequencies and monitors in the simulation that are being differentiated "
            "w.r.t. in the objective function. To proceed, please double-check the simulation "
            "setup, increase the 'max_num_adjoint_per_fwd' parameter in the run function, and re-run."
        )

    result = AdjointSetupResult(
        sims_adj,
        all_sources_underflowed=adjoint_setup_result.all_sources_underflowed,
    )
    return result if return_result else result.simulations


def _slice_field_data(
    field_data: dict, freq_indices: slice, component_indicator: str | None = None
) -> dict:
    """
    Slice field data dictionary along frequency dimension using `isel`
    and freq_indices.
    """
    sliced_data = {}

    # filter keys first to avoid unnecessary looping
    keys_to_process = (
        k for k in field_data.keys() if component_indicator is None or component_indicator in k
    )

    num_freqs = next(iter(field_data.values())).sizes["f"]

    start = freq_indices.start
    stop = freq_indices.stop
    if (start < 0) or (start >= num_freqs):
        raise IndexError(f"Frequency slice ({start}, {stop}) is out of bounds for size {num_freqs}")

    for k in keys_to_process:
        sliced_data[k] = field_data[k].isel(f=freq_indices)

    return sliced_data


def _get_freq_coords(field_data: td.FieldData) -> np.ndarray:
    """Extract frequency coordinates from a field dataset."""
    first_field_component = next(iter(field_data.field_components.values()))
    return np.array(first_field_component.coords["f"].values)


def _estimate_dataset_bytes(dataset: td.PermittivityData | td.FieldData) -> int:
    """Estimate total byte size of field components in a dataset."""
    return sum(np.asarray(comp.values).nbytes for comp in dataset.field_components.values())


def _adjoint_freq_chunk_size(n_freqs: int, combined_data_size: int) -> int:
    """Resolve the adjoint frequency chunk size from config and the memory budget.

    Shared by the volumetric, mixed, and point-only consumption paths so every
    family of adjoint data participates in the same frequency memory budget.
    """

    def estimate_peak_bytes(num_chunk_freqs: int) -> int:
        return int(
            ADJOINT_MEMORY_BASELINE_MULTIPLIER * combined_data_size
            + (num_chunk_freqs / n_freqs) * combined_data_size * ADJOINT_MEMORY_MULTIPLIER
        )

    user_desired_freqs = config.adjoint.solver_freq_chunk_size
    if user_desired_freqs is not None and user_desired_freqs > 0:
        freq_chunk_size = min(n_freqs, user_desired_freqs)
        available_bytes = system_utils.get_available_memory_bytes()
        if available_bytes > 0:
            budget_bytes = int(available_bytes * config.adjoint.memory_allotment_fraction)
            if combined_data_size > 0 and estimate_peak_bytes(freq_chunk_size) > budget_bytes:
                td.log.warning(
                    "Configured adjoint frequency chunk size may exceed the available memory budget; "
                    "continuing with the configured chunk size.",
                    log_once=True,
                )
        return freq_chunk_size

    def max_freqs_from_budget(available_bytes: int) -> int:
        budget_bytes = int(available_bytes * config.adjoint.memory_allotment_fraction)
        numerator = budget_bytes - ADJOINT_MEMORY_BASELINE_MULTIPLIER * combined_data_size
        denominator = combined_data_size * ADJOINT_MEMORY_MULTIPLIER
        return int(n_freqs * numerator / denominator) if denominator > 0 else n_freqs

    return _resolve_freq_chunk_size(
        n_freqs=n_freqs, max_freqs_from_budget=max_freqs_from_budget, fallback_num_freqs=1
    )


def _point_cloud_monitor_names(
    structure_index: int, pec_staged: bool, num_chunks: int
) -> list[str]:
    """Candidate point-cloud monitor names for one structure's staged sample sets."""
    monitor_tags = ["fld_pc"]
    monitor_tags += [
        f"eps_pc_{side}_{component}" for side in ("in", "out") for component in ("xx", "yy", "zz")
    ]
    if pec_staged:
        monitor_tags += [
            f"fld_pc_{component}_{side}"
            for side in ("out", "in")
            for component in ("ex", "ey", "ez", "hx", "hy", "hz")
        ]
    return [
        adjoint_monitor_name(
            index=structure_index,
            monitor_tag=adjoint_point_cloud_chunk_tag(monitor_tag, chunk_index),
        )
        for monitor_tag in monitor_tags
        for chunk_index in range(num_chunks)
    ]


def _point_cloud_data_size(
    sim_data_fwd: td.SimulationData,
    sim_data_adj: td.SimulationData,
    structure_index: int,
    sample_sets: StructureSampleSets,
) -> int:
    """Total bytes of one structure's point-cloud monitor data (forward and adjoint)."""
    num_chunks = len(adjoint_point_cloud_chunk_bounds(sample_sets.num_points))
    names = _point_cloud_monitor_names(
        structure_index, pec_staged=sample_sets.pec_staged, num_chunks=num_chunks
    )
    total = 0
    for sim_data in (sim_data_fwd, sim_data_adj):
        monitor_names = {monitor.name for monitor in sim_data.simulation.monitors}
        for name in names:
            if name in monitor_names:
                total += _estimate_dataset_bytes(sim_data[name])
    return total


def _require_freq_ascending(
    dataset: td.PermittivityData | td.FieldData,
    *,
    component_type: str,
    component_index: int,
    dataset_name: str,
) -> None:
    """Validate that all field components are already in ascending frequency order."""
    for key, val in dataset.field_components.items():
        freqs = np.asarray(val.coords["f"].values)
        if len(freqs) <= 1:
            continue
        if np.any(freqs[1:] < freqs[:-1]):
            raise ValueError(
                "Adjoint postprocessing expects ascending frequency coordinates. "
                f"Got non-ascending frequencies in {dataset_name} for {component_type} "
                f"{component_index}, component '{key}': {freqs}."
            )


def _validate_adjoint_frequencies(
    *,
    adjoint_frequencies: np.ndarray,
    monitor_freqs: np.ndarray,
    component_type: str,
    component_index: int,
) -> None:
    """Validate that field-data frequencies match monitor frequencies exactly."""
    if len(adjoint_frequencies) != len(monitor_freqs) or not np.allclose(
        adjoint_frequencies, monitor_freqs, rtol=1e-10, atol=0
    ):
        raise ValueError(
            f"Frequency mismatch in adjoint postprocessing for {component_type} "
            f"{component_index}. Expected frequencies from monitor: {monitor_freqs}, "
            f"but derivative map has: {adjoint_frequencies}. "
        )


def _filter_frequency_data(
    dataset: td.PermittivityData | td.FieldData,
    filter_freqs: np.ndarray,
    *,
    component_type: str,
    component_index: int,
    dataset_name: str,
) -> td.PermittivityData | td.FieldData:
    """Filter a dataset to target frequencies and keep its monitor frequencies aligned."""
    dataset_filter_freq = {}
    for key, val in dataset.field_components.items():
        dataset_filter_freq[key] = val.sel(f=filter_freqs)

    filtered_monitor = dataset.monitor.updated_copy(freqs=list(filter_freqs))
    filtered_dataset = dataset.updated_copy(monitor=filtered_monitor, **dataset_filter_freq)

    _require_freq_ascending(
        filtered_dataset,
        component_type=component_type,
        component_index=component_index,
        dataset_name=dataset_name,
    )

    return filtered_dataset


def _to_sim_fields_vjp(
    *,
    component_type: str,
    component_index: int,
    component_vjp: AutogradFieldMap,
) -> AutogradFieldMap:
    """Map component-local derivative paths to simulation-level paths."""
    sim_fields_vjp = {}
    for component_path, vjp_value in component_vjp.items():
        sim_path = (component_type, component_index, *list(component_path))
        sim_fields_vjp[sim_path] = vjp_value
    return sim_fields_vjp


@disable_local_subpixel
def postprocess_adj(
    sim_data_adj: td.SimulationData,
    *,
    postprocess_inputs: AdjointPostprocessInputs,
) -> AutogradFieldMap:
    """Postprocess some data from the adjoint simulation into the VJP for the original sim flds."""
    sim_data_orig = postprocess_inputs.sim_data_orig
    sim_data_fwd = postprocess_inputs.sim_data_fwd
    sim_fields_keys = postprocess_inputs.sim_fields_keys
    numerical_structure_map = postprocess_inputs.numerical_structure_map
    custom_vjp = postprocess_inputs.custom_vjp

    custom_vjp_lookup = expand_custom_vjp_configs(custom_vjp, sim_fields_keys)
    sample_set_exclusions = custom_vjp_geometry_exclusions(custom_vjp_lookup)

    sample_sets_artifact = postprocess_inputs.sample_sets
    required_sample_paths = shape_paths_by_structure(sim_fields_keys, exclude=sample_set_exclusions)
    if sample_sets_artifact is None:
        if required_sample_paths:
            raise AdjointError(
                "Geometry shape derivatives require the pre-collected adjoint sample sets "
                f"artifact ('{SAMPLE_SETS_FILE}'), but none was provided. This artifact is "
                "uploaded with every autograd forward task by tidy3d versions that support it. "
                "Rerun the forward simulation with an up-to-date tidy3d client, or use "
                "'local_gradient=True'."
            )
    else:
        # partial coverage is an error, never a partial recompute; the boundary helper
        # translates schema exceptions so callers only ever see tidy3d exceptions
        validate_sample_sets_coverage(sample_sets_artifact, required_sample_paths)

    # group the paths by component type and index
    sim_vjp_map = defaultdict(list)
    for component_type, component_index, *component_path in sim_fields_keys:
        sim_vjp_map[(component_type, component_index)].append(tuple(component_path))
    numerical_structure_map = numerical_structure_map or {}

    structure_indices = {
        component_index
        for (component_type, component_index) in sim_vjp_map
        if component_type == "structures"
    }
    numerical_indices = {
        component_index
        for (component_type, component_index) in sim_vjp_map
        if component_type == "numerical"
    }
    overlap_indices = structure_indices & numerical_indices
    if overlap_indices:
        overlap_str = ", ".join(map(str, sorted(overlap_indices)))
        raise AdjointError(
            "Invalid autograd field mapping: structure index(es) "
            f"{overlap_str} have both 'structures' and 'numerical' traced paths. "
            "A structure index must be handled by exactly one VJP path."
        )

    # compute the VJP for each component
    sim_fields_vjp = {}
    for (component_type, component_index), component_paths in sim_vjp_map.items():
        if component_type == "structures":
            structure_sample_sets = (
                sample_sets_artifact.for_structure(component_index)
                if sample_sets_artifact is not None
                else None
            )
            sim_fields_vjp.update(
                _process_structure_gradients(
                    sim_data_adj,
                    sim_data_orig,
                    sim_data_fwd,
                    component_index,
                    component_paths,
                    sampling_resolution=postprocess_inputs.sampling_resolutions.get(
                        component_index
                    ),
                    custom_vjp=custom_vjp_lookup.get(component_index),
                    sample_sets=structure_sample_sets,
                )
            )
        elif component_type == "sources":
            sim_fields_vjp.update(
                _process_source_gradients(
                    sim_data_adj, sim_data_orig, sim_data_fwd, component_index, component_paths
                )
            )
        elif component_type == "numerical":
            numerical_structure = numerical_structure_map.get(component_index)
            if numerical_structure is None:
                raise AdjointError(
                    "No NumericalStructureConfig found for numerical structure index "
                    f"{component_index}. Available indices: {sorted(numerical_structure_map.keys())}."
                )
            sim_fields_vjp.update(
                _process_structure_gradients(
                    sim_data_adj,
                    sim_data_orig,
                    sim_data_fwd,
                    component_index,
                    structure_paths=[],
                    sampling_resolution=postprocess_inputs.sampling_resolutions.get(
                        component_index
                    ),
                    custom_vjp=None,
                    numerical_structure=numerical_structure,
                    numerical_paths=component_paths,
                )
            )
        else:
            raise ValueError(
                f"Unexpected component_type='{component_type}' for component_index={component_index}. "
                "Expected 'structures', 'sources', or 'numerical'."
            )

    return sim_fields_vjp


def _compute_source_time_scaling(
    source: td.Source,
    simulation: td.Simulation,
    frequencies: np.ndarray,
    source_dataset_freq: float,
) -> FreqDataArray:
    """Compute frequency-dependent source-time scale for source VJP processing."""

    freqs = np.asarray(frequencies, dtype=float)
    spectrum_freqs = np.full_like(freqs, source_dataset_freq)
    spectrum = source.source_time.spectrum(
        simulation.tmesh,
        spectrum_freqs,
        simulation.dt,
    )
    spectrum = np.asarray(spectrum, dtype=complex)

    # - 2.0: real-objective / one-sided-frequency adjoint convention.
    # - 2*pi (f -> omega): convert Hz-based quantities to angular-frequency form.
    # - 2*pi (domega = 2*pi*df): Fourier measure conversion for the current convention.
    # - c0: wavelength/frequency conversion (omega * lambda = 2*pi*c0).
    real_objective_factor = 2.0
    hz_to_omega_factor = 2.0 * np.pi
    fourier_measure_factor = 2.0 * np.pi
    wavelength_frequency_factor = td.C_0
    scale_prefactor = (
        real_objective_factor
        * hz_to_omega_factor
        * fourier_measure_factor
        * wavelength_frequency_factor
    )
    scale = scale_prefactor * spectrum * (source_dataset_freq / freqs)
    return FreqDataArray(scale, coords={"f": freqs})


def _get_source_dataset_frequency(source: td.Source) -> float:
    """Get source-dataset frequency for custom sources."""
    if isinstance(source, td.CustomFieldSource):
        dataset = source.field_dataset
    elif isinstance(source, td.CustomCurrentSource):
        dataset = source.current_dataset
    elif isinstance(source, AbstractGaussianBeam):
        # Gaussian-like source derivatives are remapped through an analytic mock
        # field dataset constructed at source_time._freq0 (single frequency), so
        # use the same reference frequency for source-time scaling.
        return float(source.source_time._freq0)
    else:
        raise TypeError(
            f"Source dataset frequency is only defined for custom sources, got '{source.type}'."
        )
    component = next(iter(dataset.field_components.values()))
    freqs = np.asarray(component.coords["f"].data, dtype=float).reshape(-1)
    return float(freqs[0])


def _warn_if_nonuniform_gaussian_source_background(
    simulation: td.Simulation,
    bounds_intersect: tuple[tuple[float, float, float], tuple[float, float, float]],
    source_freqs: np.ndarray,
) -> None:
    """Warn if Gaussian source bounds contain non-uniform epsilon at sampled frequencies."""
    lower, upper = bounds_intersect
    source_box = td.Box(
        center=tuple(0.5 * (lower[axis] + upper[axis]) for axis in range(3)),
        size=tuple(upper[axis] - lower[axis] for axis in range(3)),
    )
    for freq in source_freqs:
        eps_volume = np.asarray(
            simulation.epsilon(
                box=source_box,
                coord_key="centers",
                freq=float(freq),
            ).values
        ).reshape(-1)
        if eps_volume.size == 0:
            continue
        eps_min = np.min(eps_volume)
        eps_max = np.max(eps_volume)
        if not np.isclose(eps_min, eps_max):
            log.warning(
                "Gaussian-like source derivative remap assumes a uniform background index "
                "across the source extent, but epsilon is not uniform over the source bounds "
                f"intersection at f={float(freq):.6g} Hz (eps_min={eps_min}, eps_max={eps_max}; "
                "non-uniformity may also occur at other frequencies). "
                "Using center-sampled refractive index for gradient computation."
            )
            break


def _geometry_contains_clip_operation(geometry: GeometryType) -> bool:
    """Return True when ``geometry`` contains a ClipOperation at any depth."""
    if geometry is None:
        return False

    if isinstance(geometry, td.ClipOperation):
        return True

    if isinstance(geometry, td.Transformed):
        return _geometry_contains_clip_operation(geometry.geometry)

    if isinstance(geometry, td.GeometryArray):
        return _geometry_contains_clip_operation(geometry.geometry)

    if isinstance(geometry, td.GeometryGroup):
        return any(_geometry_contains_clip_operation(g) for g in geometry.geometries)

    return False


def _process_source_gradients(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    source_index: int,
    source_paths: list[tuple],
) -> AutogradFieldMap:
    """Process gradients for a specific source."""

    source = sim_data_fwd.simulation.sources[source_index]
    monitor_name = f"source_adjoint_{source_index}"

    fld_adj = sim_data_adj[monitor_name]
    fld_adj = fld_adj.grid_corrected_copy
    _require_freq_ascending(
        fld_adj, component_type="source", component_index=source_index, dataset_name="field data"
    )

    adjoint_frequencies = _get_freq_coords(fld_adj)
    monitor_freqs = np.array(fld_adj.monitor.freqs)
    _validate_adjoint_frequencies(
        adjoint_frequencies=adjoint_frequencies,
        monitor_freqs=monitor_freqs,
        component_type="source",
        component_index=source_index,
    )

    source_dataset_freq = _get_source_dataset_frequency(source)
    source_time_scaling = _compute_source_time_scaling(
        source=source,
        simulation=sim_data_orig.simulation,
        frequencies=adjoint_frequencies,
        source_dataset_freq=source_dataset_freq,
    )

    # Apply both adjoint post-normalization and source-time scaling in one pass.
    combined_scale = sim_data_adj.simulation.post_norm * source_time_scaling
    fld_adj = scale_field_data(fld_adj, combined_scale)
    if not isinstance(source, AbstractGaussianBeam):
        fld_adj = collapse_source_adjoint_to_dataset_frequency(
            fld_adj, float(np.asarray(source_dataset_freq).reshape(-1)[0])
        )

    e_adj = {k: v for k, v in fld_adj.field_components.items() if k.startswith("E")}
    h_adj = {k: v for k, v in fld_adj.field_components.items() if k.startswith("H")}

    bounds = source.geometry.bounds
    bounds_intersect = bounds_intersection(sim_data_orig.simulation.bounds, bounds)

    center = tuple(
        0.5 * (bounds_intersect[0][axis] + bounds_intersect[1][axis]) for axis in range(3)
    )
    point_box = td.Box(center=center, size=(0.0, 0.0, 0.0))
    source_freqs = np.asarray(next(iter(e_adj.values())).coords["f"].data).reshape(-1)
    if isinstance(source, AbstractGaussianBeam):
        _warn_if_nonuniform_gaussian_source_background(
            simulation=sim_data_orig.simulation,
            bounds_intersect=bounds_intersect,
            source_freqs=source_freqs,
        )
    eps_samples = [
        np.asarray(
            sim_data_orig.simulation.epsilon(
                box=point_box,
                coord_key="centers",
                freq=float(freq),
            ).values
        ).reshape(-1)[0]
        for freq in source_freqs
    ]
    source_background_index = FreqDataArray(
        np.sqrt(np.asarray(eps_samples, dtype=complex)),
        coords={"f": source_freqs},
    )

    # Source VJP currently does not use permittivity data.
    derivative_info = DerivativeInfo(
        paths=source_paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        E_adj=e_adj,
        D_fwd={},
        D_adj={},
        H_fwd={},
        H_adj=h_adj,
        eps_data={},
        source_background_index=source_background_index,
        frequencies=_get_freq_coords(fld_adj),
        bounds=bounds,
        bounds_intersect=bounds_intersect,
        simulation_bounds=sim_data_orig.simulation.bounds,
        updated_epsilon=lambda _replacement_geometry: None,
    )

    source_vjp = source._compute_derivatives(derivative_info)

    return _to_sim_fields_vjp(
        component_type="sources",
        component_index=source_index,
        component_vjp=source_vjp,
    )


def _process_structure_gradients(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    structure_index: int,
    structure_paths: list[tuple],
    sampling_resolution: SamplingResolution | None,
    custom_vjp: dict[tuple[str, str], Callable[..., Any]] | None = None,
    numerical_structure: NumericalStructureConfig | None = None,
    numerical_paths: list[tuple] | None = None,
    sample_sets: StructureSampleSets | None = None,
) -> AutogradFieldMap:
    """Process gradients for a specific structure.

    ``sample_sets`` is the structure's pre-collected ``StructureSampleSets``, shared
    across all frequency chunks (converted to its canonical-keyed mapping only at the
    geometry boundary); when absent, geometries generate late.
    """

    structure_paths = structure_paths or []
    numerical_paths = numerical_paths or []
    use_numerical_vjp = numerical_structure is not None and bool(numerical_paths)
    numerical_value_map: dict[tuple, Any] = {}
    numerical_vjp_fn = None
    numerical_params_static = None
    numerical_paths_ordered: tuple[tuple, ...] = tuple(numerical_paths)

    if use_numerical_vjp:
        numerical_vjp_fn = numerical_structure.compute_derivatives
        numerical_params_static = np.asarray(
            [get_static(param) for param in numerical_structure.parameters]
        )

    # point-only structures: when the artifact staged monitor payload and the
    # selection matrix placed no volumetric monitors (shape-only paths), gradients
    # come entirely from the point-cloud monitor data — no volumetric fetch, and the
    # same adjoint frequency memory budget chunks the point-cloud consumption
    staged_point_path = bool(structure_paths) and sample_sets is not None and sample_sets.staged
    fwd_monitor_names = {monitor.name for monitor in sim_data_fwd.simulation.monitors}
    volumetric_available = (
        adjoint_monitor_name(index=structure_index, monitor_tag="fld") in fwd_monitor_names
    )
    if not volumetric_available:
        if not staged_point_path or use_numerical_vjp or custom_vjp:
            raise AdjointError(
                "Rerun the forward simulation to regenerate its adjoint data: no adjoint "
                f"monitor data was recorded for structure {structure_index} (neither "
                "volumetric nor point-cloud) for its traced derivative paths."
            )
        structure = sim_data_fwd.simulation.structures[structure_index]
        clipped_geometry = (
            structure.geometry if _geometry_contains_clip_operation(structure.geometry) else None
        )
        fld_adj_pc = sim_data_adj[adjoint_monitor_name(index=structure_index, monitor_tag="fld_pc")]
        adjoint_frequencies = np.array(
            next(iter(fld_adj_pc.field_components.values())).coords["f"].values
        )
        n_freqs = len(adjoint_frequencies)
        freq_chunk_size = _adjoint_freq_chunk_size(
            n_freqs=n_freqs,
            combined_data_size=_point_cloud_data_size(
                sim_data_fwd, sim_data_adj, structure_index, sample_sets
            ),
        )
        struct_bounds = structure.geometry.bounds
        vjp_value_map: dict[tuple, Any] = {}
        for chunk_start in range(0, n_freqs, freq_chunk_size):
            select_adjoint_freqs = adjoint_frequencies[
                chunk_start : min(chunk_start + freq_chunk_size, n_freqs)
            ]
            point_integrands_chunk = compute_point_integrands(
                sample_sets=sample_sets,
                structure_index=structure_index,
                sim_data_fwd=sim_data_fwd,
                sim_data_adj=sim_data_adj,
                frequencies=select_adjoint_freqs,
                adjoint_post_norm=sim_data_adj.simulation.post_norm,
                is_medium_pec=structure.medium.is_pec,
                background_medium_is_pec=bool(
                    structure.background_medium and structure.background_medium.is_pec
                ),
                clipped_geometry=clipped_geometry,
                material_length_scale=(
                    sampling_resolution.material_length_scale
                    if sampling_resolution is not None
                    else None
                ),
            )
            derivative_info = DerivativeInfo(
                paths=structure_paths,
                # volumetric datasets intentionally empty: every shape path reads the
                # precomputed point integrands through the consumption seam
                E_der_map={},
                D_der_map={},
                E_fwd={},
                E_adj={},
                D_fwd={},
                D_adj={},
                eps_data={},
                bounds=struct_bounds,
                bounds_intersect=bounds_intersection(
                    sim_data_orig.simulation.bounds, struct_bounds
                ),
                simulation_bounds=sim_data_orig.simulation.bounds,
                frequencies=select_adjoint_freqs,
                updated_epsilon=None,
                is_medium_pec=structure.medium.is_pec,
                background_medium_is_pec=bool(
                    structure.background_medium and structure.background_medium.is_pec
                ),
                clipped_geometry=clipped_geometry,
                point_integrands=point_integrands_chunk,
                **_sampling_resolution_fields(sampling_resolution, len(select_adjoint_freqs)),
            )
            vjp_chunk = structure._compute_derivatives(
                derivative_info, sample_sets=sample_sets.by_key()
            )
            _accumulate_field_map(vjp_value_map, vjp_chunk)
        return _to_sim_fields_vjp(
            component_type="structures",
            component_index=structure_index,
            component_vjp=vjp_value_map,
        )

    # grab the forward and adjoint data
    fld_fwd = sim_data_fwd._get_adjoint_data(structure_index, monitor_tag="fld")
    fld_adj = sim_data_adj._get_adjoint_data(structure_index, monitor_tag="fld")
    eps_data = sim_data_adj._get_adjoint_data(structure_index, monitor_tag="eps")

    _require_freq_ascending(
        fld_fwd,
        component_type="structure",
        component_index=structure_index,
        dataset_name="forward field data",
    )
    _require_freq_ascending(
        fld_adj,
        component_type="structure",
        component_index=structure_index,
        dataset_name="adjoint field data",
    )
    _require_freq_ascending(
        eps_data,
        component_type="structure",
        component_index=structure_index,
        dataset_name="adjoint permittivity data",
    )

    adjoint_frequencies = _get_freq_coords(fld_adj)
    monitor_freqs = np.array(fld_adj.monitor.freqs)
    _validate_adjoint_frequencies(
        adjoint_frequencies=adjoint_frequencies,
        monitor_freqs=monitor_freqs,
        component_type="structure",
        component_index=structure_index,
    )

    # post normalize the adjoint fields if a single, broadband source
    fld_adj = scale_field_data(fld_adj, sim_data_adj.simulation.post_norm)

    combined_data_size = (
        _estimate_dataset_bytes(fld_adj)
        + _estimate_dataset_bytes(eps_data)
        + _estimate_dataset_bytes(fld_fwd)
    )
    if staged_point_path:
        # mixed structures consume point-cloud data inside the same chunk loop, so
        # its size participates in the same frequency memory budget
        combined_data_size += _point_cloud_data_size(
            sim_data_fwd, sim_data_adj, structure_index, sample_sets
        )

    # Filter forward field data to match adjoint monitor frequencies.
    fld_fwd = _filter_frequency_data(
        fld_fwd,
        adjoint_frequencies,
        component_type="structure",
        component_index=structure_index,
        dataset_name="filtered forward field data",
    )
    _validate_adjoint_frequencies(
        adjoint_frequencies=_get_freq_coords(fld_fwd),
        monitor_freqs=np.array(fld_adj.monitor.freqs),
        component_type="structure",
        component_index=structure_index,
    )

    structure = sim_data_fwd.simulation.structures[structure_index]
    clipped_geometry = (
        structure.geometry if _geometry_contains_clip_operation(structure.geometry) else None
    )

    # auto permittivity detection
    sim_orig = sim_data_orig.simulation
    # The adjoint field and permittivity monitors are already expanded by one grid
    # cell during monitor construction, so reuse that geometry directly here.
    plane_eps = eps_data.monitor.geometry

    # compute bounds intersection
    struct_bounds = structure.geometry.bounds
    bounds_intersect = bounds_intersection(sim_orig.bounds, struct_bounds)

    def updated_epsilon_full_impl(
        replacement_geometry: GeometryType,
        adjoint_frequencies: FreqDataArray | None,
        structure_index: int | None,
        eps_box: Box | None,
        sim_orig: td.Simulation,
    ) -> ScalarFieldDataArray:
        """Permittivity in ``eps_box`` after replacing this structure geometry."""
        updated_sim = sim_orig.updated_copy(
            structures=[
                sim_orig.structures[idx].updated_copy(geometry=replacement_geometry)
                if idx == structure_index
                else sim_orig.structures[idx]
                for idx in range(len(sim_orig.structures))
            ],
            grid_spec=td.components.grid.grid_spec.GridSpec.from_grid(sim_orig.grid),
        )
        eps_by_f = [
            updated_sim.epsilon(box=eps_box, coord_key="centers", freq=f)
            for f in adjoint_frequencies
        ]
        return xr.concat(eps_by_f, dim="f").assign_coords(f=adjoint_frequencies)

    def make_updated_epsilon(
        select_adjoint_freqs: FreqDataArray | None,
    ) -> Callable[[GeometryType], ScalarFieldDataArray]:
        updated_epsilon_full = functools.partial(
            updated_epsilon_full_impl,
            adjoint_frequencies=adjoint_frequencies,
            structure_index=structure_index,
            eps_box=plane_eps,
            sim_orig=sim_orig,
        )

        def updated_epsilon_wrapper(
            replacement_geometry: GeometryType,
            select_adjoint_freqs: FreqDataArray | None,
            updated_epsilon_full: Callable | None,
        ) -> ScalarFieldDataArray:
            return updated_epsilon_full(replacement_geometry).sel(f=select_adjoint_freqs)

        return functools.partial(
            updated_epsilon_wrapper,
            select_adjoint_freqs=select_adjoint_freqs,
            updated_epsilon_full=updated_epsilon_full,
        )

    n_freqs = len(adjoint_frequencies)
    H_info_exists = np.all([f"H{dim}" in fld_fwd.field_components for dim in "xyz"])

    freq_chunk_size = _adjoint_freq_chunk_size(
        n_freqs=n_freqs, combined_data_size=combined_data_size
    )

    # process in chunks
    vjp_value_map = {}

    for chunk_start in range(0, n_freqs, freq_chunk_size):
        chunk_end = min(chunk_start + freq_chunk_size, n_freqs)
        freq_slice = slice(chunk_start, chunk_end)

        select_adjoint_freqs = adjoint_frequencies[freq_slice]

        fld_fwd_chunk_data = fld_fwd.updated_copy(
            **_slice_field_data(fld_fwd.field_components, freq_slice)
        )
        eps_data_chunk_data = eps_data.updated_copy(
            **_slice_field_data(eps_data.field_components, freq_slice)
        )
        fld_adj_chunk_data = fld_adj.updated_copy(
            **_slice_field_data(fld_adj.field_components, freq_slice)
        )

        der_maps = get_derivative_maps(
            fld_fwd=fld_fwd_chunk_data,
            eps_fwd=eps_data_chunk_data,
            fld_adj=fld_adj_chunk_data,
            eps_adj=eps_data_chunk_data,
        )
        E_der_map_chunk = der_maps["E"].field_components
        D_der_map_chunk = der_maps["D"].field_components

        D_fwd_chunk = E_to_D(fld_fwd_chunk_data, eps_data_chunk_data).field_components
        D_adj_chunk = E_to_D(fld_adj_chunk_data, eps_data_chunk_data).field_components

        E_fwd_chunk = {
            key: val
            for key, val in fld_fwd_chunk_data.field_components.items()
            if key.startswith("E")
        }
        E_adj_chunk = {
            key: val
            for key, val in fld_adj_chunk_data.field_components.items()
            if key.startswith("E")
        }
        eps_data_chunk = eps_data_chunk_data.field_components

        H_der_map_chunk = None
        H_fwd_chunk = None
        H_adj_chunk = None

        if H_info_exists:
            H_der_map_chunk = der_maps["H"].field_components
            H_fwd_chunk = {
                key: val
                for key, val in fld_fwd_chunk_data.field_components.items()
                if key.startswith("H")
            }
            H_adj_chunk = {
                key: val
                for key, val in fld_adj_chunk_data.field_components.items()
                if key.startswith("H")
            }

        updated_epsilon = make_updated_epsilon(
            select_adjoint_freqs=select_adjoint_freqs,
        )

        # point-cloud path: compute this chunk's per-set integrands from the
        # point-cloud monitor data; each call touches only the chunk's frequencies,
        # so point consumption honors the same memory budget as the volumetric data
        point_integrands_chunk = None
        if staged_point_path:
            point_integrands_chunk = compute_point_integrands(
                sample_sets=sample_sets,
                structure_index=structure_index,
                sim_data_fwd=sim_data_fwd,
                sim_data_adj=sim_data_adj,
                frequencies=select_adjoint_freqs,
                adjoint_post_norm=sim_data_adj.simulation.post_norm,
                is_medium_pec=structure.medium.is_pec,
                background_medium_is_pec=bool(
                    structure.background_medium and structure.background_medium.is_pec
                ),
                clipped_geometry=clipped_geometry,
                material_length_scale=(
                    sampling_resolution.material_length_scale
                    if sampling_resolution is not None
                    else None
                ),
            )

        # create derivative info with sliced data
        derivative_info = DerivativeInfo(
            paths=structure_paths if structure_paths else numerical_paths_ordered,
            E_der_map=E_der_map_chunk,
            D_der_map=D_der_map_chunk,
            H_der_map=H_der_map_chunk,
            E_fwd=E_fwd_chunk,
            E_adj=E_adj_chunk,
            D_fwd=D_fwd_chunk,
            D_adj=D_adj_chunk,
            H_fwd=H_fwd_chunk,
            H_adj=H_adj_chunk,
            eps_data=eps_data_chunk,
            frequencies=select_adjoint_freqs,  # only chunk frequencies
            updated_epsilon=updated_epsilon,
            bounds=struct_bounds,
            bounds_intersect=bounds_intersect,
            simulation_bounds=sim_data_orig.simulation.bounds,
            is_medium_pec=structure.medium.is_pec,
            background_medium_is_pec=structure.background_medium
            and structure.background_medium.is_pec,
            clipped_geometry=clipped_geometry,
            point_integrands=point_integrands_chunk,
            **_sampling_resolution_fields(sampling_resolution, len(select_adjoint_freqs)),
        )

        if structure_paths:
            # compute derivatives for chunk; sample sets are shared across chunks
            vjp_chunk = structure._compute_derivatives(
                derivative_info,
                vjp_fns=custom_vjp,
                sample_sets=sample_sets.by_key() if sample_sets is not None else None,
            )

            # accumulate results
            _accumulate_field_map(vjp_value_map, vjp_chunk)

        if use_numerical_vjp:

            def _derivative_helper_impl(
                helper_derivative_info: DerivativeInfo,
                *,
                derivative_view: DerivativeView | None = None,
                chunk_derivative_info: DerivativeInfo,
            ) -> AutogradFieldMap:
                requested_sections = {
                    path[0]
                    for path in helper_derivative_info.paths
                    if len(path) > 0 and path[0] in ("geometry", "medium")
                }

                target_geometry = structure.geometry
                target_medium = structure.medium
                if derivative_view is not None:
                    if derivative_view.geometry is not None:
                        target_geometry = derivative_view.geometry
                    if "geometry" in requested_sections:
                        if derivative_view.geometry is None:
                            log.warning(
                                "derivative_helper received geometry paths but "
                                "derivative_view.geometry is None; falling back to the original "
                                "structure geometry.",
                                log_once=True,
                            )
                    if derivative_view.medium is not None:
                        target_medium = derivative_view.medium
                    if "medium" in requested_sections:
                        if derivative_view.medium is None:
                            log.warning(
                                "derivative_helper received medium paths but "
                                "derivative_view.medium is None; falling back to the original "
                                "structure medium.",
                                log_once=True,
                            )

                target_structure = structure.updated_copy(
                    geometry=target_geometry,
                    medium=target_medium,
                )
                target_is_medium_pec = target_structure.medium.is_pec
                target_background_medium_is_pec = bool(
                    target_structure.background_medium and target_structure.background_medium.is_pec
                )
                if target_is_medium_pec and not chunk_derivative_info.is_medium_pec:
                    raise AdjointError(
                        "derivative_helper cannot switch to a PEC medium in derivative_view when "
                        "the original structure was non-PEC because the required magnetic-field "
                        "derivative data were not collected."
                    )
                if (
                    target_background_medium_is_pec
                    and not chunk_derivative_info.background_medium_is_pec
                ):
                    raise AdjointError(
                        "derivative_helper cannot switch to a PEC background medium in "
                        "derivative_view when the original structure background was non-PEC "
                        "because the required magnetic-field derivative data were not collected."
                    )
                target_bounds = target_structure.geometry.bounds
                if derivative_view is not None and derivative_view.geometry is not None:
                    if not bounds_contains(struct_bounds, target_bounds):
                        raise AdjointError(
                            "derivative_helper derivative_view.geometry bounds must be contained "
                            "within the original structure bounds because derivative fields are "
                            "evaluated on the original monitor volume."
                        )
                # Numerical VJP helpers can request paths that did not go through setup_run().
                for helper_path in helper_derivative_info.paths:
                    target_structure._resolve_autograd_route(tuple(helper_path))
                substitutes_structure = derivative_view is not None and (
                    derivative_view.geometry is not None or derivative_view.medium is not None
                )
                helper_sampling_resolution = None
                if "geometry" in requested_sections:
                    if not substitutes_structure and sampling_resolution is not None:
                        # Differentiating the structure as-is, so the precomputed resolution
                        # already describes this simulation and structure index. Reuse it: each
                        # recompute would rebuild target_sim with a cold volumetric_structures
                        # cache and rescan every overlapping medium, once per helper call.
                        helper_sampling_resolution = sampling_resolution
                    else:
                        from tidy3d.components.autograd.spacing import adjoint_sampling_resolution

                        target_structures = list(sim_orig.structures)
                        target_structures[structure_index] = target_structure
                        target_sim = sim_orig.updated_copy(
                            structures=tuple(target_structures),
                            validate=False,
                        )
                        helper_sampling_resolution = adjoint_sampling_resolution(
                            target_sim, structure_index
                        )
                shared_interpolators = helper_derivative_info.interpolators
                if shared_interpolators is None:
                    # Reuse per-chunk interpolators to avoid rebuilding in helper loops.
                    shared_interpolators = chunk_derivative_info.create_interpolators()
                # Do not replace eps_data or forward/adjoint fields here: derivative_view only
                # changes derivative dispatch geometry/medium and does not rerun simulations.
                helper_derivative_info_use = helper_derivative_info.updated_copy(
                    bounds=target_bounds,
                    bounds_intersect=bounds_intersection(sim_orig.bounds, target_bounds),
                    updated_epsilon=make_updated_epsilon(
                        select_adjoint_freqs=helper_derivative_info.frequencies,
                    ),
                    is_medium_pec=target_is_medium_pec,
                    background_medium_is_pec=target_background_medium_is_pec,
                    clipped_geometry=(
                        target_structure.geometry
                        if _geometry_contains_clip_operation(target_structure.geometry)
                        else None
                    ),
                    **_sampling_resolution_fields(
                        helper_sampling_resolution, len(helper_derivative_info.frequencies)
                    ),
                    interpolators=shared_interpolators,
                    # the helper's sample sets are late-generated for the (possibly
                    # substituted) target geometry: recorded point integrands belong
                    # to the original geometry's surface points and must never be
                    # consumed here — the helper uses the volumetric contract
                    point_integrands=None,
                    deep=False,
                )
                helper_sample_sets = None
                helper_geometry_paths = [
                    tuple(path[1:])
                    for path in helper_derivative_info_use.paths
                    if path and path[0] == "geometry"
                ]
                if helper_geometry_paths:
                    helper_ctx = SamplingContext(
                        bounds=target_bounds,
                        bounds_intersect=bounds_intersection(sim_orig.bounds, target_bounds),
                        simulation_bounds=sim_orig.bounds,
                        spacing=helper_sampling_resolution.spacing,
                        material_wavelength=helper_sampling_resolution.material_wavelength,
                    )
                    helper_sample_sets = target_structure.geometry._make_adjoint_sample_sets(
                        paths=helper_geometry_paths, ctx=helper_ctx
                    )
                return target_structure._compute_derivatives(
                    helper_derivative_info_use, sample_sets=helper_sample_sets
                )

            derivative_helper = functools.partial(
                _derivative_helper_impl,
                chunk_derivative_info=derivative_info,
            )

            if numerical_structure._uses_derivative_helper:
                gradients = numerical_vjp_fn(
                    numerical_params_static,
                    derivative_info=derivative_info,
                    derivative_helper=derivative_helper,
                )
            else:
                gradients = numerical_vjp_fn(
                    numerical_params_static,
                    derivative_info=derivative_info,
                )

            if not isinstance(gradients, dict):
                raise AdjointError(
                    "Numerical structure VJP function must return a dict mapping paths to gradients."
                )

            missing_paths = set(numerical_paths_ordered) - set(gradients.keys())
            if missing_paths:
                raise AdjointError(
                    "Numerical structure VJP function did not return gradients for paths: "
                    f"{sorted(missing_paths)}."
                )

            for path in numerical_paths_ordered:
                grad_value = gradients.get(path)
                if grad_value is None:
                    continue
                if path in numerical_value_map:
                    existing = numerical_value_map[path]
                    if isinstance(existing, (list, tuple)) and isinstance(
                        grad_value, (list, tuple)
                    ):
                        numerical_value_map[path] = type(existing)(
                            x + y for x, y in zip(existing, grad_value)
                        )
                    else:
                        numerical_value_map[path] = existing + grad_value
                else:
                    numerical_value_map[path] = grad_value

    sim_fields_vjp = {}
    if structure_paths:
        sim_fields_vjp.update(
            _to_sim_fields_vjp(
                component_type="structures",
                component_index=structure_index,
                component_vjp=vjp_value_map,
            )
        )
    if use_numerical_vjp:
        sim_fields_vjp.update(
            _to_sim_fields_vjp(
                component_type="numerical",
                component_index=structure_index,
                component_vjp=numerical_value_map,
            )
        )
    return sim_fields_vjp
