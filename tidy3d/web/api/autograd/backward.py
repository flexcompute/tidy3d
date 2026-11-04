from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from flexcompute.core._migration.em.derivatives import recorded_bytes
from flexcompute.core._migration.em.derivatives.execute import frequency_chunk_size

import tidy3d as td
from tidy3d.components.autograd import get_static
from tidy3d.components.autograd.derivative_utils import (
    DerivativeInfo,
    contains_clip_operation,
)
from tidy3d.components.autograd.monitor_names import adjoint_monitor_name
from tidy3d.components.autograd.utils import accumulate_field_map as _accumulate_field_map
from tidy3d.components.bound_ops import bounds_intersection
from tidy3d.components.thin_lens import validate_thin_lens_adjoint_background
from tidy3d.em.translate.shape_terms import warn_collar_source_cuts
from tidy3d.exceptions import AdjointError
from tidy3d.log import log
from tidy3d.packaging import disable_local_subpixel

from .flux_monitor import expand_flux_monitor_vjps
from .utils import (
    E_to_D,
    expand_custom_vjp_configs,
    filter_vjp_map,
    get_derivative_maps,
    scale_field_data,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.spacing import SamplingResolution
    from tidy3d.components.data.data_array import FreqDataArray, ScalarFieldDataArray
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.monitor import Monitor

    from .context import AdjointPostprocessInputs
    from .derivatives import DerivativeRun
    from .types import NumericalStructureConfig


@dataclass(frozen=True)
class AdjointSetupResult:
    """Result of setting up adjoint simulations for a VJP map."""

    simulations: list[td.Simulation]
    all_sources_underflowed: bool = False


def make_adjoint_monitors(
    simulation: td.Simulation, derivative_run: DerivativeRun
) -> list[Monitor]:
    """The adjoint monitors: the program's, and the volumetric ones of callback-owned paths."""

    monitors_fld, monitors_eps = simulation._make_adjoint_monitors(
        list(derivative_run.callback_keys)
    )
    return [*monitors_fld, *monitors_eps, *derivative_run.adjoint_monitors]


def _sampling_resolution_fields(
    sampling_resolution: SamplingResolution | None,
) -> dict[str, float | None]:
    """Return resolved ``DerivativeInfo`` sampling fields from optional precomputed data."""

    if sampling_resolution is None:
        return {
            "resolved_adaptive_vjp_spacing": None,
            "resolved_material_length_scale": None,
            "resolved_material_wavelength": None,
        }
    return {
        "resolved_adaptive_vjp_spacing": sampling_resolution.spacing,
        "resolved_material_length_scale": sampling_resolution.material_length_scale,
        "resolved_material_wavelength": sampling_resolution.material_wavelength,
    }


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    max_num_adjoint_per_fwd: int,
    derivative_run: DerivativeRun,
    already_filtered: bool = False,
    sim_data_fwd: td.SimulationData | None = None,
    return_result: bool = False,
) -> list[td.Simulation] | AdjointSetupResult:
    """Construct an adjoint simulation from a set of data_fields for the VJP."""

    td.log.info("Running custom vjp (adjoint) pipeline.")

    if not already_filtered:
        data_fields_vjp = filter_vjp_map(data_fields_vjp)

    # if all entries are zero, there is no adjoint sim to run
    if not data_fields_vjp:
        result = AdjointSetupResult([])
        return result if return_result else result.simulations

    # Unused overlap monitors need not satisfy the adjoint profile contract.
    for index in {path[1] for path in data_fields_vjp}:
        monitor = sim_data_orig.data[index].monitor
        if isinstance(monitor, td.ThinLensOverlapMonitor):
            traced_fields = [path for binding in derivative_run.bindings for path in binding.paths]
            validate_thin_lens_adjoint_background(
                sim_data_orig.simulation,
                monitor,
                np.asarray(monitor.freqs),
                traced_fields=[*traced_fields, *derivative_run.callback_keys],
            )

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

    adjoint_monitors = make_adjoint_monitors(sim_data_orig.simulation, derivative_run)

    adjoint_setup_result = sim_data_vjp._make_adjoint_sims_with_result(
        data_vjp_paths=data_vjp_paths,
        adjoint_monitors=adjoint_monitors,
    )
    sims_adj = adjoint_setup_result.simulations
    warn_collar_source_cuts(derivative_run.program, sims_adj)

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


def _callback_data_bytes(
    sim_data_fwd: td.SimulationData, sim_data_adj: td.SimulationData, structure_index: int
) -> int:
    """Bytes of one structure's volumetric callback data: forward and adjoint fields, permittivity.

    Sized from the raw data, since indexing the simulation data would copy every monitor.
    """
    return sum(
        recorded_bytes(data, adjoint_monitor_name(index=structure_index, monitor_tag=tag))
        for data, tag in ((sim_data_fwd, "fld"), (sim_data_adj, "fld"), (sim_data_adj, "eps"))
    )


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


def _warn_if_adjoint_did_not_decay(
    sim_data_adj: td.SimulationData, sim_data_fwd: td.SimulationData
) -> None:
    """Warn when an adjoint run ended with its fields still ringing but the forward's had not.

    Every adjoint run time is derived from the forward one, either by inheriting it or, for
    a combined temporal simulation, from the duration the forward actually ran. Neither is
    a guarantee about the adjoint itself: adjoint sources can excite longer-lived content
    than the forward sources did, and an adjoint truncated mid-ring-down biases its
    gradients without failing in any visible way. Comparing against the forward keeps this
    to the case worth acting on -- a forward that never converged is its own problem, and
    the forward pipeline already reports it.

    The forward record is the original simulation data, which is what the adjoint plan is
    built from and what a temporal pulse is sized against. A forward carrying no solver
    log is skipped rather than read from some other container: without that log the
    reduction could not verify the measured duration either, so no adjoint run was
    shortened and there is nothing here that standard grouping would not also face.
    """

    shutoff = sim_data_adj.simulation.shutoff
    if shutoff <= 0.0:
        return
    # checked rather than caught: 'field_decay' raises for a missing log, and a raised
    # 'Tidy3dError' logs itself, so catching it would emit the noise this guard avoids
    if sim_data_adj.log is None or sim_data_fwd.log is None:
        return

    adjoint_decay_data = sim_data_adj.field_decay
    forward_decay_data = sim_data_fwd.field_decay
    if len(adjoint_decay_data) == 0 or len(forward_decay_data) == 0:
        return

    adjoint_decay = float(adjoint_decay_data.values[-1])
    forward_decay = float(forward_decay_data.values[-1])
    if adjoint_decay <= shutoff or forward_decay > shutoff:
        return

    log.warning(
        "The adjoint simulation ended before its fields decayed below the shutoff "
        f"threshold (final field decay {adjoint_decay:.3g} vs threshold {shutoff:g}), "
        "while the forward simulation did decay. Adjoint sources can excite longer-lived "
        "content than the forward sources did, so the gradients from this run may be "
        "truncated. Consider increasing 'run_time' on the forward simulation; if this "
        "adjoint came from a combined field-source reduction, raising "
        "'config.adjoint.field_source_temporal_max_run_time_ratio' lengthens it.",
        log_once=True,
    )


@disable_local_subpixel
def postprocess_adj(
    sim_data_adj: td.SimulationData,
    *,
    postprocess_inputs: AdjointPostprocessInputs,
) -> AutogradFieldMap:
    """VJPs from one adjoint simulation: the program's paths, then each user callback's paths.

    A numerical callback that accepts ``query_results`` is called once per executor batch, so
    its ``derivative_info`` chunks and its query results cover the same frequencies; the batch
    plan therefore budgets those callbacks' volumetric data too.
    """

    from .derivatives import run_vjps

    _warn_if_adjoint_did_not_decay(sim_data_adj, postprocess_inputs.sim_data_orig)
    run = postprocess_inputs.derivative_run
    queried = {
        index
        for kind, index, *_ in run.callback_keys
        if kind == "numerical"
        and postprocess_inputs.numerical_structure_map[index]._accepts_query_results
    }
    callback_bytes = sum(
        _callback_data_bytes(postprocess_inputs.sim_data_fwd, sim_data_adj, index)
        for index in queried
    )
    program_vjps = run_vjps(run, postprocess_inputs.sim_data_fwd, sim_data_adj, callback_bytes)
    sim_fields_vjp = dict(program_vjps.vjps)
    custom_vjp_lookup = expand_custom_vjp_configs(
        postprocess_inputs.custom_vjp, postprocess_inputs.sim_fields_keys
    )
    owned: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for kind, index, *path in run.callback_keys:
        owned[kind, index].append(tuple(path))
    for (kind, index), paths in owned.items():
        common = {
            "sim_data_adj": sim_data_adj,
            "sim_data_orig": postprocess_inputs.sim_data_orig,
            "sim_data_fwd": postprocess_inputs.sim_data_fwd,
            "structure_index": index,
            "sampling_resolution": postprocess_inputs.sampling_resolutions.get(index),
        }
        if kind == "structures":
            vjp = _process_callback_gradients(
                **common, structure_paths=paths, custom_vjp=custom_vjp_lookup.get(index)
            )
        else:
            config = postprocess_inputs.numerical_structure_map[index]
            queried = config._accepts_query_results
            vjp = _process_callback_gradients(
                **common,
                structure_paths=[],
                numerical_structure=config,
                numerical_paths=paths,
                query_results_by_chunk=program_vjps.query_results.get(index) if queried else None,
                query_bindings=tuple(
                    b for b in run.bindings if b.paths[0][:2] == ("numerical", index)
                )
                if queried
                else (),
                freq_chunk_size=program_vjps.batch_size if queried else None,
            )
        sim_fields_vjp.update(vjp)
    return sim_fields_vjp


def _process_callback_gradients(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    structure_index: int,
    structure_paths: list[tuple],
    sampling_resolution: SamplingResolution | None,
    custom_vjp: dict[tuple[str, str], Callable[..., Any]] | None = None,
    numerical_structure: NumericalStructureConfig | None = None,
    numerical_paths: list[tuple] | None = None,
    query_results_by_chunk: list[tuple[np.ndarray, dict[str, dict[tuple, Any]]]] | None = None,
    query_bindings: tuple[Any, ...] = (),
    freq_chunk_size: int | None = None,
) -> AutogradFieldMap:
    """Gradients of one structure's callback-owned paths, from its volumetric monitors.

    ``structure_paths`` are custom-VJP paths; ``numerical_paths`` go to the numerical
    structure's ``compute_derivatives``. Both receive a ``DerivativeInfo`` per frequency chunk.
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

    combined_data_size = _callback_data_bytes(sim_data_fwd, sim_data_adj, structure_index)

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
    clipped_geometry = structure.geometry if contains_clip_operation(structure.geometry) else None

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

    if freq_chunk_size is None:
        from .derivatives import execution_settings

        freq_chunk_size = frequency_chunk_size(n_freqs, combined_data_size, execution_settings())

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
            **_sampling_resolution_fields(sampling_resolution),
        )

        if structure_paths:
            vjp_chunk = structure._custom_vjp_derivatives(derivative_info, custom_vjp)

            # accumulate results
            _accumulate_field_map(vjp_value_map, vjp_chunk)

        if use_numerical_vjp:
            if numerical_structure._accepts_query_results:
                if query_results_by_chunk:
                    batch = chunk_start // freq_chunk_size
                    frequencies, query_results = (
                        query_results_by_chunk[batch]
                        if batch < len(query_results_by_chunk)
                        else ((), {})
                    )
                    # chunks pair with program batches by position, which holds while both
                    # cover the adjoint run's frequencies in the same order
                    if not np.array_equal(frequencies, np.asarray(select_adjoint_freqs)):
                        raise AdjointError(
                            f"Numerical structure {structure_index}'s callback chunk covers "
                            f"frequencies {np.asarray(select_adjoint_freqs).tolist()}, but its "
                            f"derivative-program batch covers {np.asarray(frequencies).tolist()}; "
                            "its query results would be paired with the wrong frequencies."
                        )
                else:
                    # no executor batches means no executable terms: every declared query is a
                    # zero binding, so it binds here, on this chunk's frequencies
                    from .derivatives import bind_queries

                    query_results = bind_queries({}, query_bindings, select_adjoint_freqs).get(
                        structure_index, {}
                    )
                gradients = numerical_vjp_fn(
                    numerical_params_static,
                    derivative_info=derivative_info,
                    query_results=query_results,
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
