from __future__ import annotations

import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

import tidy3d as td
from tidy3d.components.autograd import get_static
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.utils import accumulate_field_map as _accumulate_field_map
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.source.adjoint_helpers import (
    collapse_source_adjoint_to_dataset_frequency,
)
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.packaging import disable_local_subpixel

from .utils import E_to_D, filter_vjp_map, get_derivative_maps, scale_field_data

if TYPE_CHECKING:
    from typing import Callable, Optional, Union

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.data.data_array import ScalarFieldDataArray
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.geometry.utils import GeometryType

    from .types import CustomVJPConfig, NumericalStructureConfig


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_keys: list[tuple],
    max_num_adjoint_per_fwd: int,
    already_filtered: bool = False,
) -> list[td.Simulation]:
    """Construct an adjoint simulation from a set of data_fields for the VJP."""

    td.log.info("Running custom vjp (adjoint) pipeline.")

    if not already_filtered:
        data_fields_vjp = filter_vjp_map(data_fields_vjp)

    # if all entries are zero, there is no adjoint sim to run
    if not data_fields_vjp:
        return []

    # start with the full simulation data structure and either zero out the fields
    # that have no tracer data for them or insert the tracer data
    full_sim_data_dict = sim_data_orig._strip_traced_fields(
        include_untraced_data_arrays=True, starting_paths=(("data",),)
    )
    for path in full_sim_data_dict.keys():
        if path in data_fields_vjp:
            full_sim_data_dict[path] = data_fields_vjp[path]
        else:
            full_sim_data_dict[path] *= 0

    # insert the raw VJP data into the .data of the original SimulationData
    sim_data_vjp = sim_data_orig._insert_traced_fields(field_mapping=full_sim_data_dict)

    # make adjoint simulation from that SimulationData
    data_vjp_paths = set(data_fields_vjp.keys())

    num_monitors = len(sim_data_orig.simulation.monitors)
    adjoint_monitors = sim_data_orig.simulation._with_adjoint_monitors(sim_fields_keys).monitors[
        num_monitors:
    ]

    sims_adj = sim_data_vjp._make_adjoint_sims(
        data_vjp_paths=data_vjp_paths,
        adjoint_monitors=adjoint_monitors,
    )

    if len(sims_adj) > max_num_adjoint_per_fwd:
        raise AdjointError(
            f"Number of adjoint simulations ({len(sims_adj)}) exceeds the maximum allowed "
            f"({max_num_adjoint_per_fwd}) per forward simulation. This typically means that "
            "there are many frequencies and monitors in the simulation that are being differentiated "
            "w.r.t. in the objective function. To proceed, please double-check the simulation "
            "setup, increase the 'max_num_adjoint_per_fwd' parameter in the run function, and re-run."
        )

    return sims_adj


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


def _sort_by_freq_ascending(
    dataset: Union[td.PermittivityData, td.FieldData],
) -> Union[td.PermittivityData, td.FieldData]:
    """Sort all field components by ascending frequency coordinates."""
    dataset_sorted = {}
    for key, val in dataset.field_components.items():
        dataset_sorted[key] = val.sortby("f", ascending=True)

    return dataset.updated_copy(**dataset_sorted)


def _validate_adjoint_frequencies(
    *,
    adjoint_frequencies: np.ndarray,
    monitor_freqs: np.ndarray,
    component_type: str,
    component_index: int,
) -> None:
    """Validate that field-data frequencies match monitor frequencies."""
    if len(adjoint_frequencies) != len(monitor_freqs) or not np.allclose(
        np.sort(adjoint_frequencies), np.sort(monitor_freqs), rtol=1e-10, atol=0
    ):
        raise ValueError(
            f"Frequency mismatch in adjoint postprocessing for {component_type} "
            f"{component_index}. Expected frequencies from monitor: {monitor_freqs}, "
            f"but derivative map has: {adjoint_frequencies}. "
        )


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
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_keys: list[tuple],
    numerical_structure_map: Optional[dict[int, NumericalStructureConfig]] = None,
    custom_vjp: Optional[tuple[CustomVJPConfig, ...]] = None,
) -> AutogradFieldMap:
    """Postprocess some data from the adjoint simulation into the VJP for the original sim flds."""

    def get_all_paths(match_structure_index: int) -> tuple[tuple[Any, ...], ...]:
        """Get traced autograd paths for one structure index.

        ``sim_fields_keys`` can contain entries for both ``"structures"`` and ``"sources"``.
        Restricting to ``"structures"`` here avoids mixing source paths into
        structure-level ``custom_vjp`` expansion when indices overlap.
        """
        return tuple(
            tuple(component_path)
            for component_type, component_index, *component_path in sim_fields_keys
            if component_type == "structures" and component_index == match_structure_index
        )

    custom_vjp_lookup: dict[int, dict[tuple[str, str], Callable[..., Any]]] = {}
    if custom_vjp:
        for vjp_config in custom_vjp:
            structure_index = vjp_config.structure
            vjp_fn = vjp_config.compute_derivatives
            path = vjp_config.path_key

            if path is None:
                for match_path in get_all_paths(structure_index):
                    custom_vjp_lookup.setdefault(structure_index, {})[match_path[0:2]] = vjp_fn
            else:
                custom_vjp_lookup.setdefault(structure_index, {})[path] = vjp_fn

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
            sim_fields_vjp.update(
                _process_structure_gradients(
                    sim_data_adj,
                    sim_data_orig,
                    sim_data_fwd,
                    component_index,
                    component_paths,
                    custom_vjp=custom_vjp_lookup.get(component_index),
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
    else:
        raise TypeError(
            f"Source dataset frequency is only defined for custom sources, got '{source.type}'."
        )
    component = next(iter(dataset.field_components.values()))
    freqs = np.asarray(component.coords["f"].data, dtype=float).reshape(-1)
    return float(freqs[0])


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
    fld_adj = _sort_by_freq_ascending(fld_adj)

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
    fld_adj = collapse_source_adjoint_to_dataset_frequency(fld_adj, source_dataset_freq)

    e_adj = {k: v for k, v in fld_adj.field_components.items() if k.startswith("E")}
    h_adj = {k: v for k, v in fld_adj.field_components.items() if k.startswith("H")}

    bounds = source.geometry.bounds
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
        frequencies=_get_freq_coords(fld_adj),
        bounds=bounds,
        bounds_intersect=bounds,
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
    custom_vjp: Optional[dict[tuple[str, str], Callable[..., Any]]] = None,
    numerical_structure: Optional[NumericalStructureConfig] = None,
    numerical_paths: Optional[list[tuple]] = None,
) -> AutogradFieldMap:
    """Process gradients for a specific structure."""

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
    fld_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="fld")
    eps_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="eps")
    fld_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="fld")
    eps_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="eps")

    # sort data by ascending frequency value to ensure data ordering is consistent
    fld_fwd = _sort_by_freq_ascending(fld_fwd)
    eps_fwd = _sort_by_freq_ascending(eps_fwd)
    fld_adj = _sort_by_freq_ascending(fld_adj)
    eps_adj = _sort_by_freq_ascending(eps_adj)

    freqs_adj = np.array(fld_adj.monitor.freqs)

    # post normalize the adjoint fields if a single, broadband source
    fld_adj = scale_field_data(fld_adj, sim_data_adj.simulation.post_norm)

    # maps of the E_fwd * E_adj and D_fwd * D_adj, each as as td.FieldData & 'Ex', 'Ey', 'Ez'
    der_maps = get_derivative_maps(
        fld_fwd=fld_fwd,
        eps_fwd=eps_fwd,
        fld_adj=fld_adj,
        eps_adj=eps_adj,
    )
    E_der_map = der_maps["E"]
    D_der_map = der_maps["D"]
    H_der_map = der_maps["H"]

    H_info_exists = H_der_map is not None

    def filter_adj_freq(
        dataset: Union[td.PermittivityData, td.FieldData], filter_freqs: np.ndarray
    ) -> Union[td.PermittivityData, td.FieldData]:
        dataset_filter_freq = {}
        for key, val in dataset.field_components.items():
            dataset_filter_freq[key] = val.sel(f=filter_freqs)

        return dataset.updated_copy(**dataset_filter_freq)

    fld_fwd = filter_adj_freq(fld_fwd, freqs_adj)
    eps_fwd = filter_adj_freq(eps_fwd, freqs_adj)

    D_fwd = E_to_D(fld_fwd, eps_fwd)
    D_adj = E_to_D(fld_adj, eps_fwd)

    structure = sim_data_fwd.simulation.structures[structure_index]

    # compute epsilon arrays for all frequencies
    # use frequencies from the actual computed derivative map to ensure they exist
    # in both forward and adjoint data (E_der_map = fld_fwd * fld_adj)
    adjoint_frequencies = _get_freq_coords(E_der_map)
    monitor_freqs = np.array(fld_adj.monitor.freqs)
    _validate_adjoint_frequencies(
        adjoint_frequencies=adjoint_frequencies,
        monitor_freqs=monitor_freqs,
        component_type="structure",
        component_index=structure_index,
    )

    # auto permittivity detection
    sim_orig = sim_data_orig.simulation
    plane_eps = eps_fwd.monitor.geometry
    sim_orig_grid_spec = td.components.grid.grid_spec.GridSpec.from_grid(sim_orig.grid)

    # permittivity without this structure
    structs_no_struct = list(sim_orig.structures)
    structs_no_struct.pop(structure_index)
    sim_no_structure = sim_orig.updated_copy(
        structures=structs_no_struct, monitors=[], sources=[], grid_spec=sim_orig_grid_spec
    )

    # for the outside permittivity of the structure, resize the bounds of the permittivity region
    # to make sure we capture data outside the structure bounds
    low_coords = [center - 0.5 * size for center, size in zip(plane_eps.center, plane_eps.size)]
    high_coords = [center + 0.5 * size for center, size in zip(plane_eps.center, plane_eps.size)]

    low_bounds = sim_orig.grid.boundaries.get_bounding_values(low_coords, "left", buffer=1)
    high_bounds = sim_orig.grid.boundaries.get_bounding_values(high_coords, "right", buffer=1)

    resized_center = [0.5 * (low + high) for low, high in zip(low_bounds, high_bounds)]
    resized_size = [(high - low) for low, high in zip(low_bounds, high_bounds)]

    resize_plane_eps = plane_eps.updated_copy(center=resized_center, size=resized_size)

    eps_no_structure_data = [
        sim_no_structure.epsilon(box=resize_plane_eps, coord_key="centers", freq=f)
        for f in adjoint_frequencies
    ]

    eps_no_structure = xr.concat(eps_no_structure_data, dim="f").assign_coords(
        f=adjoint_frequencies
    )

    if structure.medium.is_custom:
        # we can't make an infinite structure from a custom medium permittivity
        eps_inf_structure = None
    else:
        geometry_box = structure.geometry.bounding_box
        background_structures_2d = []
        sim_inf_background_medium = sim_orig.medium
        if np.any(np.array(geometry_box.size) == 0.0):
            zero_coordinate = tuple(geometry_box.size).index(0.0)
            new_size = [td.inf, td.inf, td.inf]
            new_size[zero_coordinate] = 0.0

            background_structures_2d = [
                structure.updated_copy(geometry=geometry_box.updated_copy(size=new_size))
            ]
        else:
            sim_inf_background_medium = structure.medium

        # permittivity with infinite structure
        structs_inf_struct = list(sim_orig.structures)[structure_index + 1 :]
        sim_inf_structure = sim_orig.updated_copy(
            structures=background_structures_2d + structs_inf_struct,
            medium=sim_inf_background_medium,
            monitors=[],
            sources=[],
            grid_spec=sim_orig_grid_spec,
        )

        eps_inf_structure_data = [
            sim_inf_structure.epsilon(box=plane_eps, coord_key="centers", freq=f)
            for f in adjoint_frequencies
        ]

        eps_inf_structure = xr.concat(eps_inf_structure_data, dim="f").assign_coords(
            f=adjoint_frequencies
        )

    # compute bounds intersection
    struct_bounds = rmin_struct, rmax_struct = structure.geometry.bounds
    rmin_sim, rmax_sim = sim_orig.bounds
    rmin_intersect = tuple([max(a, b) for a, b in zip(rmin_sim, rmin_struct)])
    rmax_intersect = tuple([min(a, b) for a, b in zip(rmax_sim, rmax_struct)])
    bounds_intersect = (rmin_intersect, rmax_intersect)

    def updated_epsilon_full_impl(
        replacement_geometry: GeometryType,
        adjoint_frequencies: Optional[FreqDataArray],
        structure_index: Optional[int],
        eps_box: Optional[Box],
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

    updated_epsilon_full = functools.partial(
        updated_epsilon_full_impl,
        adjoint_frequencies=adjoint_frequencies,
        structure_index=structure_index,
        eps_box=resize_plane_eps,
        sim_orig=sim_orig,
    )

    # get chunk size - if None, process all frequencies as one chunk
    freq_chunk_size = config.adjoint.solver_freq_chunk_size
    n_freqs = len(adjoint_frequencies)
    if not freq_chunk_size or freq_chunk_size <= 0:
        freq_chunk_size = n_freqs
    else:
        freq_chunk_size = min(freq_chunk_size, n_freqs)

    # process in chunks
    vjp_value_map = {}

    for chunk_start in range(0, n_freqs, freq_chunk_size):
        chunk_end = min(chunk_start + freq_chunk_size, n_freqs)
        freq_slice = slice(chunk_start, chunk_end)

        select_adjoint_freqs = adjoint_frequencies[freq_slice]

        # slice field data for current chunk
        E_der_map_chunk = _slice_field_data(E_der_map.field_components, freq_slice)
        D_der_map_chunk = _slice_field_data(D_der_map.field_components, freq_slice)
        E_fwd_chunk = _slice_field_data(
            fld_fwd.field_components, freq_slice, component_indicator="E"
        )
        E_adj_chunk = _slice_field_data(
            fld_adj.field_components, freq_slice, component_indicator="E"
        )
        D_fwd_chunk = _slice_field_data(D_fwd.field_components, freq_slice)
        D_adj_chunk = _slice_field_data(D_adj.field_components, freq_slice)
        eps_data_chunk = _slice_field_data(eps_fwd.field_components, freq_slice)

        H_der_map_chunk = None
        H_fwd_chunk = None
        H_adj_chunk = None

        if H_info_exists:
            H_der_map_chunk = _slice_field_data(H_der_map.field_components, freq_slice)
            H_fwd_chunk = _slice_field_data(
                fld_fwd.field_components, freq_slice, component_indicator="H"
            )
            H_adj_chunk = _slice_field_data(
                fld_adj.field_components, freq_slice, component_indicator="H"
            )

        # slice epsilon arrays
        eps_no_structure_chunk = (
            eps_no_structure.isel(f=freq_slice) if eps_no_structure is not None else None
        )
        eps_inf_structure_chunk = (
            eps_inf_structure.isel(f=freq_slice) if eps_inf_structure is not None else None
        )

        def updated_epsilon_wrapper(
            replacement_geometry: GeometryType,
            select_adjoint_freqs: Optional[FreqDataArray],
            updated_epsilon_full: Optional[Callable],
        ) -> ScalarFieldDataArray:
            return updated_epsilon_full(replacement_geometry).sel(f=select_adjoint_freqs)

        updated_epsilon = functools.partial(
            updated_epsilon_wrapper,
            select_adjoint_freqs=select_adjoint_freqs,
            updated_epsilon_full=updated_epsilon_full,
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
            eps_in=eps_inf_structure_chunk,
            eps_out=eps_no_structure_chunk,
            frequencies=select_adjoint_freqs,  # only chunk frequencies
            updated_epsilon=updated_epsilon,
            bounds=struct_bounds,
            bounds_intersect=bounds_intersect,
            simulation_bounds=sim_data_orig.simulation.bounds,
            is_medium_pec=structure.medium.is_pec,
            background_medium_is_pec=structure.background_medium
            and structure.background_medium.is_pec,
        )

        if structure_paths:
            # compute derivatives for chunk
            vjp_chunk = structure._compute_derivatives(derivative_info, vjp_fns=custom_vjp)

            # accumulate results
            _accumulate_field_map(vjp_value_map, vjp_chunk)

        if use_numerical_vjp:
            gradients = numerical_vjp_fn(numerical_params_static, derivative_info=derivative_info)

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
