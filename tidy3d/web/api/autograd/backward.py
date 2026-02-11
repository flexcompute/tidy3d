from __future__ import annotations

import functools
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

import tidy3d as td
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.utils import accumulate_field_map as _accumulate_field_map
from tidy3d.components.data.data_array import DataArray
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.packaging import disable_local_subpixel

from .utils import E_to_D, filter_vjp_map, get_derivative_maps

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Union

    from tidy3d import Medium
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.data.data_array import FreqDataArray, ScalarFieldDataArray
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.geometry.utils import GeometryType

    from .types import CustomVJPConfig


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
        include_untraced_data_arrays=True, starting_path=("data",)
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


def _compute_eps_array(medium: Medium, frequencies: list[float]) -> DataArray:
    """Compute permittivity array for all frequencies."""
    eps_data = [np.mean(medium.eps_model(f)) for f in frequencies]
    return DataArray(data=np.array(eps_data), dims=("f",), coords={"f": frequencies})


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


@disable_local_subpixel
def postprocess_adj(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_keys: list[tuple],
    custom_vjp: Optional[tuple[CustomVJPConfig, ...]] = None,
) -> AutogradFieldMap:
    """Postprocess some data from the adjoint simulation into the VJP for the original sim flds."""

    def get_all_paths(match_structure_index: int) -> tuple[tuple[str, str, int]]:
        """Get all the paths that may appear in autograd for this structure index. This allows a
        custom_vjp to be called for all autograd paths for the structure.
        """
        all_paths = tuple(
            tuple(structure_path)
            for namespace, structure_index_, *structure_path in sim_fields_keys
            if structure_index_ == match_structure_index
        )

        return all_paths

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

    # map of index into 'structures' to the paths we need VJPs for
    sim_vjp_map: defaultdict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for namespace, structure_index, *structure_path in sim_fields_keys:
        structure_path = tuple(structure_path)
        if namespace == "structures":
            sim_vjp_map[structure_index].append(structure_path)

    # store the derivative values given the forward and adjoint data
    sim_fields_vjp = {}
    all_structure_indices = sorted(set(sim_vjp_map.keys()))

    for structure_index in all_structure_indices:
        structure_paths = tuple(sim_vjp_map.get(structure_index, ()))

        # grab the forward and adjoint data
        fld_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="fld")
        eps_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="eps")
        fld_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="fld")
        eps_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="eps")

        def sort_by_freq_ascending(
            dataset: Union[td.PermittivityData, td.FieldData],
        ) -> Union[td.PermittivityData, td.FieldData]:
            dataset_sort = {}
            for key, val in dataset.field_components.items():
                dataset_sort[key] = val.sortby("f", ascending=True)

            return dataset.updated_copy(**dataset_sort)

        # sort data by ascending frequency value to ensure data ordering is consistent
        fld_fwd = sort_by_freq_ascending(fld_fwd)
        eps_fwd = sort_by_freq_ascending(eps_fwd)
        fld_adj = sort_by_freq_ascending(fld_adj)
        eps_adj = sort_by_freq_ascending(eps_adj)

        freqs_adj = np.array(fld_adj.monitor.freqs)

        # post normalize the adjoint fields if a single, broadband source
        fwd_flds_adj_normed = {}
        for key, val in fld_adj.field_components.items():
            fwd_flds_adj_normed[key] = val * sim_data_adj.simulation.post_norm

        fld_adj = fld_adj.updated_copy(**fwd_flds_adj_normed)

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
        first_field_component = next(iter(E_der_map.field_components.values()))
        adjoint_frequencies = np.array(first_field_component.coords["f"].values)

        monitor_freqs = np.array(fld_adj.monitor.freqs)
        if len(adjoint_frequencies) != len(monitor_freqs) or not np.allclose(
            np.sort(adjoint_frequencies), np.sort(monitor_freqs), rtol=1e-10, atol=0
        ):
            raise ValueError(
                f"Frequency mismatch in adjoint postprocessing for structure {structure_index}. "
                f"Expected frequencies from monitor: {monitor_freqs}, "
                f"but derivative map has: {adjoint_frequencies}. "
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
        high_coords = [
            center + 0.5 * size for center, size in zip(plane_eps.center, plane_eps.size)
        ]

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
            """Return the simulation permittivity for eps_box after replacing the geometry
            for this structure with a new geometry. This is helpful for carrying out finite
            difference permittivity computations.
            """
            update_sim = sim_orig.updated_copy(
                structures=[
                    sim_orig.structures[idx].updated_copy(geometry=replacement_geometry)
                    if idx == structure_index
                    else sim_orig.structures[idx]
                    for idx in range(len(sim_orig.structures))
                ],
                grid_spec=td.components.grid.grid_spec.GridSpec.from_grid(sim_orig.grid),
            )

            eps_by_f = [
                update_sim.epsilon(box=eps_box, coord_key="centers", freq=f)
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
                # Get permittivity function for a subset of frequencies
                return updated_epsilon_full(replacement_geometry).sel(f=select_adjoint_freqs)

            updated_epsilon = functools.partial(
                updated_epsilon_wrapper,
                select_adjoint_freqs=select_adjoint_freqs,
                updated_epsilon_full=updated_epsilon_full,
            )

            if structure_paths:
                # create derivative info with sliced data
                derivative_info_struct = DerivativeInfo(
                    paths=structure_paths,
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
                    updated_epsilon=updated_epsilon,
                    frequencies=select_adjoint_freqs,  # only chunk frequencies
                    bounds=struct_bounds,
                    bounds_intersect=bounds_intersect,
                    simulation_bounds=sim_data_orig.simulation.bounds,
                    is_medium_pec=structure.medium.is_pec,
                    background_medium_is_pec=structure.background_medium
                    and structure.background_medium.is_pec,
                )

                vjp_fns = custom_vjp_lookup.get(structure_index)
                vjp_chunk = structure._compute_derivatives(derivative_info_struct, vjp_fns=vjp_fns)

            # accumulate results
            _accumulate_field_map(vjp_value_map, vjp_chunk)

        # store vjps in output map
        for structure_path, vjp_value in vjp_value_map.items():
            sim_path = ("structures", structure_index, *list(structure_path))
            sim_fields_vjp[sim_path] = vjp_value

    return sim_fields_vjp
