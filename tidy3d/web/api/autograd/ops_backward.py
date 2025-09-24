from __future__ import annotations

from collections import defaultdict

import numpy as np
import xarray as xr

import tidy3d as td
from tidy3d.components.autograd import AutogradFieldMap, get_static
from tidy3d.components.autograd.constants import ADJOINT_FREQ_CHUNK_SIZE
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.exceptions import AdjointError

from .utils import E_to_D, get_derivative_maps


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_keys: list[tuple],
    max_num_adjoint_per_fwd: int,
) -> list[td.Simulation]:
    """Construct an adjoint simulation from a set of data_fields for the VJP."""

    td.log.info("Running custom vjp (adjoint) pipeline.")

    # filter out any data_fields_vjp with all 0's
    data_fields_vjp = {
        k: get_static(v) for k, v in data_fields_vjp.items() if not np.allclose(v, 0)
    }

    for k, v in data_fields_vjp.items():
        if np.any(np.isnan(v)):
            raise AdjointError(
                f"NaN values detected for data field {k} in the adjoint pipeline. This may be "
                f"due to NaN values in the simulation data or the computed value of your "
                f"objective function."
            )

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


def _compute_eps_array(medium, frequencies):
    """Compute permittivity array for all frequencies."""
    eps_data = [np.mean(medium.eps_model(f)) for f in frequencies]
    return td.components.data.data_array.DataArray(
        data=np.array(eps_data), dims=("f",), coords={"f": frequencies}
    )


def _slice_field_data(
    field_data: dict, freqs: np.ndarray, component_indicator: str | None = None
) -> dict:
    """Slice field data dictionary along frequency dimension."""
    if component_indicator:
        return {k: v.sel(f=freqs) for k, v in field_data.items() if component_indicator in k}
    else:
        return {k: v.sel(f=freqs) for k, v in field_data.items()}


def postprocess_adj(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_keys: list[tuple],
) -> AutogradFieldMap:
    """Postprocess some data from the adjoint simulation into the VJP for the original sim flds."""

    # map of index into 'structures' to the list of paths we need vjps for
    sim_vjp_map = defaultdict(list)
    for _, structure_index, *structure_path in sim_fields_keys:
        structure_path = tuple(structure_path)
        sim_vjp_map[structure_index].append(structure_path)

    # store the derivative values given the forward and adjoint data
    sim_fields_vjp = {}
    for structure_index, structure_paths in sim_vjp_map.items():
        # grab the forward and adjoint data
        fld_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="fld")
        eps_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="eps")
        fld_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="fld")
        eps_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="eps")

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

        D_fwd = E_to_D(fld_fwd, eps_fwd)
        D_adj = E_to_D(fld_adj, eps_fwd)

        # compute the derivatives for this structure
        structure = sim_data_fwd.simulation.structures[structure_index]

        # compute epsilon arrays for all frequencies
        adjoint_frequencies = np.array(fld_adj.monitor.freqs)

        eps_in = _compute_eps_array(structure.medium, adjoint_frequencies)
        eps_out = _compute_eps_array(sim_data_orig.simulation.medium, adjoint_frequencies)

        # handle background medium if present
        if structure.background_medium:
            eps_background = _compute_eps_array(structure.background_medium, adjoint_frequencies)
        else:
            eps_background = None

        # auto permittivity detection for non-box geometries
        if not isinstance(structure.geometry, td.Box):
            sim_orig = sim_data_orig.simulation
            plane_eps = eps_fwd.monitor.geometry

            sim_orig_grid_spec = td.components.grid.grid_spec.GridSpec.from_grid(sim_orig.grid)

            # permittivity without this structure
            structs_no_struct = list(sim_orig.structures)
            structs_no_struct.pop(structure_index)
            sim_no_structure = sim_orig.updated_copy(
                structures=structs_no_struct, monitors=[], sources=[], grid_spec=sim_orig_grid_spec
            )

            eps_no_structure_data = [
                sim_no_structure.epsilon(box=plane_eps, coord_key="centers", freq=f)
                for f in adjoint_frequencies
            ]

            eps_no_structure = xr.concat(eps_no_structure_data, dim="f").assign_coords(
                f=adjoint_frequencies
            )

            if structure.medium.is_pec:
                eps_inf_structure = None
            else:
                # permittivity with infinite structure
                structs_inf_struct = list(sim_orig.structures)[structure_index + 1 :]
                sim_inf_structure = sim_orig.updated_copy(
                    structures=structs_inf_struct,
                    medium=structure.medium,
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
        else:
            eps_no_structure = eps_inf_structure = None

        # compute bounds intersection
        struct_bounds = rmin_struct, rmax_struct = structure.geometry.bounds
        rmin_sim, rmax_sim = sim_data_orig.simulation.bounds
        rmin_intersect = tuple([max(a, b) for a, b in zip(rmin_sim, rmin_struct)])
        rmax_intersect = tuple([min(a, b) for a, b in zip(rmax_sim, rmax_struct)])
        bounds_intersect = (rmin_intersect, rmax_intersect)

        # get chunk size - if None, process all frequencies as one chunk
        freq_chunk_size = ADJOINT_FREQ_CHUNK_SIZE
        n_freqs = len(adjoint_frequencies)
        if freq_chunk_size is None:
            freq_chunk_size = n_freqs

        # process in chunks
        vjp_value_map = {}

        for chunk_start in range(0, n_freqs, freq_chunk_size):
            chunk_end = min(chunk_start + freq_chunk_size, n_freqs)
            freq_slice = slice(chunk_start, chunk_end)

            select_adjoint_freqs = adjoint_frequencies[freq_slice]

            # slice field data for current chunk
            E_der_map_chunk = _slice_field_data(E_der_map.field_components, select_adjoint_freqs)
            D_der_map_chunk = _slice_field_data(D_der_map.field_components, select_adjoint_freqs)
            E_fwd_chunk = _slice_field_data(
                fld_fwd.field_components, select_adjoint_freqs, component_indicator="E"
            )
            E_adj_chunk = _slice_field_data(
                fld_adj.field_components, select_adjoint_freqs, component_indicator="E"
            )
            D_fwd_chunk = _slice_field_data(D_fwd.field_components, select_adjoint_freqs)
            D_adj_chunk = _slice_field_data(D_adj.field_components, select_adjoint_freqs)
            eps_data_chunk = _slice_field_data(eps_fwd.field_components, select_adjoint_freqs)

            H_der_map_chunk = None
            H_fwd_chunk = None
            H_adj_chunk = None

            if H_info_exists:
                H_der_map_chunk = _slice_field_data(
                    H_der_map.field_components, select_adjoint_freqs
                )
                H_fwd_chunk = _slice_field_data(
                    fld_fwd.field_components, select_adjoint_freqs, component_indicator="H"
                )
                H_adj_chunk = _slice_field_data(
                    fld_adj.field_components, select_adjoint_freqs, component_indicator="H"
                )

            # slice epsilon arrays
            eps_in_chunk = eps_in.sel(f=select_adjoint_freqs)
            eps_out_chunk = eps_out.sel(f=select_adjoint_freqs)
            eps_background_chunk = (
                eps_background.sel(f=select_adjoint_freqs) if eps_background is not None else None
            )
            eps_no_structure_chunk = (
                eps_no_structure.sel(f=select_adjoint_freqs)
                if eps_no_structure is not None
                else None
            )
            eps_inf_structure_chunk = (
                eps_inf_structure.sel(f=select_adjoint_freqs)
                if eps_inf_structure is not None
                else None
            )

            # create derivative info with sliced data
            derivative_info = DerivativeInfo(
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
                eps_in=eps_in_chunk,
                eps_out=eps_out_chunk,
                eps_background=eps_background_chunk,
                frequencies=select_adjoint_freqs,  # only chunk frequencies
                eps_no_structure=eps_no_structure_chunk,
                eps_inf_structure=eps_inf_structure_chunk,
                bounds=struct_bounds,
                bounds_intersect=bounds_intersect,
                simulation_bounds=sim_data_orig.simulation.bounds,
                is_medium_pec=structure.medium.is_pec,
            )

            # compute derivatives for chunk
            vjp_chunk = structure._compute_derivatives(derivative_info)

            # accumulate results
            for path, value in vjp_chunk.items():
                if path in vjp_value_map:
                    val = vjp_value_map[path]
                    if isinstance(val, (list, tuple)) and isinstance(value, (list, tuple)):
                        vjp_value_map[path] = type(val)(x + y for x, y in zip(val, value))
                    else:
                        vjp_value_map[path] += value
                else:
                    vjp_value_map[path] = value

        # store vjps in output map
        for structure_path, vjp_value in vjp_value_map.items():
            sim_path = ("structures", structure_index, *list(structure_path))
            sim_fields_vjp[sim_path] = vjp_value

    return sim_fields_vjp
