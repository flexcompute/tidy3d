from __future__ import annotations

import typing
from collections import defaultdict

import numpy as np
import xarray as xr

import tidy3d as td
from tidy3d import Medium
from tidy3d.components.autograd import AutogradFieldMap, NumericalStructureInfo, get_static
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.data.data_array import DataArray, FreqDataArray, ScalarFieldDataArray
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.utils import GeometryType
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.packaging import disable_local_subpixel

from .types import (
    UserVJPConfig,
)
from .utils import E_to_D, get_derivative_maps

if typing.TYPE_CHECKING:
    pass


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


def _compute_eps_array(medium: Medium, frequencies: list[float]) -> DataArray:
    """Compute permittivity array for all frequencies."""
    eps_data = [np.mean(medium.eps_model(f)) for f in frequencies]
    return DataArray(data=np.array(eps_data), dims=("f",), coords={"f": frequencies})


def _slice_field_data(
    field_data: dict, freqs: np.ndarray, component_indicator: str | None = None
) -> dict:
    """Slice field data dictionary along frequency dimension."""
    if component_indicator:
        return {k: v.sel(f=freqs) for k, v in field_data.items() if component_indicator in k}
    else:
        return {k: v.sel(f=freqs) for k, v in field_data.items()}


@disable_local_subpixel
def postprocess_adj(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_keys: list[tuple],
    user_vjp: tuple[UserVJPConfig],
    numerical_info: dict[int, NumericalStructureInfo],
) -> AutogradFieldMap:
    """Postprocess some data from the adjoint simulation into the VJP for the original sim flds."""

    # prepare lookup for user-provided VJPs keyed by structure and field entry

    ####

    # here is where we can decide if we are using the vjp for all entries or not
    # we might want to do some checking on the user_vjp to make sure we don't have collisions
    # runtime validation of it

    ####

    # todo: fix this return typing
    def get_all_paths(match_structure_index: int) -> tuple[str, ...]:
        all_paths = tuple(
            tuple(structure_path)
            for namespace, structure_index, *structure_path in sim_fields_keys
            if structure_index == match_structure_index
        )

        return all_paths

    user_vjp_lookup: dict[int, dict[typing.Hashable, typing.Callable[..., typing.Any]]] = {}
    if user_vjp:
        for vjp_config in user_vjp:
            structure_index = vjp_config.structure_index
            vjp_fn = vjp_config.compute_derivatives
            path = vjp_config.path_key

            if path is None:
                for match_path in get_all_paths(structure_index):
                    user_vjp_lookup.setdefault(structure_index, {})[match_path[0:2]] = vjp_fn
            else:
                user_vjp_lookup.setdefault(structure_index, {})[path] = vjp_fn

    # map of index into 'structures' and 'numerical' to the paths we need VJPs for
    sim_vjp_map = defaultdict(list)
    numerical_vjp_map = defaultdict(set)
    for namespace, structure_index, *structure_path in sim_fields_keys:
        structure_path = tuple(structure_path)
        if namespace == "structures":
            sim_vjp_map[structure_index].append(structure_path)
        elif namespace == "numerical":
            numerical_vjp_map[structure_index].add(structure_path)

    # store the derivative values given the forward and adjoint data
    sim_fields_vjp = {}
    all_structure_indices = sorted(set(sim_vjp_map.keys()) | set(numerical_vjp_map.keys()))

    for structure_index in all_structure_indices:
        structure_paths = tuple(sim_vjp_map.get(structure_index, ()))
        numerical_paths_raw = numerical_vjp_map.get(structure_index, set())
        numerical_paths_ordered: tuple[tuple, ...] = ()
        numerical_value_map: dict[tuple, typing.Any] = {}
        numerical_vjp_fn = None
        numerical_params_static: tuple[typing.Any, ...] = ()

        if numerical_paths_raw:
            info = numerical_info.get(structure_index)
            if info is None:
                raise AdjointError(
                    f"Missing numerical structure metadata for index {structure_index}."
                )
            numerical_vjp_fn = info.vjp
            numerical_params_static = tuple(get_static(param) for param in info.parameters)
            numerical_paths_ordered = tuple(sorted(numerical_paths_raw))

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

        def updated_epsilon_full(
            replacement_geometry: GeometryType,
            adjoint_frequencies: typing.Optional[FreqDataArray] = adjoint_frequencies,
            structure_index: typing.Optional[int] = structure_index,
            eps_box: typing.Optional[Box] = eps_fwd.monitor.geometry,
        ) -> ScalarFieldDataArray:
            # Return the simulation permittivity for eps_box after replacing the geometry
            # for this structure with a new geometry. This is helpful for carrying out finite
            # difference permittivity computations
            sim_orig = sim_data_orig.simulation
            sim_orig_grid_spec = td.components.grid.grid_spec.GridSpec.from_grid(sim_orig.grid)

            update_sim = sim_orig.updated_copy(
                structures=[
                    sim_orig.structures[idx].updated_copy(geometry=replacement_geometry)
                    if idx == structure_index
                    else sim_orig.structures[idx]
                    for idx in range(len(sim_orig.structures))
                ],
                grid_spec=sim_orig_grid_spec,
            )

            eps_by_f = [
                update_sim.epsilon(box=eps_box, coord_key="centers", freq=f)
                for f in adjoint_frequencies
            ]

            return xr.concat(eps_by_f, dim="f").assign_coords(f=adjoint_frequencies)

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

            def updated_epsilon(
                replacement_geometry: GeometryType,
                select_adjoint_freqs: typing.Optional[FreqDataArray] = select_adjoint_freqs,
                updated_epsilon_full: typing.Optional[typing.Callable] = updated_epsilon_full,
            ) -> ScalarFieldDataArray:
                # Get permittivity function for a subset of frequencies
                return updated_epsilon_full(replacement_geometry).sel(f=select_adjoint_freqs)

            common_kwargs = {
                "E_der_map": E_der_map_chunk,
                "D_der_map": D_der_map_chunk,
                "H_der_map": H_der_map_chunk,
                "E_fwd": E_fwd_chunk,
                "E_adj": E_adj_chunk,
                "D_fwd": D_fwd_chunk,
                "D_adj": D_adj_chunk,
                "H_fwd": H_fwd_chunk,
                "H_adj": H_adj_chunk,
                "eps_data": eps_data_chunk,
                "eps_in": eps_in_chunk,
                "eps_out": eps_out_chunk,
                "eps_background": eps_background_chunk,
                "frequencies": select_adjoint_freqs,
                "eps_no_structure": eps_no_structure_chunk,
                "eps_inf_structure": eps_inf_structure_chunk,
                "updated_epsilon": updated_epsilon,
                "bounds": struct_bounds,
                "bounds_intersect": bounds_intersect,
                "simulation_bounds": sim_data_orig.simulation.bounds,
                "is_medium_pec": structure.medium.is_pec,
            }

            if structure_paths:
                derivative_info_struct = DerivativeInfo(
                    paths=structure_paths,
                    **common_kwargs,
                )

                vjp_fns = user_vjp_lookup.get(structure_index)
                vjp_chunk = structure._compute_derivatives(derivative_info_struct, vjp_fns=vjp_fns)

                for path, value in vjp_chunk.items():
                    if path in vjp_value_map:
                        existing = vjp_value_map[path]
                        if isinstance(existing, (list, tuple)) and isinstance(value, (list, tuple)):
                            vjp_value_map[path] = type(existing)(
                                x + y for x, y in zip(existing, value)
                            )
                        else:
                            vjp_value_map[path] = existing + value
                    else:
                        vjp_value_map[path] = value

            if numerical_paths_ordered and numerical_vjp_fn is not None:
                derivative_info_num = DerivativeInfo(
                    paths=numerical_paths_ordered,
                    **common_kwargs,
                )

                gradients = numerical_vjp_fn(
                    parameters=numerical_params_static, derivative_info=derivative_info_num
                )

                if isinstance(gradients, dict):
                    gradient_items = (
                        (path, gradients.get(path)) for path in numerical_paths_ordered
                    )
                else:
                    gradients_seq = tuple(gradients)
                    if len(gradients_seq) != len(numerical_paths_ordered):
                        raise AdjointError(
                            f"User VJP for numerical structure index {structure_index} returned {len(gradients_seq)} gradients, "
                            f"expected {len(numerical_paths_ordered)}."
                        )
                    gradient_items = zip(numerical_paths_ordered, gradients_seq)

                for path, grad_value in gradient_items:
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

        # store vjps in output map
        for structure_path, vjp_value in vjp_value_map.items():
            sim_path = ("structures", structure_index, *list(structure_path))
            sim_fields_vjp[sim_path] = vjp_value

        for numerical_path, gradient_value in numerical_value_map.items():
            sim_path = ("numerical", structure_index, *list(numerical_path))
            sim_fields_vjp[sim_path] = gradient_value

    return sim_fields_vjp
