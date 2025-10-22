from __future__ import annotations

import typing
from collections import defaultdict

import numpy as np
import xarray as xr

import tidy3d as td
from tidy3d import Medium
from tidy3d.components.autograd import AutogradFieldMap, NumericalStructureInfo, get_static
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.data.data_array import DataArray
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.packaging import disable_local_subpixel

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
    user_vjp,
    numerical_info: dict[int, NumericalStructureInfo],
) -> AutogradFieldMap:
    """Postprocess some data from the adjoint simulation into the VJP for the original sim flds."""

    # map of index into 'structures' to the list of paths we need vjps for
    sim_vjp_map = defaultdict(list)
    numerical_vjp_map = defaultdict(set)

    for namespace, structure_index, *structure_path in sim_fields_keys:
        if namespace == "structures":
            sim_vjp_map[structure_index].append(tuple(structure_path))
        elif namespace == "numerical":
            numerical_vjp_map[structure_index].add(tuple(structure_path))

    structure_data_cache: dict[int, dict[str, typing.Any]] = {}

    def get_structure_data(structure_index: int) -> dict[str, typing.Any]:
        if structure_index in structure_data_cache:
            return structure_data_cache[structure_index]

        structure = sim_data_fwd.simulation.structures[structure_index]
        fld_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="fld")
        eps_fwd = sim_data_fwd._get_adjoint_data(structure_index, data_type="eps")
        fld_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="fld")
        eps_adj = sim_data_adj._get_adjoint_data(structure_index, data_type="eps")

        fwd_flds_adj_normed = {
            key: val * sim_data_adj.simulation.post_norm
            for key, val in fld_adj.field_components.items()
        }
        fld_adj_normed = fld_adj.updated_copy(**fwd_flds_adj_normed)

        der_maps = get_derivative_maps(
            fld_fwd=fld_fwd,
            eps_fwd=eps_fwd,
            fld_adj=fld_adj_normed,
            eps_adj=eps_adj,
        )
        E_der_map = der_maps["E"]
        D_der_map = der_maps["D"]
        H_der_map = der_maps["H"]

        H_info_exists = H_der_map is not None

        D_fwd = E_to_D(fld_fwd, eps_fwd)
        D_adj = E_to_D(fld_adj_normed, eps_fwd)
        H_fwd_full = {
            comp: fld_fwd.field_components[comp]
            for comp in fld_fwd.field_components
            if comp.startswith("H")
        }
        H_adj_full = {
            comp: fld_adj_normed.field_components[comp]
            for comp in fld_adj_normed.field_components
            if comp.startswith("H")
        }

        adjoint_frequencies = np.array(fld_adj.monitor.freqs)

        eps_in = _compute_eps_array(structure.medium, adjoint_frequencies)
        eps_out = _compute_eps_array(sim_data_orig.simulation.medium, adjoint_frequencies)

        if structure.background_medium:
            eps_background = _compute_eps_array(structure.background_medium, adjoint_frequencies)
        else:
            eps_background = None

        if not isinstance(structure.geometry, td.Box):
            sim_orig = sim_data_orig.simulation
            plane_eps = eps_fwd.monitor.geometry

            sim_orig_grid_spec = td.components.grid.grid_spec.GridSpec.from_grid(sim_orig.grid)

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

        # struct_bounds = rmin_struct, rmax_struct = structure.geometry.bounds
        rmin_struct, rmax_struct = structure.geometry.bounds
        rmin_sim, rmax_sim = sim_data_orig.simulation.bounds
        rmin_intersect = tuple(max(a, b) for a, b in zip(rmin_sim, rmin_struct))
        rmax_intersect = tuple(min(a, b) for a, b in zip(rmax_sim, rmax_struct))
        bounds_intersect = (rmin_intersect, rmax_intersect)

        def updated_epsilon_full(replacement_geometry):
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
                update_sim.epsilon(box=eps_fwd.monitor.geometry, coord_key="centers", freq=f)
                for f in adjoint_frequencies
            ]

            return xr.concat(eps_by_f, dim="f").assign_coords(f=adjoint_frequencies)

        data = {
            "structure": structure,
            "fld_fwd": fld_fwd,
            "eps_fwd": eps_fwd,
            "fld_adj": fld_adj_normed,
            "eps_adj": eps_adj,
            "E_der_map": E_der_map,
            "D_der_map": D_der_map,
            "H_der_map": H_der_map,
            "H_info_exists": H_info_exists,
            "D_fwd": D_fwd,
            "D_adj": D_adj,
            "H_fwd_full": H_fwd_full,
            "H_adj_full": H_adj_full,
            "adjoint_frequencies": adjoint_frequencies,
            "eps_in": eps_in,
            "eps_out": eps_out,
            "eps_background": eps_background,
            "eps_no_structure": eps_no_structure,
            "eps_inf_structure": eps_inf_structure,
            "updated_epsilon_full": updated_epsilon_full,
            "struct_bounds": (rmin_struct, rmax_struct),
            "bounds_intersect": bounds_intersect,
            "simulation_bounds": sim_data_orig.simulation.bounds,
        }

        structure_data_cache[structure_index] = data
        return data

    all_structure_indices = sorted(set(sim_vjp_map.keys()) | set(numerical_vjp_map.keys()))

    def _accumulate(existing, new):
        if existing is None:
            return new
        if isinstance(existing, (list, tuple)) and isinstance(new, (list, tuple)):
            return type(existing)(x + y for x, y in zip(existing, new))
        return existing + new

    def _zero_like(value):
        if isinstance(value, (list, tuple)):
            return type(value)(_zero_like(v) for v in value)
        return 0 * value

    # store the derivative values given the forward and adjoint data
    sim_fields_vjp = {}

    for structure_index in all_structure_indices:
        structure_paths = sim_vjp_map.get(structure_index, [])
        numerical_paths_raw = numerical_vjp_map.get(structure_index, set())

        data = get_structure_data(structure_index)

        fld_fwd = data["fld_fwd"]
        eps_fwd = data["eps_fwd"]
        fld_adj = data["fld_adj"]
        # eps_adj = data["eps_adj"]
        E_der_map = data["E_der_map"]
        D_der_map = data["D_der_map"]
        H_der_map = data["H_der_map"]
        H_info_exists = data["H_info_exists"]
        D_fwd = data["D_fwd"]
        D_adj = data["D_adj"]
        adjoint_frequencies = data["adjoint_frequencies"]
        eps_in = data["eps_in"]
        eps_out = data["eps_out"]
        eps_background = data["eps_background"]
        eps_no_structure = data["eps_no_structure"]
        eps_inf_structure = data["eps_inf_structure"]
        struct_bounds = data["struct_bounds"]
        bounds_intersect = data["bounds_intersect"]
        simulation_bounds = data["simulation_bounds"]
        updated_epsilon_full = data["updated_epsilon_full"]

        # get chunk size - if None, process all frequencies as one chunk
        freq_chunk_size = config.adjoint.solver_freq_chunk_size
        n_freqs = len(adjoint_frequencies)
        if not freq_chunk_size or freq_chunk_size <= 0:
            freq_chunk_size = n_freqs
        else:
            freq_chunk_size = min(freq_chunk_size, n_freqs)

        # process in chunks
        vjp_value_map = {}
        numerical_accum = None
        numerical_paths_ordered = ()
        vjp_fn = None
        info = None
        if numerical_paths_raw:
            if user_vjp is None:
                raise AdjointError("Numerical structures detected but no 'user_vjp' provided.")
            info = numerical_info.get(structure_index)
            if info is None:
                raise AdjointError(
                    f"Missing numerical structure metadata for index {structure_index}."
                )

            vjp_fn_entry = user_vjp.get(structure_index)
            if vjp_fn_entry is None:
                raise AdjointError(
                    f"Missing user VJP for numerical structure index {structure_index}."
                )

            if callable(vjp_fn_entry):
                vjp_fn = vjp_fn_entry
            elif isinstance(vjp_fn_entry, dict):
                if "parameters" in vjp_fn_entry and callable(vjp_fn_entry["parameters"]):
                    vjp_fn = vjp_fn_entry["parameters"]
                else:
                    callables = [val for val in vjp_fn_entry.values() if callable(val)]
                    if len(callables) != 1:
                        raise AdjointError(
                            f"Numerical structure index {structure_index} requires exactly one callable in its user VJP entry."
                        )
                    vjp_fn = callables[0]
            else:
                raise AdjointError(
                    f"Invalid user VJP entry for numerical structure index {structure_index}."
                )

            name_to_path = {path[0]: path for path in numerical_paths_raw if path}
            try:
                numerical_paths_ordered = tuple(name_to_path[name] for name in info.parameter_names)
            except KeyError as exc:  # pragma: no cover - defensive
                raise AdjointError(
                    f"Numerical structure index {structure_index} missing VJP path for parameter '{exc.args[0]}'."
                ) from exc
            numerical_accum = [None] * len(info.parameters)

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
                replacement_geometry,
                select_adjoint_freqs=select_adjoint_freqs,
                updated_epsilon_full=updated_epsilon_full,
            ):
                return updated_epsilon_full(replacement_geometry).sel(f=select_adjoint_freqs)

            # create derivative info with sliced data
            chunk_kwargs = {
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
                "updated_epsilon": updated_epsilon,
                "eps_no_structure": eps_no_structure_chunk,
                "eps_inf_structure": eps_inf_structure_chunk,
                "bounds": struct_bounds,
                "bounds_intersect": bounds_intersect,
                "simulation_bounds": simulation_bounds,
                "is_medium_pec": data["structure"].medium.is_pec,
            }

            if structure_paths:
                derivative_info_struct = DerivativeInfo(
                    paths=tuple(structure_paths),
                    **chunk_kwargs,
                )

                vjp_fns = None
                if (user_vjp is not None) and (structure_index in user_vjp):
                    vjp_fns = user_vjp[structure_index]

                vjp_chunk = data["structure"]._compute_derivatives(
                    derivative_info_struct, vjp_fns=vjp_fns
                )

                for path, value in vjp_chunk.items():
                    if path in vjp_value_map:
                        vjp_value_map[path] = _accumulate(vjp_value_map[path], value)
                    else:
                        vjp_value_map[path] = value

            if numerical_accum is not None:
                derivative_info_num = DerivativeInfo(
                    paths=numerical_paths_ordered,
                    **chunk_kwargs,
                )

                gradients = vjp_fn(parameters=info.parameters, derivative_info=derivative_info_num)

                if len(gradients) != len(info.parameters):
                    raise AdjointError(
                        f"User VJP for numerical structure index {structure_index} returned {len(gradients)} gradients, "
                        f"expected {len(info.parameters)}."
                    )

                print(f"gradients = {gradients}")
                for idx, grad_key in enumerate(gradients):
                    print(f"grad = {gradients[grad_key]}")
                    numerical_accum[idx] = _accumulate(numerical_accum[idx], gradients[grad_key])

        for structure_path, vjp_value in vjp_value_map.items():
            sim_path = ("structures", structure_index, *list(structure_path))
            sim_fields_vjp[sim_path] = vjp_value

        print(f"numerical accum = {numerical_accum}")

        if numerical_accum is not None:
            for name, grad, param in zip(info.parameter_names, numerical_accum, info.parameters):
                if grad is None:
                    print(f"ok = {param}")
                    grad = _zero_like(param)

                print(f"name = {name}")
                print(f"param = {param}")
                print(f"numerical accum and grad = {grad}")
                sim_fields_vjp[("numerical", structure_index, name)] = grad

    return sim_fields_vjp
