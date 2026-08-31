# utility functions for autograd web API
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import tidy3d as td
from tidy3d.components.autograd import get_static
from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.data.data_array import DataArray

    from .types import CustomVJPConfig

""" E and D field gradient map calculation helpers. """


def scale_field_data(
    fld_data: td.FieldData,
    scale: float | complex | DataArray,
) -> td.FieldData:
    """Scale all field components in a ``FieldData`` object."""

    field_components = {
        name: component * scale for name, component in fld_data.field_components.items()
    }
    return fld_data.updated_copy(**field_components)


def get_derivative_maps(
    fld_fwd: td.FieldData,
    eps_fwd: td.PermittivityData,
    fld_adj: td.FieldData,
    eps_adj: td.PermittivityData,
) -> dict[str, td.FieldData | None]:
    """Get electric and displacement field derivative maps."""
    der_map_E = derivative_map_E(fld_fwd=fld_fwd, fld_adj=fld_adj)
    der_map_D = derivative_map_D(fld_fwd=fld_fwd, eps_fwd=eps_fwd, fld_adj=fld_adj, eps_adj=eps_adj)

    make_H_der_map = np.all([f"H{dim}" in fld_fwd.field_components for dim in "xyz"])
    der_map_H = None
    if make_H_der_map:
        der_map_H = derivative_map_H(fld_fwd=fld_fwd, fld_adj=fld_adj)

    return {"E": der_map_E, "D": der_map_D, "H": der_map_H}


def derivative_map_E(fld_fwd: td.FieldData, fld_adj: td.FieldData) -> td.FieldData:
    """Get td.FieldData where the Ex, Ey, Ez components store the gradients w.r.t. these."""
    return multiply_field_data(fld_fwd, fld_adj, fld_key="E")


def derivative_map_H(fld_fwd: td.FieldData, fld_adj: td.FieldData) -> td.FieldData:
    """Get td.FieldData where the Hx, Hy, Hz components store the gradients w.r.t. these."""
    return multiply_field_data(fld_fwd, fld_adj, fld_key="H")


def derivative_map_D(
    fld_fwd: td.FieldData,
    eps_fwd: td.PermittivityData,
    fld_adj: td.FieldData,
    eps_adj: td.PermittivityData,
) -> td.FieldData:
    """Get td.FieldData where the Ex, Ey, Ez components store the gradients w.r.t. D fields."""
    fwd_D = E_to_D(fld_data=fld_fwd, eps_data=eps_fwd)
    adj_D = E_to_D(fld_data=fld_adj, eps_data=eps_adj)

    return multiply_field_data(fwd_D, adj_D, fld_key="E")


def E_to_D(fld_data: td.FieldData, eps_data: td.PermittivityData) -> td.FieldData:
    """Convert electric field to displacement field."""

    return multiply_field_data(fld_data, eps_data, fld_key="E")


def multiply_field_data(
    fld_1: td.FieldData, fld_2: td.FieldData | td.PermittivityData, fld_key: str
) -> td.FieldData:
    """Elementwise multiply two field data objects, writes data into ``fld_1`` copy."""

    def get_field_key(dim: str, fld_data: td.FieldData | td.PermittivityData) -> str:
        """Get the key corresponding to the scalar field along this dimension."""
        return f"{fld_key}{dim}" if isinstance(fld_data, td.FieldData) else f"eps_{dim}{dim}"

    field_components = {}
    for dim in "xyz":
        key_1 = get_field_key(dim=dim, fld_data=fld_1)
        key_2 = get_field_key(dim=dim, fld_data=fld_2)
        cmp_1 = fld_1.field_components[key_1]
        cmp_2 = fld_2.field_components[key_2]
        mult = cmp_1 * cmp_2
        field_components[key_1] = mult
    return fld_1.updated_copy(**field_components)


def filter_vjp_map(data_fields_vjp: AutogradFieldMap) -> AutogradFieldMap:
    """Filter VJP map to static, nonzero entries and validate NaNs."""
    data_fields_vjp_static = {}
    for k, v in data_fields_vjp.items():
        v_static = get_static(v)
        if np.count_nonzero(v_static) == 0:
            continue
        if np.any(np.isnan(v_static)):
            raise AdjointError(
                f"NaN values detected for data field {k} in the adjoint pipeline. "
                "This may be due to NaN values in the simulation data or the computed "
                "value of your objective function."
            )
        data_fields_vjp_static[k] = v_static
    return data_fields_vjp_static


def zero_vjp_map(sim_fields_original: AutogradFieldMap) -> AutogradFieldMap:
    """Return a VJP map with zero values matching ``sim_fields_original`` types/shapes."""
    return {
        key: (type(value)(0 * x for x in value) if isinstance(value, (list, tuple)) else 0 * value)
        for key, value in sim_fields_original.items()
    }


def expand_custom_vjp_configs(
    custom_vjp: Sequence[CustomVJPConfig] | None,
    sim_fields_keys: list[tuple],
) -> dict[int, dict[tuple[str, str], Callable[..., Any]]]:
    """Expand custom-vjp configs into per-structure ``(med_or_geo, path head)`` handlers.

    A config with ``path_key=None`` claims every traced path of its structure.
    """

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

    return custom_vjp_lookup


def custom_vjp_geometry_exclusions(
    custom_vjp_lookup: dict[int, dict[tuple[str, str], Callable[..., Any]]],
) -> dict[int, tuple[tuple[str], ...]]:
    """Geometry path-head exclusions implied by custom vjps, as 1-tuple prefixes.

    Sample-set collection and coverage validation both take these exclusions, so
    custom-vjp-owned paths are neither collected for nor demanded of the artifact.
    """
    exclusions = {}
    for structure_index, vjp_fns in custom_vjp_lookup.items():
        heads = tuple((key[1],) for key in vjp_fns if key[0] == "geometry")
        if heads:
            exclusions[structure_index] = heads
    return exclusions
