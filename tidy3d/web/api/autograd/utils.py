# utility functions for autograd web API
from __future__ import annotations

import typing

import numpy as np

import tidy3d as td

""" E and D field gradient map calculation helpers. """


def get_derivative_maps(
    fld_fwd: td.FieldData,
    eps_fwd: td.PermittivityData,
    fld_adj: td.FieldData,
    eps_adj: td.PermittivityData,
) -> dict[str, td.FieldData]:
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
    fld_1: td.FieldData, fld_2: typing.Union[td.FieldData, td.PermittivityData], fld_key: str
) -> td.FieldData:
    """Elementwise multiply two field data objects, writes data into ``fld_1`` copy."""

    def get_field_key(dim: str, fld_data: typing.Union[td.FieldData, td.PermittivityData]) -> str:
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
