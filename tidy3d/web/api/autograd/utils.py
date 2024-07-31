# utility functions for autograd web API
from __future__ import annotations

import typing

import pydantic as pd

import tidy3d as td
from tidy3d.components.autograd.types import AutogradFieldMap, dict_ag
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayLike, tidycomplex

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
    return dict(E=der_map_E, D=der_map_D)


def derivative_map_E(fld_fwd: td.FieldData, fld_adj: td.FieldData) -> td.FieldData:
    """Get td.FieldData where the Ex, Ey, Ez components store the gradients w.r.t. these."""
    return multiply_field_data(fld_fwd, fld_adj)


def derivative_map_D(
    fld_fwd: td.FieldData,
    eps_fwd: td.PermittivityData,
    fld_adj: td.FieldData,
    eps_adj: td.PermittivityData,
) -> td.FieldData:
    """Get td.FieldData where the Ex, Ey, Ez components store the gradients w.r.t. D fields."""
    fwd_D = E_to_D(fld_data=fld_fwd, eps_data=eps_fwd)
    adj_D = E_to_D(fld_data=fld_adj, eps_data=eps_adj)
    return multiply_field_data(fwd_D, adj_D)


def E_to_D(fld_data: td.FieldData, eps_data: td.PermittivityData) -> td.FieldData:
    """Convert electric field to displacement field."""
    return multiply_field_data(fld_data, eps_data)


def multiply_field_data(
    fld_1: td.FieldData, fld_2: typing.Union[td.FieldData, td.PermittivityData]
) -> td.FieldData:
    """Elementwise multiply two field data objects, writes data into ``fld_1`` copy."""

    def get_field_key(dim: str, fld_data: typing.Union[td.FieldData, td.PermittivityData]) -> str:
        """Get the key corresponding to the scalar field along this dimension."""
        return f"E{dim}" if isinstance(fld_data, td.FieldData) else f"eps_{dim}{dim}"

    field_components = {}
    for dim in "xyz":
        key_1 = get_field_key(dim=dim, fld_data=fld_1)
        key_2 = get_field_key(dim=dim, fld_data=fld_2)
        cmp_1 = fld_1.field_components[key_1]
        cmp_2 = fld_2.field_components[key_2]
        mult = cmp_1 * cmp_2
        field_components[key_1] = mult
    return fld_1.updated_copy(**field_components)


class Tracer(Tidy3dBaseModel):
    """Class to store a single traced field."""

    path: tuple[typing.Any, ...] = pd.Field(
        ...,
        title="Path to the traced object in the model dictionary.",
    )

    data: typing.Union[float, tidycomplex, ArrayLike] = pd.Field(..., title="Tracing data")


class FieldMap(Tidy3dBaseModel):
    """Class to store a collection of traced fields."""

    tracers: tuple[Tracer, ...] = pd.Field(
        ...,
        title="Collection of Tracers.",
    )

    @property
    def to_autograd_field_map(self) -> AutogradFieldMap:
        """Convert to ``AutogradFieldMap`` autograd dictionary."""
        return dict_ag({tracer.path: tracer.data for tracer in self.tracers})

    @classmethod
    def from_autograd_field_map(cls, autograd_field_map) -> FieldMap:
        """Initialize from an ``AutogradFieldMap`` autograd dictionary."""
        tracers = []
        for path, data in autograd_field_map.items():
            tracers.append(Tracer(path=path, data=data))

        return cls(tracers=tuple(tracers))


class TracerKeys(Tidy3dBaseModel):
    """Class to store a collection of tracer keys."""

    keys: tuple[tuple[typing.Any, ...], ...] = pd.Field(
        ...,
        title="Collection of tracer keys.",
    )
