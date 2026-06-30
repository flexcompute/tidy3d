"""Shared helpers for electromagnetic field components."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, get_args

from tidy3d.components.types import EMField, PointCloudFieldComponent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .data_array import DataArray

EM_FIELD_COMPONENTS = cast(tuple[EMField, ...], get_args(EMField))
POINT_CLOUD_FIELD_COMPONENTS = cast(
    tuple[PointCloudFieldComponent, ...], get_args(PointCloudFieldComponent)
)


def field_symmetry_eigenvalue(field: str, dim: int) -> int:
    """Positive field-component eigenvalue under reflection in a coordinate dimension."""
    component_axis = "xyz".index(field[-1])
    if field[0] in ("E", "D"):
        return -1 if component_axis == dim else 1
    return 1 if component_axis == dim else -1


def em_field_symmetry_eigenvalues() -> dict[str, Callable[[int], int]]:
    """Return symmetry eigenvalue functions for all E/H field components."""
    return {
        field: lambda dim, field=field: field_symmetry_eigenvalue(field, dim)
        for field in EM_FIELD_COMPONENTS
    }


def point_cloud_field_symmetry_eigenvalues() -> dict[str, Callable[[int], int]]:
    """Return symmetry eigenvalue functions for point-cloud E/H/D field components."""
    return {
        field: lambda dim, field=field: field_symmetry_eigenvalue(field, dim)
        for field in POINT_CLOUD_FIELD_COMPONENTS
    }


def frequency_normalized_field_components(
    field_components: Mapping[str, DataArray], source_spectrum_fn: Callable[[float], complex]
) -> dict[str, DataArray]:
    """Return frequency-domain field components normalized by a source spectrum."""
    fields_norm = {}
    for field_name, field_data in field_components.items():
        src_amps = source_spectrum_fn(field_data.f)
        fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

    return fields_norm
