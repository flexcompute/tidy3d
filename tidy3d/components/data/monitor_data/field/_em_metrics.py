from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from flex_em.numerical.raw import field_data as field_data_numerics

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import DataArray, FluxDataArray, FreqModeDataArray
from tidy3d.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from tidy3d.components.data.data_array import ScalarFieldDataArray
    from tidy3d.components.data.monitor_data.field.electromagnetic import ElectromagneticFieldData
    from tidy3d.components.geometry.base import Box


@property
def intensity(self: ElectromagneticFieldData) -> ScalarFieldDataArray:
    """Return the sum of the squared absolute electric field components."""
    return self.field_intensity()


def field_intensity(
    self: ElectromagneticFieldData, components: str | Sequence[str] = ("Ex", "Ey", "Ez")
) -> ScalarFieldDataArray:
    """Return the sum of the squared absolute selected electric field components.

    The selected components must be stored in the data, and the intensity is evaluated using
    the same symmetry-expanded and colocated fields as :attr:`intensity`. For non-colocated
    monitors, fields are interpolated to the colocated grid before they are summed.
    """
    components = (components,) if isinstance(components, str) else tuple(components)
    if not components:
        raise ValueError("At least one electric field component must be specified.")

    valid_components = ("Ex", "Ey", "Ez")
    invalid_components = tuple(cmp for cmp in components if cmp not in valid_components)
    if invalid_components:
        raise ValueError(
            "Invalid electric field component(s) for intensity: "
            f"{invalid_components}. Valid components are {valid_components}."
        )
    duplicate_components = tuple(
        cmp for idx, cmp in enumerate(components) if cmp in components[:idx]
    )
    if duplicate_components:
        raise ValueError(
            "Duplicate electric field component(s) for intensity: "
            f"{duplicate_components}. Each component may be specified at most once."
        )

    self._check_fields_stored(list(components))
    drop_dims = ["xyz"[dim] for dim in self.monitor.zero_dims]
    fields = self._colocated_fields
    if any(cmp not in fields for cmp in components):
        raise KeyError("Can't compute intensity, all selected E field components must be present.")

    intensity = fields[components[0]].abs ** 2
    for cmp in components[1:]:
        intensity = intensity + fields[cmp].abs ** 2
    return intensity.squeeze(dim=drop_dims, drop=True)


@property
def complex_poynting(self: ElectromagneticFieldData) -> ScalarFieldDataArray:
    """Time-averaged Poynting vector for frequency-domain data associated to a 2D monitor,
    projected to the direction normal to the monitor plane."""

    # Tangential fields are ordered as E1, E2, H1, H2
    tan_fields = self._colocated_tangential_fields
    dim1, dim2 = self._tangential_dims

    e1 = tan_fields["E" + dim1]
    e2 = tan_fields["E" + dim2]
    h1 = tan_fields["H" + dim1]
    h2 = tan_fields["H" + dim2]

    return field_data_numerics.complex_poynting(e1, e2, h1, h2)


@property
def poynting(self: ElectromagneticFieldData) -> ScalarFieldDataArray:
    """Time-averaged Poynting vector for frequency-domain data associated to a 2D monitor,
    projected to the direction normal to the monitor plane."""
    return self.complex_poynting.real


def _prepare_fields_for_flux(
    self: ElectromagneticFieldData,
    fields: dict[str, DataArray],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Make numpy arrays transposed for use in flux calculations.

    Final arrays have spatial dimensions last: (..., Nu, Nv).
    """
    tangential_dims = self._tangential_dims
    test_field = next(iter(fields.values()))

    # Non-spatial dims first (f, optionally mode_index), then spatial
    non_spatial = [d for d in test_field.dims if d not in tangential_dims]
    dim_order = (*non_spatial, *tangential_dims)

    prepped_fields = {key: field.transpose(*dim_order).values for key, field in fields.items()}

    non_spatial_dims = [d for d in test_field.dims if d not in tangential_dims]
    final_coords = {d: test_field.coords[d].values for d in non_spatial_dims}

    return final_coords, prepped_fields


def package_flux_results(self: ElectromagneticFieldData, flux_values: DataArray) -> Any:
    """How to package flux based on the coordinates present in the data."""
    # Choose appropriate data array type based on coordinates
    if "mode_index" in flux_values.dims:
        return FreqModeDataArray(flux_values)
    return FluxDataArray(flux_values)


def _compute_complex_flux(self: ElectromagneticFieldData) -> FluxDataArray | FreqModeDataArray:
    """Compute complex flux."""

    if getattr(self.monitor, "use_colocated_integration", self.monitor.colocate):
        fields = self._colocated_tangential_fields
        dS = self._diff_area.to_numpy()
        dS_numpy = (dS, dS)
    else:
        fields = self._tangential_fields
        dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(truncate_to_monitor_bounds=True)
        dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

    final_coords, prepped_fields = self._prepare_fields_for_flux(fields)

    u, v = self._tangential_dims
    E = (prepped_fields["E" + u], prepped_fields["E" + v])
    H = (prepped_fields["H" + u], prepped_fields["H" + v])

    flux_result = field_data_numerics.complex_power_flow(E, H, dS_numpy)

    if "mode_index" in final_coords:
        return FreqModeDataArray(flux_result, coords=final_coords)
    return FluxDataArray(flux_result, coords=final_coords)


@cached_property
def complex_flux(self: ElectromagneticFieldData) -> FluxDataArray | FreqModeDataArray:
    """Complex flux for data corresponding to a 2D monitor."""
    return self._compute_complex_flux()


@cached_property
def flux(self: ElectromagneticFieldData) -> FluxDataArray | FreqModeDataArray:
    """Flux for data corresponding to a 2D monitor."""
    return self.complex_flux.real


@cached_property
def mode_area(self: ElectromagneticFieldData) -> FreqModeDataArray:
    r"""Effective mode area corresponding to a 2D monitor.

    .. math:

       \frac{\left(\int |E|^2 \, {\rm d}S\right)^2}{\int |E|^4 \, {\rm d}S}
    """
    intensity = self.intensity
    # integrate over the plane
    d_area = self._diff_area
    num = (intensity * d_area).sum(dim=d_area.dims) ** 2
    den = (intensity**2 * d_area).sum(dim=d_area.dims)

    area = num / den
    if hasattr(self.monitor, "mode_spec"):
        area *= np.cos(self.monitor.mode_spec.angle_theta)

    return FreqModeDataArray(area)


def _bounding_box_mask(self: ElectromagneticFieldData, bounding_box: Box) -> DataArray:
    """Create a mask selecting cells whose centers lie within ``bounding_box``."""

    tan_dims = self._tangential_dims
    intensity = self.intensity
    coords = {dim: intensity.coords[dim].values for dim in tan_dims}

    lower, upper = bounding_box.bounds
    axis_indices = ["xyz".index(dim) for dim in tan_dims]

    masks_1d = []
    for dim, axis_idx in zip(tan_dims, axis_indices):
        coord_vals = coords[dim]
        lower_bound = lower[axis_idx]
        upper_bound = upper[axis_idx]
        masks_1d.append((coord_vals >= lower_bound) & (coord_vals <= upper_bound))

    if len(masks_1d) != 2:
        raise DataError("Bounding box masking currently supports planar monitors only.")

    mask_values = (masks_1d[0][:, None] & masks_1d[1][None, :]).astype(float)
    mask = DataArray(mask_values, coords={dim: coords[dim] for dim in tan_dims}, dims=tan_dims)
    return mask


def fill_fraction(self: ElectromagneticFieldData, bounding_box: Box) -> FreqModeDataArray:
    """Return the field-energy fill fraction within ``bounding_box``.
    The fill fraction is defined as the ratio between the integrated field intensity inside
    the tangential projection of the bounding box and the total integrated intensity over the
    monitor plane. The position and extent of the box normal to the monitor are ignored.

    Parameters
    ----------
    bounding_box : Box
        The bounding box used to compute the fill fraction.

    Returns
    -------
    FreqModeDataArray
        Fill fraction values for each frequency and mode index.
    """

    self._check_fields_stored(["Ex", "Ey", "Ez"])

    intensity = self.intensity
    area = self._diff_area
    mask = self._bounding_box_mask(bounding_box)

    weighted_total = (intensity * area).sum(dim=area.dims)
    weighted_box = (intensity * mask * area).sum(dim=area.dims)

    fill_values = (weighted_box / weighted_total.where(weighted_total != 0)).fillna(0.0)

    return FreqModeDataArray(fill_values)


@cached_property
def fill_fraction_box(self: ElectromagneticFieldData) -> FreqModeDataArray:
    """Convenience accessor using the :class:`~tidy3d.Box` defined on ``sort_spec``.

    The position and extent of the box along the monitor's normal axis do not influence the fill
    fraction.
    """

    sort_spec = getattr(self.monitor.mode_spec, "sort_spec", None)
    bounding_box = None if sort_spec is None else sort_spec.bounding_box
    if bounding_box is None:
        raise DataError(
            "ModeSortSpec.bounding_box must be set to access 'fill_fraction_box' metric."
        )
    return self.fill_fraction(bounding_box)
