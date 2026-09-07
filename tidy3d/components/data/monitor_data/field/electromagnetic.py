from __future__ import annotations

from abc import ABC

from pydantic import Field

from tidy3d.components.data.dataset import ElectromagneticFieldDataset
from tidy3d.components.data.monitor_data._types import GRID_CORRECTION_TYPE
from tidy3d.components.data.monitor_data.base import AbstractFieldData

from . import _em_algebra as _algebra_methods
from . import _em_grid as _grid_methods
from . import _em_io as _io_methods
from . import _em_metrics as _metrics_methods


class ElectromagneticFieldData(
    AbstractFieldData,
    ElectromagneticFieldDataset,
    ABC,
):
    """Collection of electromagnetic fields."""

    grid_primal_correction: GRID_CORRECTION_TYPE = Field(
        default=1.0,
        title="Field correction factor",
        description="Correction factor that needs to be applied for data corresponding to a 2D "
        "monitor to take into account the finite grid in the normal direction in the simulation in "
        "which the data was computed. The factor is applied to fields defined on the primal grid "
        "locations along the normal direction.",
    )
    grid_dual_correction: GRID_CORRECTION_TYPE = Field(
        default=1.0,
        title="Field correction factor",
        description="Correction factor that needs to be applied for data corresponding to a 2D "
        "monitor to take into account the finite grid in the normal direction in the simulation in "
        "which the data was computed. The factor is applied to fields defined on the dual grid "
        "locations along the normal direction.",
    )

    # Bind focused electromagnetic behavior directly onto this model.

    # Grid
    _expanded_grid_field_coords = _grid_methods._expanded_grid_field_coords
    _grid_correction_dict = _grid_methods._grid_correction_dict
    _normal_dim = _grid_methods._normal_dim
    _tangential_dims = _grid_methods._tangential_dims
    _diff_area_at_yee_positions = _grid_methods._diff_area_at_yee_positions
    _clamp_grid_expanded_bounds = _grid_methods._clamp_grid_expanded_bounds
    _plane_grid_boundaries = _grid_methods._plane_grid_boundaries
    _plane_grid_centers = _grid_methods._plane_grid_centers
    _diff_area = _grid_methods._diff_area
    _tangential_corrected = _grid_methods._tangential_corrected
    _tangential_fields = _grid_methods._tangential_fields
    _colocated_fields = _grid_methods._colocated_fields
    _colocated_tangential_fields = _grid_methods._colocated_tangential_fields
    grid_corrected_copy = _grid_methods.grid_corrected_copy

    # Metrics
    intensity = _metrics_methods.intensity
    field_intensity = _metrics_methods.field_intensity
    complex_poynting = _metrics_methods.complex_poynting
    poynting = _metrics_methods.poynting
    _prepare_fields_for_flux = _metrics_methods._prepare_fields_for_flux
    package_flux_results = _metrics_methods.package_flux_results
    _compute_complex_flux = _metrics_methods._compute_complex_flux
    complex_flux = _metrics_methods.complex_flux
    flux = _metrics_methods.flux
    mode_area = _metrics_methods.mode_area
    _bounding_box_mask = _metrics_methods._bounding_box_mask
    fill_fraction = _metrics_methods.fill_fraction
    fill_fraction_box = _metrics_methods.fill_fraction_box

    # Algebra
    _prepare_fields_for_dot = _algebra_methods._prepare_fields_for_dot
    dot = _algebra_methods.dot
    _fields_share_tangential_coords = _algebra_methods._fields_share_tangential_coords
    _tangential_fields_match_coords = _algebra_methods._tangential_fields_match_coords
    _interpolated_tangential_fields = _algebra_methods._interpolated_tangential_fields
    _prepare_fields_for_outer_dot = _algebra_methods._prepare_fields_for_outer_dot
    outer_dot = _algebra_methods.outer_dot

    # I/O
    time_reversed_copy = _io_methods.time_reversed_copy
    _check_fields_stored = _io_methods._check_fields_stored
    translated_copy = _io_methods.translated_copy
    to_zbf = _io_methods.to_zbf
    _interpolated_copies_if_needed = _io_methods._interpolated_copies_if_needed
