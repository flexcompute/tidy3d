from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from flexcompute.core._migration.em.numerical.raw import (
    field_data as field_data_numerics,
)
from flexcompute.core._migration.em.numerical.raw import (
    mode as mode_numerics,
)

from tidy3d.components.data.data_array import (
    FreqDataArray,
    FreqModeDataArray,
    MixedModeDataArray,
)
from tidy3d.components.data.utils import (
    _get_broadcast_selection,
    _get_intersection_selection,
)
from tidy3d.exceptions import DataError
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.data.monitor_data.field.data import FieldData
    from tidy3d.components.data.monitor_data.field.electromagnetic import ElectromagneticFieldData
    from tidy3d.components.data.monitor_data.mode.data import ModeData
    from tidy3d.components.data.monitor_data.mode.solver import ModeSolverData
    from tidy3d.components.types import ArrayFloat2D


def _prepare_fields_for_dot(
    self: ElectromagneticFieldData,
    fields_self: dict[str, DataArray],
    fields_other: dict[str, DataArray],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align coordinates and convert to numpy for "dot" overlap calculations.

    Both returned field dicts have 4-D arrays: ``(N_f, N_modes, Nu, Nv)``.
    When a dataset lacks ``mode_index``, a size-1 dummy dimension is inserted
    (numpy broadcasting handles the mismatch).

    Coordinate alignment:
    - If other has length 1 along ``f`` or ``mode_index``, broadcast to match
      self's coordinate values.
    - Otherwise, the intersection of coordinate values is used.
    """
    tangential_dims = self._tangential_dims
    test_field = "E" + tangential_dims[0]

    # Get coords from both datasets
    self_coords = fields_self[test_field].coords
    other_coords = fields_other[test_field].coords
    self_has_mode = "mode_index" in self_coords
    other_has_mode = "mode_index" in other_coords

    # Compute selections for frequency (always present in both)
    f_sel_self, f_sel_other, final_freqs = _get_broadcast_selection(
        self_coords["f"].values, other_coords["f"].values
    )

    # Compute selections for mode_index
    if self_has_mode and other_has_mode:
        m_sel_self, m_sel_other, final_modes = _get_broadcast_selection(
            self_coords["mode_index"].values, other_coords["mode_index"].values
        )
    elif self_has_mode:
        m_sel_self, m_sel_other = slice(None), None
        final_modes = self_coords["mode_index"].values
    elif other_has_mode:
        m_sel_self, m_sel_other = None, slice(None)
        final_modes = other_coords["mode_index"].values
    else:
        m_sel_self, m_sel_other, final_modes = None, None, None

    def prepare_field_dict(
        fields: dict[str, DataArray],
        f_sel: int | slice | np.ndarray,
        m_sel: int | slice | np.ndarray | None,
        has_mode: bool,
    ) -> dict[str, np.ndarray]:
        """Select, expand dims, transpose, extract numpy."""
        result = {}
        for key, field in fields.items():
            da = field.isel(f=f_sel)
            if has_mode and m_sel is not None:
                da = da.isel(mode_index=m_sel)
            if not has_mode:
                da = da.expand_dims("mode_index")
            result[key] = da.transpose("f", "mode_index", *tangential_dims).values
        return result

    prepped_fields_self = prepare_field_dict(fields_self, f_sel_self, m_sel_self, self_has_mode)
    prepped_fields_other = prepare_field_dict(
        fields_other, f_sel_other, m_sel_other, other_has_mode
    )

    # Build final_coords dict
    final_coords = {"f": final_freqs}
    if final_modes is not None:
        final_coords["mode_index"] = final_modes

    return final_coords, prepped_fields_self, prepped_fields_other


def dot(
    self: ElectromagneticFieldData,
    field_data: FieldData | ModeData | ModeSolverData,
    conjugate: bool = True,
    bidirectional: bool = True,
) -> FreqDataArray | FreqModeDataArray:
    r"""Dot product (modal overlap) with another :class:`.FieldData` object. Both datasets have
    to be frequency-domain data associated with a 2D monitor.

    When either monitor uses ``colocate=True`` (default) or
    ``use_colocated_integration=True``, the tangential fields from ``field_data`` are
    interpolated onto this object's grid, so the two datasets may have different spatial
    discretizations. Otherwise, both datasets must share the same tangential grid.

    Along the normal direction, the monitor position may differ and is ignored.
    Non-spatial coordinates (``f``, ``mode_index``) are aligned by intersection;
    broadcasting is also supported when the other dataset has size 1 along a coordinate
    dimension.

    The dot product is defined as:

    .. math:

       \frac{1}{4} \int \left( E_0^* \times H_1 + E_1 \times H_0^* \right) \, {\rm d}S

    If ``bidirectional=False``, the dot product is instead:

    .. math:

       \frac{1}{2} \int \left( E_0^* \times H_1 \right) \, {\rm d}S

    Parameters
    ----------
    field_data : :class:`.FieldData` | :class:`.ModeData` | :class:`.ModeSolverData`
        A data instance to compute the dot product with.
    conjugate : bool, optional
        If ``True`` (default), the dot product is defined as above. If ``False``, the definition
        is similar, but without the complex conjugation of the fields.
    bidirectional : bool = True
        If ``True`` (default), computes the symmetric bidirectional overlap:
        ``1/4 * integral(E1* x H2 + E2 x H1*) dS``.
        If ``False``, computes just: ``1/2 * integral(E1* x H2) dS``.

    Returns
    -------
    :class:`.FreqDataArray` | :class:`.FreqModeDataArray`
        Data array with the complex-valued modal overlaps.

        - If neither dataset has ``mode_index``: returns :class:`.FreqDataArray`.
        - If either dataset has ``mode_index``: returns :class:`.FreqModeDataArray`.

    Note
    ----
        The dot product with and without conjugation is equivalent (up to a phase) for
        modes in lossless waveguides but differs for modes in lossy materials. In that case,
        the conjugated dot product can be interpreted as the fraction of the power carried by
        the second mode, but modes are not orthogonal with respect to that product
        and the sum of carried power fractions may be different from the total flux.
        In the non-conjugated definition, modes are orthogonal, but the interpretation of the
        dot product as power carried by a given mode is no longer valid.
    """
    use_colocated = mode_numerics.resolve_colocated_default(
        (
            {
                "colocate": self.monitor.colocate,
                "use_colocated_integration": getattr(
                    self.monitor, "use_colocated_integration", None
                ),
            },
            {
                "colocate": field_data.monitor.colocate,
                "use_colocated_integration": getattr(
                    field_data.monitor, "use_colocated_integration", None
                ),
            },
        ),
        None,
    )
    if not use_colocated:
        fields_self = self._tangential_fields
        fields_other = field_data._tangential_fields
        if not self._fields_share_tangential_coords(fields_self, fields_other):
            log.warning(
                "Tangential field coordinates do not match in 'dot'; "
                "switching to colocated-based computation."
            )
            use_colocated = True

    if use_colocated:
        fields_self = self._colocated_tangential_fields
        fields_other = field_data._interpolated_tangential_fields(self._plane_grid_boundaries)
        d_area = self._diff_area.to_numpy()
        dS_numpy = (d_area, d_area)
    else:
        dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(truncate_to_monitor_bounds=False)
        dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

    # Determine broadcast behavior and final dimensions (returns numpy arrays directly)
    final_coords, prepped_fields_self, prepped_fields_other = self._prepare_fields_for_dot(
        fields_self, fields_other
    )

    # Extract tangential components as tuples (already numpy arrays)
    u, v = self._tangential_dims
    E1 = (prepped_fields_self["E" + u], prepped_fields_self["E" + v])
    H1 = (prepped_fields_self["H" + u], prepped_fields_self["H" + v])
    E2 = (prepped_fields_other["E" + u], prepped_fields_other["E" + v])
    H2 = (prepped_fields_other["H" + u], prepped_fields_other["H" + v])

    dot_result = field_data_numerics.dot(
        E1, H1, E2, H2, dS_numpy, conjugate=conjugate, bidirectional=bidirectional
    )

    # Squeeze out mode_index dimension (axis 1) if not in final_coords
    if "mode_index" not in final_coords:
        dot_result = dot_result.squeeze(axis=1)
        return FreqDataArray(dot_result, coords=final_coords)
    else:
        return FreqModeDataArray(dot_result, coords=final_coords)


def _fields_share_tangential_coords(
    self: ElectromagneticFieldData,
    fields_a: dict[str, DataArray],
    fields_b: dict[str, DataArray],
) -> bool:
    """Check whether two tangential field sets share the same coordinates."""
    for key in fields_a:
        if key not in fields_b:
            return False
        for dim in self._tangential_dims:
            coords_a = fields_a[key].coords[dim].values
            coords_b = fields_b[key].coords[dim].values
            if coords_a.size != coords_b.size:
                return False
            if not np.allclose(coords_a, coords_b):
                return False
    return True


def _tangential_fields_match_coords(self: ElectromagneticFieldData, coords: ArrayFloat2D) -> bool:
    """Check if the tangential fields already match given coords in the tangential plane."""
    for field in self._tangential_fields.values():
        for idim, dim in enumerate(self._tangential_dims):
            if field.coords[dim].values.size != coords[idim].size or not np.all(
                field.coords[dim].values == coords[idim]
            ):
                return False
    return True


def _interpolated_tangential_fields(
    self: ElectromagneticFieldData, coords: ArrayFloat2D
) -> dict[str, DataArray]:
    """For 2D monitors, interpolate this fields to given coords in the tangential plane.

    When ``solver_field_bounds`` is set, uses clip-aware interpolation so that
    zero-padded values outside the solver grid do not contaminate boundary
    values, consistent with ``_colocated_fields``.

    Parameters
    ----------
    coords : ArrayFloat2D
        Interpolation coords in the monitor's tangential plane.

    Return
    ------
        Dictionary with interpolated fields.
    """
    fields = self._tangential_fields

    # If coords already match, just return the tangential fields directly.
    if self._tangential_fields_match_coords(coords):
        return fields

    # Interpolate if data has more than one coordinate along a dimension
    interp_dict = {}
    # If single coordinate, just sel "nearest", i.e. just propagate the same data everywhere
    sel_dict = {"method": "nearest"}
    for dim, cents in zip(self._tangential_dims, coords):
        if cents.size > 0:
            if list(fields.values())[0].coords[dim].size > 1:
                interp_dict[dim] = cents
            else:
                sel_dict[dim] = cents

    # Use symmetry-expanded bounds since _tangential_fields returns
    # symmetry-expanded data.
    clip_bounds = self.symmetry_expanded.solver_field_bounds
    if clip_bounds is not None:
        for component, field in fields.items():
            fields[component] = field.interp_within_domain(
                interp_dict, clip_bounds, assume_sorted=True
            ).sel(**sel_dict)
    else:
        interp_dict["assume_sorted"] = True
        kwargs = {"bounds_error": False, "fill_value": 0.0}
        for component, field in fields.items():
            fields[component] = field.interp(kwargs=kwargs, **interp_dict).sel(**sel_dict)

    return fields


def _prepare_fields_for_outer_dot(
    self: ElectromagneticFieldData,
    fields_self: dict[str, DataArray],
    fields_other: dict[str, DataArray],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align coordinates and convert to numpy for "outer dot" overlap calculations.

    Both returned field dicts have 4-D arrays: ``(N_f, N_modes, Nu, Nv)``.
    When a dataset lacks ``mode_index``, a size-1 dummy dimension is inserted.

    Coordinate alignment:
    - Frequency: intersection only (no broadcasting).
    - Mode index: each side keeps its own modes (no alignment); ``final_coords``
      uses ``mode_index_0`` / ``mode_index_1`` to distinguish them.
    """
    tangential_dims = self._tangential_dims
    test_field = "E" + tangential_dims[0]

    # Get coords from both datasets
    self_coords = fields_self[test_field].coords
    other_coords = fields_other[test_field].coords
    self_has_mode = "mode_index" in self_coords
    other_has_mode = "mode_index" in other_coords

    # Frequency: intersection only (no broadcasting)
    f_sel_self, f_sel_other, common_freqs = _get_intersection_selection(
        self_coords["f"].values, other_coords["f"].values
    )

    def prepare_field_dict(
        fields: dict[str, DataArray],
        f_sel: int | slice | np.ndarray,
        has_mode: bool,
    ) -> dict[str, np.ndarray]:
        """Select freq, expand dims, transpose, extract numpy."""
        result = {}
        for key, field in fields.items():
            da = field.isel(f=f_sel)
            if not has_mode:
                da = da.expand_dims("mode_index")
            result[key] = da.transpose("f", "mode_index", *tangential_dims).values
        return result

    prepped_fields_self = prepare_field_dict(fields_self, f_sel_self, self_has_mode)
    prepped_fields_other = prepare_field_dict(fields_other, f_sel_other, other_has_mode)

    # Prepare final coords
    final_coords = {"f": common_freqs}

    # Determine mode_index coordinate handling:
    # - Neither has mode_index → just f → FreqDataArray
    # - At least one has mode_index → f + mode_index_0 + mode_index_1 → MixedModeDataArray
    if self_has_mode and other_has_mode:
        final_coords["mode_index_0"] = self_coords["mode_index"].values
        final_coords["mode_index_1"] = other_coords["mode_index"].values
    elif self_has_mode:
        final_coords["mode_index_0"] = self_coords["mode_index"].values
    elif other_has_mode:
        final_coords["mode_index_1"] = other_coords["mode_index"].values

    return final_coords, prepped_fields_self, prepped_fields_other


def outer_dot(
    self: ElectromagneticFieldData,
    field_data: FieldData | ModeData | ModeSolverData,
    conjugate: bool = True,
    bidirectional: bool = True,
    truncate_to_monitor_bounds: bool = False,
) -> FreqDataArray | MixedModeDataArray:
    r"""Outer dot product (pairwise modal overlap matrix) with another :class:`.FieldData`
    object.

    When either monitor uses ``colocate=True`` (default) or
    ``use_colocated_integration=True``, the tangential fields from ``field_data`` are
    interpolated onto this object's grid, so the two datasets may have different spatial
    discretizations. Otherwise, both datasets must share the same tangential grid.

    The calculation is performed for all common frequencies between the two datasets.

    The dot product is defined as:

    .. math:

       \frac{1}{4} \int \left( E_0^* \times H_1 + E_1 \times H_0^* \right) \, {\rm d}S

    If ``bidirectional=False``, the dot product is instead:

    .. math:

       \frac{1}{2} \int \left( E_0^* \times H_1 \right) \, {\rm d}S

    Parameters
    ----------
    field_data : :class:`.FieldData` | :class:`.ModeData` | :class:`.ModeSolverData`
        A data instance to compute the dot product with.
    conjugate : bool = True
        If ``True`` (default), the dot product is defined as above. If ``False``, the definition
        is similar, but without the complex conjugation of the fields.
    bidirectional : bool = True
        If ``True`` (default), computes the symmetric bidirectional overlap:
        ``1/4 * integral(E1* x H2 + E2 x H1*) dS``.
        If ``False``, computes just: ``1/2 * integral(E1* x H2) dS``.
    truncate_to_monitor_bounds : bool = False
        Only used in the non-colocated integration path (when ``colocate=False``).
        If ``True``, clamp integration area to monitor bounds, consistent with
        ``complex_flux``. If ``False`` (default), use grid-enclosing bounds.

    Returns
    -------
    :class:`.FreqDataArray` | :class:`.MixedModeDataArray`
        Data array with the complex-valued modal overlaps.

        - If neither dataset has ``mode_index``: returns :class:`.FreqDataArray`.
        - If only ``self`` has ``mode_index``: returns :class:`.MixedModeDataArray` with
          ``mode_index_0`` coordinate.
        - If only ``field_data`` has ``mode_index``: returns :class:`.MixedModeDataArray` with
          ``mode_index_1`` coordinate.
        - If both datasets have ``mode_index``: returns :class:`.MixedModeDataArray` with
          ``mode_index_0`` and ``mode_index_1`` coordinates.

    See also
    --------
    :meth:`dot`
    """

    tan_dims = self._tangential_dims

    if not all(a == b for a, b in zip(tan_dims, field_data._tangential_dims)):
        raise DataError("Tangential dimensions must match between the two monitors.")

    use_colocated = mode_numerics.resolve_colocated_default(
        (
            {
                "colocate": self.monitor.colocate,
                "use_colocated_integration": getattr(
                    self.monitor, "use_colocated_integration", None
                ),
            },
            {
                "colocate": field_data.monitor.colocate,
                "use_colocated_integration": getattr(
                    field_data.monitor, "use_colocated_integration", None
                ),
            },
        ),
        None,
    )
    if not use_colocated:
        fields_self = self._tangential_fields
        fields_other = field_data._tangential_fields
        if not self._fields_share_tangential_coords(fields_self, fields_other):
            log.warning(
                "Tangential field coordinates do not match in 'outer_dot'; "
                "switching to colocated-based computation."
            )
            use_colocated = True

    if use_colocated:
        fields_self = self._colocated_tangential_fields
        fields_other = field_data._interpolated_tangential_fields(self._plane_grid_boundaries)
        d_area = self._diff_area.to_numpy()
        dS_numpy = (d_area, d_area)
    else:
        dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(
            truncate_to_monitor_bounds=truncate_to_monitor_bounds
        )
        dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

    # Determine broadcast behavior and final dimensions (returns numpy arrays directly)
    final_coords, prepped_fields_self, prepped_fields_other = self._prepare_fields_for_outer_dot(
        fields_self, fields_other
    )

    # Extract tangential components as tuples (already numpy arrays)
    u, v = self._tangential_dims
    E1 = (prepped_fields_self["E" + u], prepped_fields_self["E" + v])
    H1 = (prepped_fields_self["H" + u], prepped_fields_self["H" + v])
    E2 = (prepped_fields_other["E" + u], prepped_fields_other["E" + v])
    H2 = (prepped_fields_other["H" + u], prepped_fields_other["H" + v])
    numpy_result = field_data_numerics.outer_dot(
        E1, H1, E2, H2, dS_numpy, conjugate=conjugate, bidirectional=bidirectional
    )

    # Determine return type based on final_coords
    # numpy_result shape is (n_freq, n_modes_0, n_modes_1)
    has_mode_index_0 = "mode_index_0" in final_coords
    has_mode_index_1 = "mode_index_1" in final_coords

    if has_mode_index_0 and has_mode_index_1:
        # Both have mode_index: return MixedModeDataArray (no squeeze)
        return MixedModeDataArray(numpy_result, coords=final_coords)
    elif has_mode_index_0:
        # Only self has mode_index: squeeze axis 2 and return MixedModeDataArray
        squeezed = numpy_result.squeeze(axis=2)
        return MixedModeDataArray(squeezed, coords=final_coords)
    elif has_mode_index_1:
        # Only other has mode_index: squeeze axis 1 and return MixedModeDataArray
        squeezed = numpy_result.squeeze(axis=1)
        return MixedModeDataArray(squeezed, coords=final_coords)
    else:
        # Neither has mode_index: return FreqDataArray
        squeezed = numpy_result.squeeze(axis=(1, 2))
        return FreqDataArray(squeezed, coords=final_coords)
