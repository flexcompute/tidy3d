from __future__ import annotations

import struct
from typing import TYPE_CHECKING, get_args

import autograd.numpy as np

from tidy3d.components.types import UnitsZBF
from tidy3d.constants import (
    C_0,
    UnitScaling,
)
from tidy3d.exceptions import DataError
from tidy3d.log import log

if TYPE_CHECKING:
    from os import PathLike

    from tidy3d.components.data.data_array import ScalarFieldDataArray
    from tidy3d.components.data.monitor_data.field.data import FieldData
    from tidy3d.components.data.monitor_data.field.electromagnetic import ElectromagneticFieldData
    from tidy3d.components.types import Coordinate


@property
def time_reversed_copy(self: ElectromagneticFieldData) -> FieldData:
    """Make a copy of the data with time-reversed fields."""

    # Time reversal for frequency-domain fields; overwritten in :class:`FieldTimeData`,
    # :class:`ModeData`, and :class:`ModeSolverData`.
    new_data = {}
    for comp, field in self.field_components.items():
        if comp[0] == "H":
            new_data[comp] = -np.conj(field)
        else:
            new_data[comp] = np.conj(field)
    return self.copy(deep=False, update=new_data)


def _check_fields_stored(self: ElectromagneticFieldData, components: list[str]) -> None:
    """Check that all requested field components are stored in the data."""
    missing_comps = [comp for comp in components if comp not in self.field_components.keys()]
    if len(missing_comps) > 0:
        raise DataError(
            f"Field components {missing_comps} not included in this data object. Use "
            "the 'fields' argument of a field monitor to select which components are stored."
        )


def translated_copy(self: ElectromagneticFieldData, vector: Coordinate) -> ElectromagneticFieldData:
    """Create a copy of the :class:`.ElectromagneticFieldData` with fields translated
    by the provided vector. Can be used together with ``dot`` or ``outer_dot``
    to compute overlaps between field data at different locations.

    Parameters
    ----------
    vector: :class:`.Coordinate`
        Translation vector to apply to the field data.

    Returns
    -------
    :class:`ElectromagneticFieldData`
        A data object with the translated fields.
    """
    field_kwargs = {}
    for key, val in self.field_components.items():
        coords = dict(val.coords)
        coords["x"] = coords["x"] + vector[0]
        coords["y"] = coords["y"] + vector[1]
        coords["z"] = coords["z"] + vector[2]
        field_kwargs[key] = val.assign_coords(coords)

    symmetry_center = self.symmetry_center
    if symmetry_center is not None:
        symmetry_center = tuple([x + y for (x, y) in zip(symmetry_center, vector)])
    grid_expanded = self.grid_expanded._translated_copy(vector=vector)

    monitor_center = tuple([x + y for (x, y) in zip(self.monitor.center, vector)])
    monitor = self.monitor.updated_copy(center=monitor_center)

    return self.updated_copy(
        monitor=monitor,
        symmetry=self.symmetry,
        symmetry_center=symmetry_center,
        grid_expanded=grid_expanded,
        **self._grid_correction_dict,
        **field_kwargs,
        deep=False,
    )


def to_zbf(
    self: ElectromagneticFieldData,
    fname: PathLike,
    units: UnitsZBF = "mm",
    background_refractive_index: float = 1,
    n_x: int | None = None,
    n_y: int | None = None,
    freq: float | None = None,
    mode_index: int | None = None,
    r_x: float = 0,
    r_y: float = 0,
    z_x: float = 0,
    z_y: float = 0,
    rec_efficiency: float = 0,
    sys_efficiency: float = 0,
) -> tuple[ScalarFieldDataArray, ScalarFieldDataArray]:
    """For a 2D monitor, export the fields to a Zemax Beam File (``.zbf``).

    The mode area is used to approximate the beam waist, which is only valid
    if the beam profile approximates a Gaussian beam.

    Parameters
    ----------
    fname : PathLike
        Full path to the ``.zbf`` file to be written.
    units : UnitsZBF = "mm"
        Spatial units used for the ``.zbf`` file. Options are ``"mm"``, ``"cm"``, ``"in"``, or ``"m"``.
        Defaults to ``"mm"``.
    background_refractive_index : float = 1
        Refractive index of the medium surrounding the monitor. Defaults to ``1``.
    n_x : Optional[int] = None
        Number of field samples along x.
        Must be a power of 2, between 2^5 and 2^13 inclusive per Zemax's requirements.
        Defaults to ``None``, in which case a value is chosen for the user depending on the coordinates in the field data.
    n_y : Optional[int] = None
        Number of field samples along y.
        Must be a power of 2, between 2^5 and 2^13 inclusive per Zemax's requirements.
        Defaults to ``None``, in which case a value is chosen for the user depending on the coordinates in the field data.
    freq : Optional[float] = None
        Field frequency selection. If ``None``, the average of the recorded frequencies is used.
    mode_index : Optional[int] = None
        For :class:`.ModeData`, choose which mode to save.
    r_x : float = 0
        Pilot beam Rayleigh distance in x, um. Defaults to ``0``.
    r_y : float = 0
        Pilot beam Rayleigh distance in y, um. Defaults to ``0``.
    z_x : float = 0
        Pilot beam z position with respect to the waist in x, um. Defaults to ``0``.
    z_y : float = 0
        Pilot beam z position with respect to the waist in y, um. Defaults to ``0``.
    rec_efficiency : float = 0
        Receiver efficiency, zero if fiber coupling is not computed. Defaults to ``0``.
    sys_efficiency : float = 0
        System efficiency, zero if fiber coupling is not computed. Defaults to ``0``.

    Returns
    -------
    tuple[:class:`.ScalarFieldDataArray`,:class:`.ScalarFieldDataArray`]
        The two E field components being exported to ``.zbf``.
    """
    log.warning(
        "'FieldData.to_zbf()' is currently an experimental feature."
        " If any issues are encountered, please contact Flexcompute support 'https://www.flexcompute.com/tidy3d/technical-support/'"
    )

    # Check that appropriate units are used
    if units not in get_args(UnitsZBF):
        raise ValueError("'units' must be either 'mm', 'cm', 'in', or 'm'.")

    # Mode area calculation ensures all E components are present
    mode_area = self.mode_area
    dim1, dim2 = self._tangential_dims

    # Using file-local coordinates x, y for the tangential components
    e_x = self._tangential_fields["E" + dim1]
    e_y = self._tangential_fields["E" + dim2]
    x = e_x.coords[dim1].values
    y = e_x.coords[dim2].values

    # Use the mean frequency if freq is not specified
    if freq is None:
        log.warning(
            "'freq' was not specified for 'FieldData.to_zbf()'. Defaulting to the mean frequency of the dataset."
        )
        freq = np.mean(e_x.coords["f"].values)
    else:
        freq = np.array(freq)

    if freq.size > 1:
        raise ValueError("'freq' must be a single value, not an array.")
    else:
        freq = freq.item()

    # If the data has just one frequency, avoid Nans at the interpolation
    if len(e_x.f) > 1:
        mode_area = mode_area.interp(f=freq)
        e_x = e_x.interp(f=freq)
        e_y = e_y.interp(f=freq)
    else:
        e_x = e_x.isel(f=0, drop=True)
        e_y = e_y.isel(f=0, drop=True)

    # If the data is ModeData, choose one of the modes to save
    if "mode_index" in e_x.coords:
        if mode_index is None:
            raise ValueError("'mode_index' is required for 'ModeData.to_zbf()'")
        mode_area = mode_area.isel(mode_index=mode_index, drop=True)
        e_x = e_x.isel(mode_index=mode_index, drop=True)
        e_y = e_y.isel(mode_index=mode_index, drop=True)

    # Header info
    version = 1
    polarized = 1
    unit_mapping = {"mm": 0, "cm": 1, "in": 2, "m": 3}
    unit_key = unit_mapping[units]
    unit_scaling = UnitScaling[units]
    lda = C_0 / freq * unit_scaling

    # Pilot (reference) beam waist: use the mode area to approximate the expected value
    w_x = (mode_area.item() / np.pi) ** 0.5 * unit_scaling
    w_y = w_x

    # Pilot beam Rayleigh distance (ignored on input)
    r_x *= unit_scaling
    r_y *= unit_scaling

    # Pilot beam z position w.r.t. the waist
    z_x *= unit_scaling
    z_y *= unit_scaling

    # defaults for n_x and n_y
    if n_x is None:
        n_x = 2 ** min(13, max(5, int(np.log2(x.size) + 1)))
        log.warning(
            f"'n_x' was not specified for 'FieldData.to_zbf()'. Defaulting to 'n_x' = {n_x}."
        )
    if n_y is None:
        n_y = 2 ** min(13, max(5, int(np.log2(y.size) + 1)))
        log.warning(
            f"'n_y' was not specified for 'FieldData.to_zbf()'. Defaulting to 'n_y' = {n_y}."
        )

    # Check that requirements are met for n_x and n_y
    # n_x and n_y must be powers of 2
    if (n_x & (n_x - 1)) != 0:
        raise ValueError("'n_x' must be a power of 2.")
    if (n_y & (n_y - 1)) != 0:
        raise ValueError("'n_y' must be a power of 2.")
    # 32 <= n_x and n_y <= 2^13
    if n_x < 32 or n_x > 2**13:
        raise ValueError("'n_x' must be between 2^5 and 2^13, inclusive.")
    if n_y < 32 or n_y > 2**13:
        raise ValueError("'n_y' must be between 2^5 and 2^13, inclusive.")

    # Interpolating coordinates
    x = np.linspace(x.min(), x.max(), n_x)
    y = np.linspace(y.min(), y.max(), n_y)

    # Interpolate fields
    coords = {dim1: x, dim2: y}
    e_x = e_x.interp(coords, assume_sorted=True)
    e_y = e_y.interp(coords, assume_sorted=True)

    # Sampling distance
    d_x = np.mean(np.diff(x)) * unit_scaling
    d_y = np.mean(np.diff(y)) * unit_scaling

    with open(fname, "wb") as fout:
        fout.write(struct.pack("<5I", version, n_x, n_y, polarized, unit_key))
        fout.write(struct.pack("<4I", 0, 0, 0, 0))  # unused values
        fout.write(struct.pack("<8d", d_x, d_y, z_x, r_x, w_x, z_y, r_y, w_y))
        fout.write(
            struct.pack("<4d", lda, background_refractive_index, rec_efficiency, sys_efficiency)
        )
        fout.write(struct.pack("<8d", 0, 0, 0, 0, 0, 0, 0, 0))  # unused values
        for e in (e_x, e_y):
            e_flat = e.values.flatten(order="F")
            # Interweave real and imaginary parts
            e_values = np.ravel(np.column_stack((e_flat.real, e_flat.imag)))
            fout.write(struct.pack(f"<{2 * n_x * n_y}d", *e_values))

    return e_x, e_y


def _interpolated_copies_if_needed(
    self: ElectromagneticFieldData, other: ElectromagneticFieldData
) -> tuple[ElectromagneticFieldData, ElectromagneticFieldData]:
    """Return interpolated copies of self, other if needed (different interp_spec)."""
    from tidy3d.components.data.monitor_data.mode import ModeSolverData

    mode_spec1 = self.monitor.mode_spec if isinstance(self, ModeSolverData) else None
    mode_spec2 = other.monitor.mode_spec if isinstance(other, ModeSolverData) else None
    if (
        mode_spec1 is not None
        and mode_spec2 is not None
        and self.monitor.mode_spec._same_nontrivial_interp_spec(other=other.monitor.mode_spec)
    ):
        return self, other
    self_copy = self.interpolated_copy if isinstance(self, ModeSolverData) else self
    other_copy = other.interpolated_copy if isinstance(other, ModeSolverData) else other
    return self_copy, other_copy
