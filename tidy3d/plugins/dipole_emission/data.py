"""Result data objects for the dipole emission plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import C_0, MU_0, STERADIAN
from tidy3d.exceptions import DataError
from tidy3d.plugins.dipole_emission.data_array import (
    DipoleEmissionStudyDataArray,
    DipoleEmissionStudyPositionDataArray,
)
from tidy3d.plugins.dipole_emission.study import DipoleEmissionStudy

if TYPE_CHECKING:
    from tidy3d.components.types import ArrayFloat1D

# Total power emitted by a unit dipole (|d| = 1 C*um) in a uniform medium of index n,
# per |S(omega)|^2: P_bulk = n * omega^4 * MU_0 / (12 pi c) * |S|^2. The tidy3d
# constants are micrometer-based (MU_0 per um, C_0 in um/s), so the prefactor is
# directly per squared dipole moment in C*um with no further unit conversion.
_BULK_POWER_PREFACTOR = MU_0 / (12 * np.pi * C_0)

# The transfer methods return a normalized ratio (radiated / bulk emitted power),
# which is per solid angle, i.e. 1/sr -- not the intensity units carried by the
# stored ``radiation_intensity`` arrays. The returned arrays are relabeled accordingly.
_TRANSFER_UNITS = f"1/{STERADIAN}"


class DipoleEmissionStudyData(Tidy3dBaseModel):
    """Reduced result from a :class:`.DipoleEmissionStudy`.

    The default data arrays are summed over the study ``positions`` using any
    position- and axis-dependent ``position_weights`` and are indexed by
    Cartesian dipole orientation, far-field polarization, observation angle, and
    frequency. They represent angular radiation intensity per squared electric
    dipole moment, with dipole moment expressed in C*um, into the collection
    half-space defined by the study.
    ``radiation_intensity_transfer(...)`` normalizes the stored intensity by the
    total power the same dipoles would emit in a uniform medium at the emitter
    refractive index. Optional position-resolved arrays are present only when
    the study ``store_position_indexes`` is nonempty. Raw ``BatchData`` is
    intentionally not stored here; use
    ``DipoleEmissionStudy.run(..., return_batch_data=True)`` or manual batch
    execution when diagnostic monitor data is needed.
    """

    radiation_intensity: DipoleEmissionStudyDataArray = Field(
        title="Radiation Intensity",
        description=(
            "Radiated angular power density per squared electric dipole moment "
            "in C*um. The ``source_time`` spectrum from the study is included."
        ),
    )

    radiation_intensity_at_positions: DipoleEmissionStudyPositionDataArray | None = Field(
        None,
        title="Position-Resolved Radiation Intensity",
        description=(
            "Radiation intensity at positions selected by the study ``store_position_indexes``."
        ),
    )

    study: DipoleEmissionStudy = Field(
        title="Study",
        description="Study definition used to generate this result.",
    )

    @model_validator(mode="after")
    def _validate_angle_dimension(self) -> DipoleEmissionStudyData:
        """Check the ``angle`` dimension matches the study angles.

        Angle metadata lives only in ``study.angles`` (the result arrays carry
        integer ``angle`` indices, not ``theta``/``phi`` coordinates). The
        ``theta``/``phi`` properties read from there. Validate the index axis is
        consistent rather than duplicating the angle values on the arrays.
        """
        num_angles = self.study.angles.sizes["index"]
        expected = np.arange(num_angles)
        for name in ("radiation_intensity", "radiation_intensity_at_positions"):
            array = getattr(self, name)
            if array is None:
                continue
            if array.sizes["angle"] != num_angles or not np.array_equal(
                array.coords["angle"].values, expected
            ):
                self._raise_validation_error_at_loc(
                    f"'{name}' has an 'angle' dimension inconsistent with 'study.angles'.",
                    name,
                )
        return self

    @model_validator(mode="after")
    def _validate_stored_position_index(self) -> DipoleEmissionStudyData:
        """Check the position-resolved ``index`` axis matches the stored positions.

        ``radiation_intensity_transfer_at_positions`` aligns a per-stored-position
        ``bulk_refractive_index`` to this array's ``index`` coordinate, so that
        coordinate must carry the ``positions`` labels selected by
        ``study.store_position_indexes`` (the labels ``compose`` assigns). Without
        this check a malformed result would silently normalize each stored
        position against the wrong refractive index.
        """
        array = self.radiation_intensity_at_positions
        if array is None:
            return self
        position_index = np.asarray(self.study.positions.coords["index"].values)
        expected = position_index[list(self.study.store_position_indexes)]
        actual = np.asarray(array.coords["index"].values)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            self._raise_validation_error_at_loc(
                "'radiation_intensity_at_positions' has an 'index' dimension inconsistent "
                "with the study positions selected by 'store_position_indexes'.",
                "radiation_intensity_at_positions",
            )
        return self

    @property
    def theta(self) -> np.ndarray:
        """Polar observation angles (radians) of the ``angle`` dimension."""
        return np.asarray(self.study.angles.sel(spherical_coordinate="theta").values, dtype=float)

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal observation angles (radians) of the ``angle`` dimension."""
        return np.asarray(self.study.angles.sel(spherical_coordinate="phi").values, dtype=float)

    def _bulk_power_spectrum(self) -> np.ndarray:
        """Bulk emitted power per frequency for a unit dipole in a unit-index medium.

        ``omega**4 * mu_0 / (12 pi c) * |S(omega)|**2``, the index-independent factor of
        ``P_bulk``. Computed in float64: the ``omega**4`` intermediate exceeds the
        float32 range.
        """
        omega = 2 * np.pi * np.asarray(self.study.freqs, dtype=np.float64)
        pulse_spectrum_abs2 = np.asarray(self.study.pulse_spectrum_abs2, dtype=np.float64)
        return _BULK_POWER_PREFACTOR * omega**4 * pulse_spectrum_abs2

    def _resolve_bulk_index(
        self, bulk_refractive_index: float | ArrayFloat1D, num: int, label: str
    ) -> np.ndarray:
        """Validate and broadcast ``bulk_refractive_index`` to a length-``num`` array."""
        raw_index = np.asarray(bulk_refractive_index)
        if np.iscomplexobj(raw_index):
            raise DataError("'bulk_refractive_index' must contain real values.")
        try:
            index = np.asarray(raw_index, dtype=np.float64)
        except (TypeError, ValueError):
            raise DataError("'bulk_refractive_index' must contain real values.") from None
        if index.ndim == 0:
            index = np.full(num, float(index))
        if index.shape != (num,):
            raise DataError(
                f"'bulk_refractive_index' must be a scalar or provide one value per {label} "
                f"({num}); got shape {index.shape}."
            )
        if not np.all(np.isfinite(index)) or np.any(index <= 0):
            raise DataError("'bulk_refractive_index' must contain finite positive values.")
        return index

    def radiation_intensity_transfer(
        self, bulk_refractive_index: float | ArrayFloat1D
    ) -> DipoleEmissionStudyDataArray:
        """Position-summed cavity transfer, in 1/sr.

        The stored ``radiation_intensity`` (the ``position_weights``-weighted sum
        over the sampled dipoles) is normalized by the total power the same
        weighted dipoles would emit in a uniform medium:
        ``T = radiation_intensity / sum_i(w_i * P_bulk_i)`` with
        ``P_bulk_i(omega) = n_i * omega**4 * mu_0 / (12 pi c) * |S(omega)|**2``
        the bulk emitted power of dipole ``i`` per squared dipole moment in C*um.
        Integrating ``T`` over solid angle gives the radiative Purcell factor; a
        bulk emitter integrates to 1 over the full sphere, so a single study,
        which collects one half-space (``abs(theta) < pi/2``), integrates to 1/2.

        A fully zeroed orientation column in 2D ``position_weights`` (e.g. a
        horizontal-only emitter with no z-dipole) gives ``NaN`` for that
        orientation: both the radiated and the bulk power are identically zero,
        so the transfer is undefined there rather than numerically wrong.

        Parameters
        ----------
        bulk_refractive_index : float | ArrayFloat1D
            Real refractive index of the uniform reference medium containing the
            emitters. This is the emitter-side index of the bulk reference; it is
            distinct from the collection-side background index already baked into
            ``radiation_intensity``, and the two coincide only when emitter and
            collection media match. Provide a scalar, or one value per sampled
            position (length ``positions.sizes["index"]``, in ``positions``
            order) when positions span different media.

        Returns
        -------
        DipoleEmissionStudyDataArray
            Transfer with dimensions ``(dipole_axis, polarization, angle, f)``,
            in the dtype of ``radiation_intensity``.
        """
        num_positions = self.study.positions.sizes["index"]
        index = self._resolve_bulk_index(bulk_refractive_index, num_positions, "sampled position")
        bulk_power = index[:, None] * self._bulk_power_spectrum()[None, :]
        weights = np.asarray(self.study.position_weights_array, dtype=np.float64)
        if weights.ndim == 1:
            total_bulk_power = np.einsum("i,if->f", weights, bulk_power)
            denominator = total_bulk_power[None, None, None, :]
        else:
            total_bulk_power = np.einsum("ia,if->af", weights, bulk_power)
            denominator = total_bulk_power[:, None, None, :]

        values = self.radiation_intensity.values.astype(np.float64) / denominator
        transfer = DipoleEmissionStudyDataArray(
            values.astype(self.radiation_intensity.dtype),
            dims=DipoleEmissionStudyDataArray._dims,
            coords=self.radiation_intensity.coords,
        )
        transfer.attrs = {
            "long_name": "angular radiation intensity transfer (cavity / bulk)",
            "units": _TRANSFER_UNITS,
        }
        return transfer

    def radiation_intensity_transfer_at_positions(
        self, bulk_refractive_index: float | ArrayFloat1D
    ) -> DipoleEmissionStudyPositionDataArray | None:
        """Position-resolved cavity transfer at the stored positions, in 1/sr.

        Each stored dipole's radiation intensity is normalized by its own bulk
        emitted power ``P_bulk_i``; see :meth:`radiation_intensity_transfer`.
        Returns ``None`` when the study stores no individual positions.

        Parameters
        ----------
        bulk_refractive_index : float | ArrayFloat1D
            Real refractive index of the uniform reference medium. Provide a
            scalar, or one value per stored position (length matching the
            ``index`` dimension of ``radiation_intensity_at_positions``, in its
            ``index`` coordinate order).

        Returns
        -------
        DipoleEmissionStudyPositionDataArray | None
            Transfer with dimensions ``(index, dipole_axis, polarization, angle,
            f)``, in the dtype of ``radiation_intensity_at_positions``.
        """
        if self.radiation_intensity_at_positions is None:
            return None
        num_stored = self.radiation_intensity_at_positions.sizes["index"]
        index = self._resolve_bulk_index(bulk_refractive_index, num_stored, "stored position")
        bulk_power = index[:, None] * self._bulk_power_spectrum()[None, :]
        denominator = bulk_power[:, None, None, None, :]

        values = self.radiation_intensity_at_positions.values.astype(np.float64) / denominator
        transfer = DipoleEmissionStudyPositionDataArray(
            values.astype(self.radiation_intensity_at_positions.dtype),
            dims=DipoleEmissionStudyPositionDataArray._dims,
            coords=self.radiation_intensity_at_positions.coords,
        )
        transfer.attrs = {
            "long_name": "position-resolved angular radiation intensity transfer (cavity / bulk)",
            "units": _TRANSFER_UNITS,
        }
        return transfer
