"""Defines time modulation to the medium"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isclose
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, PositiveFloat, field_validator, model_validator

from tidy3d.constants import HERTZ, RADIAN
from tidy3d.exceptions import ValidationError

from .base import Tidy3dBaseModel, cached_property
from .data.data_array import SpatialDataArray
from .data.validators import validate_no_nans
from .time import AbstractTimeDependence
from .types import InterpMethod

if TYPE_CHECKING:
    from pydantic import FieldValidationInfo

    from tidy3d.compat import Self

    from .types import ArrayComplex1D, ArrayFloat1D, Bound


class AbstractTimeModulation(AbstractTimeDependence, ABC):
    """Base class for modulation in time.

    Note
    ----
    This class describes the time dependence part of the separable space-time modulation type
    as shown below,

    .. math::

        amp(r, t) = \\Re[amp\\_time(t) \\cdot amp\\_space(r)]

    """

    @cached_property
    @abstractmethod
    def max_modulation(self) -> float:
        """Estimated maximum modulation amplitude."""


class ContinuousWaveTimeModulation(AbstractTimeDependence):
    """Class describing modulation with a harmonic time dependence.

    Note
    ----
    .. math::

        amp\\_time(t) = amplitude \\cdot \\
                e^{i \\cdot phase - 2 \\pi i \\cdot freq0 \\cdot t}

    Note
    ----
    The full space-time modulation is,

    .. math::

        amp(r, t) = \\Re[amp\\_time(t) \\cdot amp\\_space(r)]


    Example
    -------
    >>> cw = ContinuousWaveTimeModulation(freq0=200e12, amplitude=1, phase=0)
    """

    freq0: PositiveFloat = Field(
        title="Modulation Frequency",
        description="Modulation frequency.",
        json_schema_extra={"units": HERTZ},
    )

    def amp_time(self, time: float | ArrayFloat1D) -> complex | ArrayComplex1D:
        """Complex-valued source amplitude as a function of time."""

        omega = 2 * np.pi * self.freq0
        return self.amplitude * np.exp(-1j * omega * time + 1j * self.phase)

    @cached_property
    def max_modulation(self) -> float:
        """Estimated maximum modulation amplitude."""
        return abs(self.amplitude)


TimeModulationType = ContinuousWaveTimeModulation


class AbstractSpaceModulation(ABC, Tidy3dBaseModel):
    """Base class for modulation in space.

    Note
    ----
    This class describes the 2nd term in the full space-time modulation below,
    .. math::

        amp(r, t) = \\Re[amp\\_time(t) \\cdot amp\\_space(r)]

    """

    @cached_property
    @abstractmethod
    def max_modulation(self) -> float:
        """Estimated maximum modulation amplitude."""


class SpaceModulation(AbstractSpaceModulation):
    """The modulation profile with a user-supplied spatial distribution of
    amplitude and phase.

    Note
    ----
    .. math::

        amp\\_space(r) = amplitude(r) \\cdot e^{i \\cdot phase(r)}

    The full space-time modulation is,

    .. math::

        amp(r, t) = \\Re[amp\\_time(t) \\cdot amp\\_space(r)]

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> amp = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> phase = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> space = SpaceModulation(amplitude=amp, phase=phase)
    """

    amplitude: float | SpatialDataArray = Field(
        1,
        title="Amplitude of modulation in space",
        description="Amplitude of modulation that can vary spatially. "
        "It takes the unit of whatever is being modulated.",
    )

    phase: float | SpatialDataArray = Field(
        0,
        title="Phase of modulation in space",
        description="Phase of modulation that can vary spatially.",
        json_schema_extra={"units": RADIAN},
    )

    interp_method: InterpMethod = Field(
        "nearest",
        title="Interpolation method",
        description="Method of interpolation to use to obtain values at spatial locations on the Yee grids.",
    )

    _no_nans = validate_no_nans("amplitude", "phase")

    @field_validator("amplitude", "phase")
    @classmethod
    def _validate_fields_real(
        cls, val: float | SpatialDataArray, info: FieldValidationInfo
    ) -> float | SpatialDataArray:
        """Assert that the amplitude is real."""
        if np.iscomplexobj(val):
            raise ValidationError(f"'{info.field_name}' must be real.")
        return val

    @cached_property
    def max_modulation(self) -> float:
        """Estimated maximum modulation amplitude."""
        return np.max(abs(np.array(self.amplitude)))

    def sel_inside(self, bounds: Bound) -> Self:
        """Return a new space modulation that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        SpaceModulation
            SpaceModulation with reduced data.
        """

        if isinstance(self.amplitude, SpatialDataArray):
            amp_reduced = self.amplitude.sel_inside(bounds)
        else:
            amp_reduced = self.amplitude

        if isinstance(self.phase, SpatialDataArray):
            phase_reduced = self.phase.sel_inside(bounds)
        else:
            phase_reduced = self.phase

        return self.updated_copy(amplitude=amp_reduced, phase=phase_reduced)


SpaceModulationType = SpaceModulation


class SpaceTimeModulation(Tidy3dBaseModel):
    """Space-time modulation applied to a medium, adding
    on top of the time-independent part.


    Note
    ----
    The space-time modulation must be separable in space and time.
    e.g. when applied to permittivity,

    .. math::

        \\delta \\epsilon(r, t) = \\Re[amp\\_time(t) \\cdot amp\\_space(r)]
    """

    space_modulation: SpaceModulationType = Field(
        default_factory=SpaceModulation,
        title="Space modulation",
        description="Space modulation part from the separable SpaceTimeModulation.",
    )

    time_modulation: TimeModulationType = Field(
        title="Time modulation",
        description="Time modulation part from the separable SpaceTimeModulation.",
    )

    @cached_property
    def max_modulation(self) -> float:
        """Estimated maximum modulation amplitude."""
        return self.time_modulation.max_modulation * self.space_modulation.max_modulation

    @cached_property
    def negligible_modulation(self) -> bool:
        """whether the modulation is weak enough to be regarded as zero."""
        # if isclose(np.diff(time_modulation.range), 0) and
        if isclose(self.max_modulation, 0):
            return True
        return False

    def sel_inside(self, bounds: Bound) -> Self:
        """Return a new space-time modulation that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        SpaceTimeModulation
            SpaceTimeModulation with reduced data.
        """

        return self.updated_copy(space_modulation=self.space_modulation.sel_inside(bounds))


class ModulationSpec(Tidy3dBaseModel):
    """Specification adding space-time modulation to the non-dispersive part of medium
    including relative permittivity at infinite frequency and electric conductivity.
    """

    permittivity: SpaceTimeModulation | None = Field(
        None,
        title="Space-time modulation of relative permittivity",
        description="Space-time modulation of relative permittivity at infinite frequency "
        "applied on top of the base permittivity at infinite frequency.",
    )

    conductivity: SpaceTimeModulation | None = Field(
        None,
        title="Space-time modulation of conductivity",
        description="Space-time modulation of electric conductivity "
        "applied on top of the base conductivity.",
    )

    @model_validator(mode="after")
    def _check_same_modulation_frequency(self) -> Self:
        """Assert same time-modulation applied to permittivity and conductivity."""
        if self.conductivity is not None and self.permittivity is not None:
            if self.conductivity.time_modulation != self.permittivity.time_modulation:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        "'permittivity' and 'conductivity' should have the same time modulation."
                    ),
                    "conductivity",
                )
        return self

    @cached_property
    def applied_modulation(self) -> bool:
        """Check if any modulation has been applied to ``permittivity`` or ``conductivity``."""
        return self.permittivity is not None or self.conductivity is not None

    def sel_inside(self, bounds: Bound) -> Self:
        """Return a new modulation specficiation that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        ModulationSpec
            ModulationSpec with reduced data.
        """

        perm_reduced = None
        if self.permittivity is not None:
            perm_reduced = self.permittivity.sel_inside(bounds)

        cond_reduced = None
        if self.conductivity is not None:
            cond_reduced = self.conductivity.sel_inside(bounds)

        return self.updated_copy(permittivity=perm_reduced, conductivity=cond_reduced)
