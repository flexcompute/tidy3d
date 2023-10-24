""" Defines time modulation to the medium"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union
from math import isclose

import pydantic.v1 as pd
import numpy as np

from .base import Tidy3dBaseModel, cached_property
from .types import InterpMethod
from .time import AbstractTimeDependence
from .data.data_array import SpatialDataArray
from ..exceptions import ValidationError
from ..constants import HERTZ, RADIAN


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

    freq0: pd.PositiveFloat = pd.Field(
        ..., title="Modulation Frequency", description="Modulation frequency.", units=HERTZ
    )

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        omega = 2 * np.pi * self.freq0
        return self.amplitude * np.exp(-1j * omega * time + 1j * self.phase)

    @cached_property
    def max_modulation(self) -> float:
        """Estimated maximum modulation amplitude."""
        return abs(self.amplitude)


TimeModulationType = Union[ContinuousWaveTimeModulation]


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

    amplitude: Union[float, SpatialDataArray] = pd.Field(
        1,
        title="Amplitude of modulation in space",
        description="Amplitude of modulation that can vary spatially. "
        "It takes the unit of whatever is being modulated.",
    )

    phase: Union[float, SpatialDataArray] = pd.Field(
        0,
        title="Phase of modulation in space",
        description="Phase of modulation that can vary spatially.",
        units=RADIAN,
    )

    interp_method: InterpMethod = pd.Field(
        "nearest",
        title="Interpolation method",
        description="Method of interpolation to use to obtain values at spatial locations on the Yee grids.",
    )

    @pd.validator("amplitude", always=True)
    def _real_amplitude(cls, val):
        """Assert that the amplitude is real."""
        if np.iscomplexobj(val):
            raise ValidationError("'amplitude' must be real.")
        return val

    @pd.validator("phase", always=True)
    def _real_phase(cls, val):
        """Assert that the phase is real."""
        if np.iscomplexobj(val):
            raise ValidationError("'phase' must be real.")
        return val

    @cached_property
    def max_modulation(self) -> float:
        """Estimated maximum modulation amplitude."""
        return np.max(abs(np.array(self.amplitude)))


SpaceModulationType = Union[SpaceModulation]


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

    space_modulation: SpaceModulationType = pd.Field(
        SpaceModulation(),
        title="Space modulation",
        description="Space modulation part from the separable SpaceTimeModulation.",
        # discriminator=TYPE_TAG_STR,
    )

    time_modulation: TimeModulationType = pd.Field(
        ...,
        title="Time modulation",
        description="Time modulation part from the separable SpaceTimeModulation.",
        # discriminator=TYPE_TAG_STR,
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


class ModulationSpec(Tidy3dBaseModel):
    """Specification adding space-time modulation to the non-dispersive part of medium
    including relative permittivity at infinite frequency and electric conductivity.
    """

    permittivity: SpaceTimeModulation = pd.Field(
        None,
        title="Space-time modulation of relative permittivity",
        description="Space-time modulation of relative permittivity at infinite frequency "
        "applied on top of the base permittivity at infinite frequency.",
    )

    conductivity: SpaceTimeModulation = pd.Field(
        None,
        title="Space-time modulation of conductivity",
        description="Space-time modulation of electric conductivity "
        "applied on top of the base conductivity.",
    )

    @pd.validator("conductivity", always=True)
    def _same_modulation_frequency(cls, val, values):
        """Assert same time-modulation applied to permittivity and conductivity."""
        permittivity = values.get("permittivity")
        if val is not None and permittivity is not None:
            if val.time_modulation != permittivity.time_modulation:
                raise ValidationError(
                    "'permittivity' and 'conductivity' should have the same time modulation."
                )
        return val

    @cached_property
    def applied_modulation(self) -> bool:
        """Check if any modulation has been applied to `permittivity` or `conductivity`."""
        return self.permittivity is not None or self.conductivity is not None
