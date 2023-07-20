""" Defines time modulation to the medium"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Tuple
from math import isclose

import pydantic as pd
import numpy as np

from .base import Tidy3dBaseModel, cached_property
from .types import InterpMethod
from .data.data_array import SpatialDataArray
from ..exceptions import ValidationError
from ..constants import HERTZ


class AbstractTimeModulation(ABC, Tidy3dBaseModel):
    """Base class describing the time modulation applied to a medium, adding
    on top of the time-independent permittivity of the medium. The
    modulation can be spatially nonuniform.
    """

    interp_method: InterpMethod = pd.Field(
        "nearest",
        title="Interpolation method",
        description="Interpolation method to obtain modulation values "
        "that are not supplied at the Yee grids.",
    )

    @cached_property
    @abstractmethod
    def range(self) -> Tuple[float, float]:
        """Estimated minimal and maximal perturbation to the relative permittivity."""

    @cached_property
    @abstractmethod
    def negligible_modulation_speed(self) -> Tuple[float, float]:
        """whether the modulation is slow enough to be regarded as zero."""

    @cached_property
    def negligible_modulation_amp(self) -> bool:
        """whether the modulation is small enough to be regarded as zero."""
        if isclose(self.range[0], 0) and isclose(self.range[1], 0):
            return True
        return False

    @cached_property
    def negligible_modulation(self) -> bool:
        """whether the modulation is small or slow enough to be regarded as zero."""
        if self.negligible_modulation_amp or self.negligible_modulation_speed:
            return True
        return False

    @staticmethod
    def _validate_isreal_dataarray(dataarray: SpatialDataArray) -> bool:
        """Validate that the dataarray is real"""
        return np.all(np.isreal(dataarray.values))


class ContinuousWaveModulation(AbstractTimeModulation):
    """Continuous wave modulation of a parameter of the medium. Adding on top of the
    time-independent part of the medium, the time dependent
    part, e.g. relative permittivity, is described by,

    Note
    ----
    .. math::

        \\delta\\epsilon(t, r) = A(r) \\cos(\\omega t + \\phi(r))
    """

    freq: float = pd.Field(
        ...,
        title="Modulation frequency",
        description="Modulation frequency applied to the medium",
        units=HERTZ,
    )

    amplitude: Union[float, SpatialDataArray] = pd.Field(
        ...,
        title="Amplitude of modulation",
        description="Amplitude of modulation that can vary spatially.",
    )

    phase: Union[float, SpatialDataArray] = pd.Field(
        ...,
        title="Phase of modulation",
        description="Phase of modulation that can vary spatially.",
    )

    @pd.validator("amplitude", always=True)
    def _real_amplitude(cls, val):
        """Assert that the amplitude is real."""
        if isinstance(
            val, SpatialDataArray
        ) and not ContinuousWaveModulation._validate_isreal_dataarray(val):
            raise ValidationError("'amplitude' must be real.")
        return val

    @pd.validator("phase", always=True)
    def _real_phase(cls, val):
        """Assert that the phase is real."""
        if isinstance(
            val, SpatialDataArray
        ) and not ContinuousWaveModulation._validate_isreal_dataarray(val):
            raise ValidationError("'phase' must be real.")
        return val

    @pd.validator("phase", always=True)
    def _correct_shape(cls, val, values):
        """Assert phase and amplitude are defined over the same grids."""
        amp = values.get("amplitude")
        if amp is None:
            raise ValidationError("'amplitude' failed validation.")

        if isinstance(val, SpatialDataArray) and isinstance(amp, SpatialDataArray):
            if amp.coords != val.coords:
                raise ValidationError("'amplitude' and 'phase' must have the same coordinates.")
        return val

    @cached_property
    def _amplitude_array(self) -> SpatialDataArray:
        """Convert amplitude into SpatialDataArray if it's not."""
        if isinstance(self.amplitude, SpatialDataArray):
            return self.amplitude
        return SpatialDataArray(
            np.ones((1, 1, 1)) * self.amplitude, coords=dict(x=[0], y=[0], z=[0])
        )

    @cached_property
    def _phase_array(self) -> SpatialDataArray:
        """Convert phase into SpatialDataArray if it's not."""
        if isinstance(self.phase, SpatialDataArray):
            return self.phase
        return SpatialDataArray(np.ones((1, 1, 1)) * self.phase, coords=dict(x=[0], y=[0], z=[0]))

    @cached_property
    def range(self) -> Tuple[float, float]:
        """Estimated minimal and maximal perturbation to the relative permittivity."""
        amplitude_max = max(self._amplitude_array.values.ravel())
        return -amplitude_max, amplitude_max

    @cached_property
    def negligible_modulation_speed(self) -> Tuple[float, float]:
        """whether the modulation is slow enough to be regarded as zero."""
        if isclose(self.freq, 0):
            return True
        return False


# time modulation allowed in medium
TimeModulationType = Union[ContinuousWaveModulation]
