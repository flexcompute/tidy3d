"""Defines properties of good conductor"""

from __future__ import annotations

from typing import Union

import numpy as np
from pydantic.v1 import Field, NonNegativeFloat, PositiveInt, validator

from ..constants import C_0, HERTZ
from ..exceptions import SetupError, ValidationError
from .base import Tidy3dBaseModel, cached_property
from .fitter.fit_fast import DEFAULT_MAX_POLES, DEFAULT_TOLERANCE_RMS, FastDispersionFitter
from .medium import Medium, MediumType3DTmp, MediumTypeTmp, PoleResidue
from .types import Ax, FreqBound
from .viz import add_ax_if_none

DEFAULT_SAMPLING_FREQUENCY = 20
SCALED_REAL_PART = 10.0


class SkinDepthFitterParam(Tidy3dBaseModel):
    """Advanced parameters for fitting complex-valued skin depth ``2j/k`` over its frequency
    bandwidth, where k is the complex-valued wavenumber inside the lossy metal. Real part
    of this quantity corresponds to physical skin depth.
    """

    max_num_poles: PositiveInt = Field(
        DEFAULT_MAX_POLES,
        title="Maximal number of poles",
        description="Maximal number of poles in complex-conjugate poles residue model for "
        "fitting complex-valued skin depth.",
    )

    tolerance_rms: NonNegativeFloat = Field(
        DEFAULT_TOLERANCE_RMS,
        title="Tolerance in fitting",
        description="Tolerance in fitting complex-valued skin depth.",
    )

    frequency_sampling_points: PositiveInt = Field(
        DEFAULT_SAMPLING_FREQUENCY,
        title="Number of sampling frequencies",
        description="Number of sampling frequencies used in fitting.",
    )


class LossyMetal(Medium):
    """Material with high DC-conductivity that can be modeled with surface
    impedance boundary condition (SIBC). The model is accurate when the skin depth
    is much smaller than the structure feature size.
    """

    frequency_range: FreqBound = Field(
        ...,
        title="Frequency Range",
        description="Frequency range of validity for the medium.",
        units=(HERTZ, HERTZ),
    )

    fit_param: SkinDepthFitterParam = Field(
        SkinDepthFitterParam(),
        title="Complex-valued skin depth fitting parameters",
        description="Parameters for fitting complex-valued dispersive skin depth over "
        "the frequency range by using pole-residue pair model.",
    )

    @validator("frequency_range")
    def _validate_frequency_range(cls, val):
        """Validate that frequency range is finite and non-zero."""
        for freq in val:
            if not np.isfinite(freq):
                raise ValidationError("Values in 'frequency_range' must be finite.")
            if freq <= 0:
                raise ValidationError("Values in 'frequency_range' must be positive.")
        return val

    @cached_property
    def skin_depth_model(self) -> PoleResidue:
        """Fit complex-valued skin depth using pole-residue pair model within ``frequency_range``."""
        wvl_um = C_0 / self.sampling_frequencies
        skin_depth = self.complex_skin_depth(self.sampling_frequencies)

        # let's use scaled `skin_depth` in fitting: minimal real part equals ``SCALED_REAL_PART``
        min_skin_depth_real = np.min(skin_depth.real)
        if min_skin_depth_real <= 0:
            raise SetupError("Physical skin depth cannot be non-positive. Something is wrong.")

        scaling_factor = SCALED_REAL_PART / min_skin_depth_real
        skin_depth *= scaling_factor

        fitter = FastDispersionFitter.from_complex_permittivity(
            wvl_um, skin_depth.real, skin_depth.imag
        )
        material, _ = fitter.fit(
            max_num_poles=self.fit_param.max_num_poles, tolerance_rms=self.fit_param.tolerance_rms
        )
        return material._scaled_permittivity(1.0 / scaling_factor)

    @cached_property
    def num_poles(self) -> int:
        """Number of poles in the fitted model."""
        return len(self.skin_depth_model.poles)

    def complex_skin_depth(self, frequencies):
        """Compute complex-valued skin_depth."""
        # compute complex-valued skin depth
        n, k = self.nk_model(frequencies)
        wavenumber = 2 * np.pi * frequencies * (n + 1j * k) / C_0
        return 2j / wavenumber

    @cached_property
    def sampling_frequencies(self):
        """Sampling frequencies used in fitting."""
        if self.fit_param.frequency_sampling_points < 2:
            return np.array([np.mean(self.frequency_range)])

        return np.logspace(
            np.log10(self.frequency_range[0]),
            np.log10(self.frequency_range[1]),
            self.fit_param.frequency_sampling_points,
        )

    @add_ax_if_none
    def plot(
        self,
        ax: Ax = None,
    ) -> Ax:
        """Make plot of model vs data, at a set of wavelengths (if supplied).

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Axes to plot the data on, if None, a new one is created.

        Returns
        -------
        matplotlib.axis.Axes
            Matplotlib axis corresponding to plot.
        """
        frequencies = self.sampling_frequencies
        skin_depth = self.complex_skin_depth(frequencies)

        ax.plot(frequencies, skin_depth.real, "x", label="Real")
        ax.plot(frequencies, skin_depth.imag, "+", label="Imag")

        skin_depth_from_model = self.skin_depth_model.eps_model(frequencies)
        ax.plot(frequencies, skin_depth_from_model.real, label="Real (model)")
        ax.plot(frequencies, skin_depth_from_model.imag, label="Imag (model)")

        ax.set_ylabel("Skin depth")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()

        return ax


MediumType = Union[MediumTypeTmp, LossyMetal]
MediumType3D = Union[MediumType3DTmp, LossyMetal]
