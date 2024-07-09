"""Fit PoleResidue Dispersion models to optical NK data"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tidy3d_fitter
from pydantic.v1 import Field, NonNegativeFloat, PositiveFloat, PositiveInt, validator

from ...components.base import Tidy3dBaseModel
from ...components.medium import PoleResidue
from ...constants import C_0, HBAR
from ...exceptions import ValidationError
from ...log import get_logging_console, log
from .fit import DispersionFitter

tidy3d_fitter.log = log

# parameters for passivity optimization
PASSIVITY_NUM_ITERS_DEFAULT = 50
SLSQP_CONSTRAINT_SCALE_DEFAULT = 1e35
# min value of the rms for default weights calculated based on Re[eps], Im[eps]
RMS_MIN = 0.1

DEFAULT_MAX_POLES = 5
DEFAULT_NUM_ITERS = 20
DEFAULT_TOLERANCE_RMS = 1e-5


class AdvancedFastFitterParam(Tidy3dBaseModel):
    """Advanced fast fitter parameters."""

    loss_bounds: Tuple[float, float] = Field(
        (0, np.inf),
        title="Loss bounds",
        description="Bounds (lower, upper) on Im[eps]. Default corresponds to only passivity. "
        "A lower bound of 0 or greater ensures passivity. To fit a gain medium without "
        "additional constraints, use ``loss_bounds=(-np.inf, np.inf)``. "
        "Increasing the lower bound could help with simulation stability. "
        "A finite upper bound may be helpful when fitting lossless materials. "
        "In this case, consider also increasing the weight for fitting the imaginary part.",
    )
    weights: Tuple[NonNegativeFloat, NonNegativeFloat] = Field(
        None,
        title="Weights",
        description="Weights (real, imag) in objective function for fitting. The weights "
        "are applied to the real and imaginary parts of the permittivity epsilon. The weights "
        "will be rescaled together so they average to 1. If ``None``, the weights are calculated "
        "according to the typical value of the real and imaginary part, so that the relative error "
        "in the real and imaginary part of the fit should be comparable. "
        "More precisely, the RMS value ``rms`` of the real and imaginary parts are "
        "calculated, and the default weights are 1 / max(``rms``, ``RMS_MIN``). "
        "Changing this can be helpful if fitting either the real or imaginary part is "
        "more important than the other.",
    )
    show_progress: bool = Field(
        True,
        title="Show progress bar",
        description="Whether to show progress bar during fitter run.",
    )
    show_unweighted_rms: bool = Field(
        False,
        title="Show unweighted RMS",
        description="Whether to show unweighted RMS error in addition to the default weighted "
        'RMS error. Requires ``td.config.logging_level = "INFO"``.',
    )
    relaxed: Optional[bool] = Field(
        None,
        title="Relaxed",
        description="Whether to use relaxed fitting algorithm, which "
        "has better pole relocation properties. If ``None``, will try both original and relaxed "
        "algorithms.",
    )
    smooth: Optional[bool] = Field(
        None,
        title="Smooth",
        description="Whether to use real starting poles, which can help when fitting smooth data. "
        "If ``None``, will try both real and complex starting poles.",
    )
    logspacing: Optional[bool] = Field(
        None,
        title="Log spacing",
        description="Whether to space the poles logarithmically. "
        "If ``None``, will try both log and linear spacing.",
    )

    # more technical parameters
    num_iters: PositiveInt = Field(
        DEFAULT_NUM_ITERS,
        title="Number of iterations",
        description="Number of iterations of the fitting algorithm. Make this smaller to "
        "speed up fitter, or make it larger to try to improve fit.",
    )
    passivity_num_iters: PositiveInt = Field(
        PASSIVITY_NUM_ITERS_DEFAULT,
        title="Number of loss bounds enforcement iterations",
        description="Number of loss bounds enforcement iterations of the fitting algorithm. "
        "Make this smaller to speed up fitter. There will be a warning if this value "
        "is too small. To fit a gain medium, use the ``loss_bounds`` parameter instead.",
    )
    slsqp_constraint_scale: PositiveFloat = Field(
        SLSQP_CONSTRAINT_SCALE_DEFAULT,
        title="Scale factor for SLSQP",
        description="Passivity constraint is weighted relative to fit quality by this factor, "
        "before running passivity optimization using the SLSQP algorithm. "
        "There will be a warning if this value is too small.",
    )

    @validator("loss_bounds", always=True)
    def _max_loss_geq_min_loss(cls, val):
        """Must have max_loss >= min_loss."""
        if val[0] > val[1]:
            raise ValidationError(
                "The loss lower bound cannot be larger than the loss upper bound."
            )
        return val

    @validator("weights", always=True)
    def _weights_average_to_one(cls, val):
        """Weights must average to one."""
        if val is None:
            return None
        avg = (val[0] + val[1]) / 2
        new_val = (val[0] / avg, val[1] / avg)
        return new_val


class FastDispersionFitter(DispersionFitter):
    """Tool for fitting refractive index data to get a
    dispersive medium described by :class:`.PoleResidue` model."""

    def fit(
        self,
        min_num_poles: PositiveInt = 1,
        max_num_poles: PositiveInt = DEFAULT_MAX_POLES,
        eps_inf: float = None,
        tolerance_rms: NonNegativeFloat = DEFAULT_TOLERANCE_RMS,
        advanced_param: AdvancedFastFitterParam = None,
    ) -> Tuple[PoleResidue, float]:
        """Fit data using a fast fitting algorithm.

        Note
        ----
        The algorithm is described in::

            B. Gustavsen and A. Semlyen, "Rational approximation
            of frequency domain responses by vector fitting,"
            IEEE Trans. Power. Deliv. 14, 3 (1999).

            B. Gustavsen, "Improving the pole relocation properties
            of vector fitting," IEEE Trans. Power Deliv. 21, 3 (2006).

            B. Gustavsen, "Enforcing Passivity for Admittance Matrices
            Approximated by Rational Functions," IEEE Trans. Power
            Syst. 16, 1 (2001).

        Note
        ----
        The fit is performed after weighting the real and imaginary parts,
        so the RMS error is also weighted accordingly. By default, the weights
        are chosen based on typical values of the data. To change this behavior,
        use 'AdvancedFastFitterParam.weights'.


        Parameters
        ----------
        min_num_poles: PositiveInt, optional
            Minimum number of poles in the model.
        max_num_poles: PositiveInt, optional
            Maximum number of poles in the model.
        eps_inf : float, optional
            Value of eps_inf to use in fit. If None, then eps_inf is also fit.
            Note: fitting eps_inf is not guaranteed to yield a global optimum, so
            the result may occasionally be better with a fixed value of eps_inf.
        tolerance_rms : float, optional
            Weighted RMS error below which the fit is successful and the result is returned.
        advanced_param : :class:`AdvancedFastFitterParam`, optional
            Advanced parameters for fitting.

        Returns
        -------
        Tuple[:class:`.PoleResidue`, float]
            Best fitting result: (dispersive medium, weighted RMS error).
        """
        omega_data = 2 * np.pi * self.freqs[::-1]
        eps_data = self.eps_data[::-1]
        fitter = tidy3d_fitter.DispersionFitter(omega_data=omega_data, resp_data=eps_data)
        fitter_advanced_param = None
        if advanced_param:
            fitter_advanced_param = tidy3d_fitter.AdvancedFitterParam(
                loss_bounds=advanced_param.loss_bounds,
                weights=advanced_param.weights,
                show_progress=advanced_param.show_progress,
                show_unweighted_rms=advanced_param.show_unweighted_rms,
                relaxed=advanced_param.relaxed,
                smooth=advanced_param.smooth,
                logspacing=advanced_param.logspacing,
                num_iters=advanced_param.num_iters,
                passivity_num_iters=advanced_param.passivity_num_iters,
                slsqp_constraint_scale=advanced_param.slsqp_constraint_scale,
            )
        res, err = fitter.fit(
            min_num_poles=min_num_poles,
            max_num_poles=max_num_poles,
            resp_inf=eps_inf,
            tolerance_rms=tolerance_rms,
            advanced_param=fitter_advanced_param,
            console=get_logging_console(),
            scale_factor=HBAR,
        )
        medium = PoleResidue(
            frequency_range=self.frequency_range,
            eps_inf=res.resp_inf,
            poles=res.poles,
        )
        return medium, err

    @classmethod
    def constant_loss_tangent_model(
        cls,
        eps_real: float,
        loss_tangent: float,
        frequency_range: Tuple[float, float],
        max_num_poles: PositiveInt = DEFAULT_MAX_POLES,
        number_sampling_frequency: PositiveInt = 10,
        tolerance_rms: NonNegativeFloat = DEFAULT_TOLERANCE_RMS,
    ) -> PoleResidue:
        """Fit a constant loss tangent material model.

        Parameters
        ----------
        eps_real : float
            Real part of permittivity
        loss_tangent : float
            Loss tangent.
        frequency_range : Tuple[float, float]
            Freqquency range for the material to exhibit constant loss tangent response.
        max_num_poles : PositiveInt, optional
            Maximum number of poles in the model.
        number_sampling_frequency : PositiveInt, optional
            Number of sampling frequencies to compute RMS error for fitting.
        tolerance_rms : float, optional
            Weighted RMS error below which the fit is successful and the result is returned.

        Returns
        -------
        :class:`.PoleResidue
            Best results of multiple fits.
        """
        if number_sampling_frequency < 2:
            frequencies = np.array([np.mean(frequency_range)])
        else:
            frequencies = np.linspace(
                frequency_range[0], frequency_range[1], number_sampling_frequency
            )
        wvl_um = C_0 / frequencies
        eps_real_array = np.ones_like(frequencies) * eps_real
        loss_tangent_array = np.ones_like(frequencies) * loss_tangent
        fitter = cls.from_loss_tangent(wvl_um, eps_real_array, loss_tangent_array)
        material, _ = fitter.fit(max_num_poles=max_num_poles, tolerance_rms=tolerance_rms)
        return material
