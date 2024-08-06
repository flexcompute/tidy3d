"""Fit PoleResidue Dispersion models to optical NK data"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pydantic.v1 import NonNegativeFloat, PositiveInt

from ...components.dispersion_fitter import AdvancedFastFitterParam, fit
from ...components.medium import PoleResidue
from ...constants import C_0, HBAR
from .fit import DispersionFitter

# numerical tolerance for pole relocation for fast fitter
TOL = 1e-8
# numerical cutoff for passivity testing
CUTOFF = np.finfo(np.float32).eps
# parameters for passivity optimization
PASSIVITY_NUM_ITERS_DEFAULT = 50
SLSQP_CONSTRAINT_SCALE_DEFAULT = 1e35
# min value of the rms for default weights calculated based on Re[eps], Im[eps]
RMS_MIN = 0.1

DEFAULT_MAX_POLES = 5
DEFAULT_NUM_ITERS = 20
DEFAULT_TOLERANCE_RMS = 1e-5

# this avoids divide by zero errors with lossless poles
SCALE_FACTOR = 1.01

# when poles are close to omega, can cause invalid response function, and we reject model
OMEGA_POLE_CLOSE_ATOL = 1e-10


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

        omega_data = PoleResidue.Hz_to_angular_freq(self.freqs)
        eps_data = self.eps_data
        scale_factor = HBAR

        params, error = fit(
            omega_data=omega_data,
            resp_data=eps_data,
            min_num_poles=min_num_poles,
            max_num_poles=max_num_poles,
            resp_inf=eps_inf,
            tolerance_rms=tolerance_rms,
            advanced_param=advanced_param,
            scale_factor=scale_factor,
        )

        eps_inf, poles, residues = params

        medium = PoleResidue(eps_inf=eps_inf, poles=list(zip(poles, residues)))

        return medium, error

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
