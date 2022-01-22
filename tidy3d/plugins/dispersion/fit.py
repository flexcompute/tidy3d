"""Fit PoleResidue Dispersion models to optical NK data
"""

from typing import Tuple

import nlopt
import numpy as np
from rich.progress import Progress

from ...components import PoleResidue, AbstractMedium
from ...constants import C_0, HBAR
from ...components.viz import add_ax_if_none
from ...components.types import Ax, Numpy
from ...log import log


def _unpack_complex(complex_num):
    """Returns real and imaginary parts from complex number.

    Parameters
    ----------
    complex_num : complex
        Complex number.

    Returns
    -------
    Tuple[float, float]
        Real and imaginary parts of the complex number.
    """
    return complex_num.real, complex_num.imag


def _pack_complex(real_part, imag_part):
    """Returns complex number from real and imaginary parts.

    Parameters
    ----------
    real_part : float
        Real part of the complex number.
    imag_part : float
        Imaginary part of the complex number.

    Returns
    -------
    complex
        The complex number.
    """
    return real_part + 1j * imag_part


def _unpack_coeffs(coeffs):
    """Unpacks coefficient vector into complex pole parameters.

    Parameters
    ----------
    coeffs : np.ndarray[real]
        Array of real coefficients for the pole residue fit.

    Returns
    -------
    Tuple[np.ndarray[complex], np.ndarray[complex]]
        "a" and "c" poles for the PoleResidue model.
    """
    assert len(coeffs) % 4 == 0, "len(coeffs) must be multiple of 4."
    num_poles = len(coeffs) // 4
    indices = 4 * np.arange(num_poles)

    a_real = coeffs[indices + 0]
    a_imag = coeffs[indices + 1]
    c_real = coeffs[indices + 2]
    c_imag = coeffs[indices + 3]

    poles_a = _pack_complex(a_real, a_imag)
    poles_c = _pack_complex(c_real, c_imag)
    return poles_a, poles_c


def _pack_coeffs(pole_a, pole_c):
    """Packs complex a and c pole parameters into coefficient array.

    Parameters
    ----------
    pole_a : np.ndarray[complex]
        Array of complex "a" poles for the PoleResidue dispersive model.
    pole_c : np.ndarray[complex]
        Array of complex "c" poles for the PoleResidue dispersive model.

    Returns
    -------
    np.ndarray[float]
        Array of real coefficients for the pole residue fit.
    """
    a_real, a_imag = _unpack_complex(pole_a)
    c_real, c_imag = _unpack_complex(pole_c)
    stacked_coeffs = np.stack((a_real, a_imag, c_real, c_imag), axis=1)
    return stacked_coeffs.flatten()


def _coeffs_to_poles(coeffs):
    """Converts model coefficients to poles.

    Parameters
    ----------
    coeffs : np.ndarray[float]
        Array of real coefficients for the pole residue fit.

    Returns
    -------
    List[Tuple[complex, complex]]
        List of complex poles (a, c)
    """
    coeffs_scaled = coeffs / HBAR
    poles_a, poles_c = _unpack_coeffs(coeffs_scaled)
    poles = [(complex(a), complex(c)) for (a, c) in zip(poles_a, poles_c)]
    # poles = [((a.real, a.imag), (c.real, c.imag)) for (a, c) in zip(poles_a, poles_c)]
    return poles


def _poles_to_coeffs(poles):
    """Converts poles to model coefficients.

    Parameters
    ----------
    poles : List[Tuple[complex, complex]]
        List of complex poles (a, c)

    Returns
    -------
    np.ndarray[float]
        Array of real coefficients for the pole residue fit.
    """
    poles_a, poles_c = np.array([[a, c] for (a, c) in poles]).T
    coeffs = _pack_coeffs(poles_a, poles_c)
    return coeffs * HBAR


class DispersionFitter:
    """Tool for fitting refractive index data to get a dispersive ``Medium``."""

    def __init__(self, wvl_um: Numpy, n_data: Numpy, k_data: Numpy = None):
        """Make a ``DispersionFitter`` with raw wavelength-nk data.

        Parameters
        ----------
        wvl_um : Numpy
            Wavelength data in micrometers.
        n_data : Numpy
            Real part of refractive index in micrometers.
        k_data : Numpy, optional
            Imaginary part of refractive index in micrometers.
        """

        self._validate_data(wvl_um, n_data, k_data)
        self.wvl_um = wvl_um
        self.n_data = n_data
        self.k_data = k_data
        self.lossy = True

        # handle lossless case
        if k_data is None:
            self.k_data = np.zeros_like(n_data)
            self.lossy = False
        self.eps_data = AbstractMedium.nk_to_eps_complex(n=self.n_data, k=self.k_data)
        self.freqs = C_0 / wvl_um
        self.frequency_range = (np.min(self.freqs), np.max(self.freqs))

    @staticmethod
    def _validate_data(wvl_um: Numpy, n_data: Numpy, k_data: Numpy = None):
        """make sure raw data is correctly shaped.

        Parameters
        ----------
        wvl_um : Numpy
            Wavelength data in micrometers.
        n_data : Numpy
            Real part of refractive index in micrometers.
        k_data : Numpy, optional
            Imaginary part of refractive index in micrometers.
        """
        assert wvl_um.shape == n_data.shape
        if k_data is not None:
            assert wvl_um.shape == k_data.shape

    def fit(
        self,
        num_poles: int = 3,
        num_tries: int = 100,
        tolerance_rms: float = 0.0,
    ) -> Tuple[PoleResidue, float]:
        """Fits data a number of times and returns best results.

        Parameters
        ----------
        num_poles : int, optional
            Number of poles in the model.
        num_tries : int, optional
            Number of optimizations to run with random initial guess.
        tolerance_rms : float, optional
            RMS error below which the fit is successful and the result is returned.

        Returns
        -------
        Tuple[``PoleResidue``, float]
            Best results of multiple fits: (dispersive medium, RMS error).
        """

        # Run it a number of times.
        best_medium = None
        best_rms = np.inf

        with Progress() as progress:

            task = progress.add_task(
                f"Fitting with {num_poles} to RMS of {tolerance_rms}...", total=num_tries
            )

            while not progress.finished:

                medium, rms_error = self.fit_single(num_poles=num_poles)

                # if improvement, set the best RMS and coeffs
                if rms_error < best_rms:
                    best_rms = rms_error
                    best_medium = medium

                progress.update(
                    task, advance=1, description=f"best RMS error so far: {best_rms:.2e}"
                )

                # if below tolerance, return
                if best_rms < tolerance_rms:
                    log.info(f"\tfound optimal fit with RMS error = {best_rms:.2e}, returning")
                    return best_medium, best_rms

        # if exited loop, did not reach tolerance (warn)
        log.warning(
            f"\twarning: did not find fit "
            f"with RMS error under tolerance_rms of {tolerance_rms:.2e}"
        )
        log.info(f"\treturning best fit with RMS error {best_rms:.2e}")
        return best_medium, best_rms

    def _make_medium(self, coeffs):
        """returns medium from coeffs from optimizer

        Parameters
        ----------
        coeffs : np.ndarray[float]
            Array of real coefficients for the pole residue fit.

        Returns
        -------
        ``PoleResidue``
            Dispersive medium corresponding to this set of ``coeffs``.
        """
        poles_complex = _coeffs_to_poles(coeffs)
        return PoleResidue(poles=poles_complex, frequency_range=self.frequency_range)

    def fit_single(
        self,
        num_poles: int = 3,
    ) -> Tuple[PoleResidue, float]:
        """Perform a single fit to the data and return optimization result.

        Parameters
        ----------
        num_poles : int, optional
            Number of poles in the model.

        Returns
        -------
        Tuple[``PoleResidue``, float]
            Results of single fit: (dispersive medium, RMS error).
        """

        def constraint(coeffs, _grad):
            """Evaluates the nonlinear stability criterion of
            Hongjin Choi, Jae-Woo Baek, and Kyung-Young Jung,
            "Comprehensive Study on Numerical Aspects of Modified
            Lorentz Model Based Dispersive FDTD Formulations,"
            IEEE TAP 2019.  Note: not used.

            Parameters
            ----------
            coeffs : np.ndarray[float]
                Array of real coefficients for the pole residue fit.
            _grad : np.ndarray[float]
                Gradient of ``constraint`` w.r.t coeffs, not used.

            Returns
            -------
            float
                Value of constraint.
            """
            poles_a, poles_c = _unpack_coeffs(coeffs)
            a_real, a_imag = _unpack_complex(poles_a)
            c_real, c_imag = _unpack_complex(poles_c)
            prstar = a_real * c_real + a_imag * c_imag
            res = 2 * prstar * a_real - c_real * (a_real * a_real + a_imag * a_imag)
            res[res >= 0] = 0
            return np.sum(res)

        def obj(coeffs, _grad):
            """objective function for fit

            Parameters
            ----------
            coeffs : np.ndarray[float]
                Array of real coefficients for the pole residue fit.
            _grad : np.ndarray[float]
                Gradient of ``obj`` w.r.t coeffs, not used.

            Returns
            -------
            float
                RMS error correponding to current coeffs.
            """

            medium = self._make_medium(coeffs)
            eps_model = medium.eps_model(self.freqs)
            residual = self.eps_data - eps_model
            rms_error = np.sqrt(np.sum(np.square(np.abs(residual))) / len(self.eps_data))
            # cons = constraint(coeffs, _grad)
            return rms_error

        # set initial guess
        num_coeffs = num_poles * 4
        coeffs0 = 2 * (np.random.random(num_coeffs) - 0.5)

        # set method and objective
        method = nlopt.LN_NELDERMEAD
        opt = nlopt.opt(method, num_coeffs)
        opt.set_min_objective(obj)

        # set bounds
        bounds_upper = np.zeros(num_coeffs, dtype=float)
        bounds_lower = np.zeros(num_coeffs, dtype=float)
        indices = 4 * np.arange(num_poles)

        if self.lossy:
            # if lossy, the real parts can take on values
            bounds_lower[indices] = -np.inf
            bounds_upper[indices + 2] = np.inf
            coeffs0[indices] = -np.abs(coeffs0[indices])
            coeffs0[indices + 2] = +np.abs(coeffs0[indices + 2])
        else:
            # otherwise, they need to be 0
            coeffs0[indices] = 0
            coeffs0[indices + 2] = 0

        bounds_lower[indices + 1] = -np.inf
        bounds_upper[indices + 1] = np.inf
        bounds_lower[indices + 3] = -np.inf
        bounds_upper[indices + 3] = np.inf

        opt.set_lower_bounds(bounds_lower.tolist())
        opt.set_upper_bounds(bounds_upper.tolist())

        # opt.add_inequality_constraint(constraint)
        opt.set_xtol_rel(1e-5)
        opt.set_ftol_rel(1e-5)

        # run global optimization with opt as inner loop
        optglob = nlopt.opt(nlopt.LN_AUGLAG, num_coeffs)
        optglob.set_min_objective(obj)
        optglob.set_lower_bounds(bounds_lower.tolist())
        optglob.set_upper_bounds(bounds_upper.tolist())
        optglob.add_inequality_constraint(constraint)
        optglob.set_xtol_rel(1e-5)
        optglob.set_maxeval(10000)
        optglob.set_ftol_rel(1e-7)
        optglob.set_local_optimizer(opt)
        coeffs = optglob.optimize(coeffs0)
        rms_error = optglob.last_optimum_value()

        # set the latest fit
        medium = self._make_medium(coeffs)
        return medium, rms_error

    @add_ax_if_none
    def plot(
        self,
        medium: PoleResidue = None,
        wvl_um: Numpy = None,
        ax: Ax = None,
    ) -> Ax:
        """Make plot of model vs data, at a set of wavelengths (if supplied).

        Parameters
        ----------
        medium : PoleResidue, optional
            medium containing model to plot against data
        wvl_um : Numpy, optional
            Wavelengths to evaluate model at for plot in micrometers.
        ax : Ax, optional
            Axes to plot the data on, if None, a new one is created.

        Returns
        -------
        matplotlib.axis.Axes
            Matplotlib axis corresponding to plot.
        """

        if wvl_um is None:
            wvl_um = self.wvl_um

        freqs = C_0 / wvl_um
        eps_model = medium.eps_model(freqs)
        n_model, k_model = AbstractMedium.eps_complex_to_nk(eps_model)

        dot_sizes = 25
        linewidth = 3

        _ = ax.scatter(self.wvl_um, self.n_data, s=dot_sizes, c="black", label="n (data)")
        ax.plot(wvl_um, n_model, linewidth=linewidth, color="crimson", label="n (model)")

        if self.lossy:
            ax.scatter(self.wvl_um, self.k_data, s=dot_sizes, c="black", label="k (data)")
            ax.plot(wvl_um, k_model, linewidth=linewidth, color="blueviolet", label="k (model)")

        ax.set_ylabel("value")
        ax.set_xlabel("Wavelength ($\\mu m$)")
        ax.legend()

        return ax

    @classmethod
    def from_file(cls, fname, **loadtxt_kwargs):
        """Loads ``DispersionFitter`` from file contining wavelength, n, k data.

        Parameters
        ----------
        fname : str
            Path to file containing wavelength (um), n, k (optional) data in columns.
        **loadtxt_kwargs
            Kwargs passed to ``np.loadtxt``, such as ``skiprows``, ``delimiter``.

        Returns
        -------
        DispersionFitter
            A ``DispersionFitter`` instance.
        """
        data = np.loadtxt(fname, **loadtxt_kwargs)
        assert len(data.shape) == 2, "data must contain [wavelength, ndata, kdata] in columns"
        assert data.shape[-1] in (2, 3), "data must have either 2 or 3 rows (if k data)"
        if data.shape[-1] == 2:
            wvl_um, n_data = data.T
            k_data = None
        else:
            wvl_um, n_data, k_data = data.T
        return cls(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
