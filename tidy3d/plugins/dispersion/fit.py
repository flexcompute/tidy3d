""" basically copy and paste contents from fit.py with some modifications """

from typing import Tuple

from tqdm import tqdm
import nlopt
import numpy as np
import matplotlib.pylab as plt

from ...components import PoleResidue, nk_to_eps_complex, eps_complex_to_nk
from ...constants import C_0, HBAR


def _unpack_complex(complex_num):
    """returns real and imaginary parts from complex number"""
    return complex_num.real, complex_num.imag


def _pack_complex(real_part, imag_part):
    """returns complex number from real and imaginary parts"""
    return real_part + 1j * imag_part


def _unpack_coeffs(coeffs):
    """unpacks coefficient vector into complex a and c pole parameters"""
    assert len(coeffs) % 4 == 0
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
    """packs complex a and c pole parameters into coefficient vector"""
    a_real, a_imag = _unpack_complex(pole_a)
    c_real, c_imag = _unpack_complex(pole_c)
    stacked_coeffs = np.stack((a_real, a_imag, c_real, c_imag), axis=1)
    return stacked_coeffs.flatten()


def _coeffs_to_poles(coeffs):
    """Converts model coefficients to poles"""
    coeffs_scaled = coeffs / HBAR
    poles_a, poles_c = _unpack_coeffs(coeffs_scaled)
    poles = [(complex(a), complex(c)) for (a, c) in zip(poles_a, poles_c)]
    return poles


def _poles_to_coeffs(poles):
    """Converts poles to model coefficients"""
    poles_a, poles_c = np.array([[a, c] for (a, c) in poles]).T
    coeffs = _pack_coeffs(poles_a, poles_c)
    return coeffs * HBAR


class DispersionFitter:
    """Tool for fitting raw nk data to get Dispersive Medium"""

    def __init__(self, wvl_um: np.ndarray, n_data: np.ndarray, k_data: np.ndarray = None):
        """initialize fitter with raw data"""

        self._validate_data(wvl_um, n_data, k_data)
        self.wvl_um = wvl_um
        self.n_data = n_data
        if k_data is None:
            self.k_data = np.zeros_like(n_data)
            self.lossy = False
        else:
            self.k_data = k_data
            self.lossy = True
        self.eps_data = nk_to_eps_complex(n=self.n_data, k=self.k_data)
        self.freqs = C_0 / wvl_um
        self.frequency_range = (np.min(self.freqs), np.max(self.freqs))

    @staticmethod
    def _validate_data(wvl_um: np.ndarray, n_data: np.ndarray, k_data: np.ndarray = None):
        """make sure data is correctly shaped"""
        assert wvl_um.shape == n_data.shape
        if k_data is not None:
            assert wvl_um.shape == k_data.shape

    def fit(
        self,
        num_poles: int = 3,
        num_tries: int = 100,
        tolerance_rms: float = 0.0,
        verbose: bool = True,
    ) -> Tuple[PoleResidue, float]:
        """Fits data a number of times and returns best results.

        Parameters
        ----------
        num_poles : int, optional
            Number of poles in model.
        num_tries : int, optional
            Number of optimizations to run with different initial guess.
        tolerance_rms : float, optional
            RMS error below which the fit is successful and result is returned.
        plot : bool, optional
            Plot the results at the end.
        verbose: bool, optional
            Whether to print out information about fit

        Returns
        -------
        PoleResidue, rms_error
            Medium containing fit result and the corresponding error of the fit
        """

        # Run it a number of times.
        best_medium = None
        best_rms = np.inf
        pbar = tqdm(range(num_tries)) if verbose else range(num_tries)
        for _ in pbar:
            medium, rms_error = self.fit_single(num_poles=num_poles)

            # if improvement, set the best RMS and coeffs
            if rms_error < best_rms:
                best_rms = rms_error
                best_medium = medium

            # update status
            if verbose:
                pbar.set_description(f"best RMS error so far: {best_rms:.2e}")

            # if below tolerance, return
            if best_rms < tolerance_rms:
                if verbose:
                    print(f"\tfound optimal fit with RMS error = {best_rms:.2e}, returning")
                return best_medium, best_rms

        # if exited loop, did not reach tolerance (warn)
        if verbose:
            print(
                f"\twarning: did not find fit"
                f"with RMS error under tolerance_rms of {tolerance_rms:.2e}"
            )
            print(f"\treturning best fit with RMS error {best_rms:.2e}")
        return best_medium, best_rms

    def _make_medium(self, coeffs):
        """returns medium from coeffs from optimizer"""
        poles_complex = _coeffs_to_poles(coeffs)
        poles_re_im = [(_unpack_complex(a), _unpack_complex(c)) for (a, c) in poles_complex]
        return PoleResidue(poles=poles_re_im, frequency_range=self.frequency_range)

    def fit_single(
        self,
        num_poles: int = 3,
    ) -> Tuple[PoleResidue, float]:
        """Perform a single fit to the data and return optimization result."""

        def constraint(coeffs, _grad):
            """Evaluates the nonlinear stability criterion of
            Hongjin Choi, Jae-Woo Baek, and Kyung-Young Jung,
            "Comprehensive Study on Numerical Aspects of Modified
            Lorentz Model Based Dispersive FDTD Formulations,"
            IEEE TAP 2019.  Note: not used."""
            poles_a, poles_c = _unpack_coeffs(coeffs)
            a_real, a_imag = _unpack_complex(poles_a)
            c_real, c_imag = _unpack_complex(poles_c)
            prstar = a_real * c_real + a_imag * c_imag
            res = 2 * prstar * a_real - c_real * (a_real * a_real + a_imag * a_imag)
            res[res >= 0] = 0
            return np.sum(res)

        def obj(coeffs, _grad):
            """objective function for fit"""

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

    def plot(
        self,
        medium: PoleResidue = None,
        wvl_um: np.ndarray = None,
        axis=None,
    ):
        """Make plot of model vs data, at a set of wavelengths (if supplied).

        Parameters
        ----------
        medium: td.PoleReside
            medium containing model to plot against data
        wvl_um : array-like, optional
            (micron) wavelengths to evaluate modeal at.
        ax : matplotlib.axis.Axes, optional
            axis to plot the data on.
        dot_sizes : float, optional
            Size of input data scatter plots.
        linewidth : float, optional
            Width of model plot lines.
        n_data_color : str, optional
            Color (matplotlib) of n data.
        k_data_color : str, optional
            Color (matplotlib) of k data.
        n_model_color : str, optional
            Color (matplotlib) of n model.
        k_model_color : str, optional
            Color (matplotlib) of k model.

        Returns
        -------
        Matplotlib image object.
        """

        if wvl_um is None:
            wvl_um = self.wvl_um

        freqs = C_0 / wvl_um
        eps_model = medium.eps_model(freqs)
        n_model, k_model = eps_complex_to_nk(eps_model)

        if axis is None:
            _, axis = plt.subplots(1, 1)

        dot_sizes = 25
        linewidth = 3

        image = axis.scatter(self.wvl_um, self.n_data, s=dot_sizes, c="black", label="n (data)")
        axis.plot(wvl_um, n_model, linewidth=linewidth, color="crimson", label="n (model)")

        if self.lossy:
            axis.scatter(self.wvl_um, self.k_data, s=dot_sizes, c="black", label="k (data)")
            axis.plot(wvl_um, k_model, linewidth=linewidth, color="blueviolet", label="k (model)")

        axis.set_ylabel("value")
        axis.set_xlabel("Wavelength ($\\mu m$)")
        axis.legend()

        return image

    @classmethod
    def load(cls, fname, **loadtxt_kwargs):
        """Loads nk data from file, performs validation on input. wvl_um
        must be in micron.

        Parameters
        ----------
        fname : str
            Path to file containing wavelength (um), n, k (optional) data in columns.
        **loadtxt_kwargs
            Kwargs passed to ``np.loadtxt``.

        Returns
        -------
        dispersion fitter
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
