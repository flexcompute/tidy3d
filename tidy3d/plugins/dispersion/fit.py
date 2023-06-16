"""Fit PoleResidue Dispersion models to optical NK data
"""

from typing import Tuple, List, Optional
import csv
import codecs
import requests

import scipy.optimize as opt
import numpy as np
from rich.progress import Progress
from pydantic import Field, validator

from ...log import log
from ...components.base import Tidy3dBaseModel, cached_property
from ...components.medium import PoleResidue, AbstractMedium
from ...components.viz import add_ax_if_none
from ...components.types import Ax, ArrayFloat1D
from ...constants import C_0, HBAR, MICROMETER
from ...exceptions import ValidationError, WebError, SetupError
from ...web.environment import Env


class DispersionFitter(Tidy3dBaseModel):
    """Tool for fitting refractive index data to get a
    dispersive medium described by :class:`.PoleResidue` model."""

    wvl_um: ArrayFloat1D = Field(
        ...,
        title="Wavelength data",
        description="Wavelength data in micrometers.",
        units=MICROMETER,
    )

    n_data: ArrayFloat1D = Field(
        ...,
        title="Index of refraction data",
        description="Real part of the complex index of refraction.",
    )

    k_data: ArrayFloat1D = Field(
        None,
        title="Extinction coefficient data",
        description="Imaginary part of the complex index of refraction.",
    )

    wvl_range: Tuple[Optional[float], Optional[float]] = Field(
        (None, None),
        title="Wavelength range [wvl_min,wvl_max] for fitting",
        description="Truncate the wavelength, n and k data to the wavelength range '[wvl_min, "
        "wvl_max]' for fitting.",
        units=MICROMETER,
    )

    @validator("wvl_um", always=True)
    def _setup_wvl(cls, val):
        """Convert wvl_um to a numpy array."""
        if val.size == 0:
            raise ValidationError("Wavelength data cannot be empty.")
        return val

    @validator("n_data", always=True)
    def _ndata_length_match_wvl(cls, val, values):
        """Validate n_data"""
        if val.shape != values["wvl_um"].shape:
            raise ValidationError("The length of 'n_data' doesn't match 'wvl_um'.")
        return val

    @validator("k_data", always=True)
    def _kdata_setup_and_length_match(cls, val, values):
        """Validate the length of k_data, or setup k if it's None."""
        if val is None:
            return np.zeros_like(values["wvl_um"])
        if val.shape != values["wvl_um"].shape:
            raise ValidationError("The length of 'k_data' doesn't match 'wvl_um'.")
        return val

    @cached_property
    def data_in_range(self) -> Tuple[ArrayFloat1D, ArrayFloat1D, ArrayFloat1D]:
        """Filter the wavelength-nk data to wavelength range for fitting.

        Returns
        -------
        Tuple[ArrayFloat1D, ArrayFloat1D, ArrayFloat1D]
            Filtered wvl_um, n_data, k_data
        """

        ind_select = np.ones(self.wvl_um.shape, dtype=bool)
        if self.wvl_range[0] is not None:
            ind_select = np.logical_and(self.wvl_um >= self.wvl_range[0], ind_select)

        if self.wvl_range[1] is not None:
            ind_select = np.logical_and(self.wvl_um <= self.wvl_range[1], ind_select)

        if not np.any(ind_select):
            raise SetupError("No data within 'wvl_range'")

        return self.wvl_um[ind_select], self.n_data[ind_select], self.k_data[ind_select]

    @cached_property
    def lossy(self) -> bool:
        """Find out if the medium is lossy or lossless
        based on the filtered input data.

        Returns
        -------
        bool
            True for lossy medium; False for lossless medium
        """
        _, _, k_data = self.data_in_range
        return k_data is not None and np.any(k_data)

    @property
    def eps_data(self) -> complex:
        """Convert filtered input n(k) data into complex permittivity.

        Returns
        -------
        complex
            Complex-valued relative permittivty.
        """
        _, n_data, k_data = self.data_in_range
        return AbstractMedium.nk_to_eps_complex(n=n_data, k=k_data)

    @property
    def freqs(self) -> Tuple[float, ...]:
        """Convert filtered input wavelength data to frequency.

        Returns
        -------
        Tuple[float, ...]
            Frequency array converted from filtered input wavelength data
        """

        wvl_um, _, _ = self.data_in_range
        return C_0 / wvl_um

    @property
    def frequency_range(self) -> Tuple[float, float]:
        """Frequency range of filtered input data

        Returns
        -------
        Tuple[float, float]
            The minimal frequency and the maximal frequency
        """

        return self.freqs.min(), self.freqs.max()

    @staticmethod
    def _unpack_coeffs(coeffs):
        """Unpack coefficient vector into complex pole parameters.

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

        a_real = coeffs[0::4]
        a_imag = coeffs[1::4]
        c_real = coeffs[2::4]
        c_imag = coeffs[3::4]

        poles_a = a_real + 1j * a_imag
        poles_c = c_real + 1j * c_imag
        return poles_a, poles_c

    @staticmethod
    def _pack_coeffs(pole_a, pole_c):
        """Pack complex a and c pole parameters into coefficient array.

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
        stacked_coeffs = np.stack((pole_a.real, pole_a.imag, pole_c.real, pole_c.imag), axis=1)
        return stacked_coeffs.flatten()

    @staticmethod
    def _coeffs_to_poles(coeffs):
        """Convert model coefficients to poles.

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
        poles_a, poles_c = DispersionFitter._unpack_coeffs(coeffs_scaled)
        return list(zip(poles_a, poles_c))

    @staticmethod
    def _poles_to_coeffs(poles):
        """Convert poles to model coefficients.

        Parameters
        ----------
        poles : List[Tuple[complex, complex]]
            List of complex poles (a, c)

        Returns
        -------
        np.ndarray[float]
            Array of real coefficients for the pole residue fit.
        """
        poles = np.array(poles, dtype=complex)
        coeffs = DispersionFitter._pack_coeffs(poles[:, 0], poles[:, 1])
        return coeffs * HBAR

    @staticmethod
    def _eV_to_Hz(f_eV: float):  # pylint:disable=invalid-name
        """Convert frequency in unit of eV to Hz.

        Parameters
        ----------
        f_eV : float
            Frequency in unit of eV
        """
        return f_eV / (HBAR * 2 * np.pi)

    @staticmethod
    def _Hz_to_eV(f_Hz: float):  # pylint:disable=invalid-name
        """Convert frequency in unit of Hz to eV.

        Parameters
        ----------
        f_Hz : float
            Frequency in unit of Hz
        """
        return f_Hz * HBAR * 2 * np.pi

    def fit(
        self,
        num_poles: int = 1,
        num_tries: int = 50,
        tolerance_rms: float = 1e-2,
    ) -> Tuple[PoleResidue, float]:
        """Fit data a number of times and returns best results.

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
        Tuple[:class:`.PoleResidue`, float]
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

                medium, rms_error = self._fit_single(num_poles=num_poles)

                # if improvement, set the best RMS and coeffs
                if rms_error < best_rms:
                    best_rms = rms_error
                    best_medium = medium

                progress.update(
                    task,
                    advance=1,
                    description=f"Best RMS error so far: {best_rms:.3g}",
                    refresh=True,
                )

                # if below tolerance, return
                if best_rms < tolerance_rms:
                    progress.update(
                        task,
                        completed=num_tries,
                        description=f"Best RMS error: {best_rms:.3g}",
                        refresh=True,
                    )
                    log.info("Found optimal fit with RMS error %.3g", best_rms)
                    return best_medium, best_rms

        # if exited loop, did not reach tolerance (warn)
        log.warning("Unable to fit with RMS error under 'tolerance_rms' of %.3g", tolerance_rms)
        log.info("Returning best fit with RMS error %.3g", best_rms)
        return best_medium, best_rms

    def _make_medium(self, coeffs):
        """Return medium from coeffs from optimizer.

        Parameters
        ----------
        coeffs : np.ndarray[float]
            Array of real coefficients for the pole residue fit.

        Returns
        -------
        :class:`.PoleResidue`
            Dispersive medium corresponding to this set of ``coeffs``.
        """
        poles_complex = DispersionFitter._coeffs_to_poles(coeffs)
        return PoleResidue(poles=poles_complex, frequency_range=self.frequency_range)

    def _fit_single(
        self,
        num_poles: int = 3,
    ) -> Tuple[PoleResidue, float]:
        """Perform a single fit to the data and return optimization result.

        Parameters
        ----------
        num_poles : int = 3
            Number of poles in the model.

        Returns
        -------
        Tuple[:class:`.PoleResidue`, float]
            Results of single fit: (dispersive medium, RMS error).
        """

        # NOTE: Not used
        def constraint(coeffs, _grad=None):
            """Evaluate the nonlinear stability criterion of Hongjin Choi, Jae-Woo Baek, and
            Kyung-Young Jung, "Comprehensive Study on Numerical Aspects of Modified Lorentz Model
            Based Dispersive FDTD Formulations," IEEE TAP 2019.

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
            poles_a, poles_c = DispersionFitter._unpack_coeffs(coeffs)
            a_real = poles_a.real
            a_imag = poles_a.imag
            c_real = poles_c.real
            c_imag = poles_c.imag
            prstar = a_real * c_real + a_imag * c_imag
            res = 2 * prstar * a_real - c_real * (a_real * a_real + a_imag * a_imag)
            res[res >= 0] = 0
            return np.sum(res)

        def objective(coeffs, _grad=None):
            """Objective function for fit

            Parameters
            ----------
            coeffs : np.ndarray[float]
                Array of real coefficients for the pole residue fit.
            _grad : np.ndarray[float]
                Gradient of ``objective`` w.r.t coeffs, not used.

            Returns
            -------
            float
                RMS error correponding to current coeffs.
            """

            medium = self._make_medium(coeffs)
            eps_model = medium.eps_model(self.freqs)
            residual = self.eps_data - eps_model
            # cons = constraint(coeffs, _grad)
            return np.sqrt(np.sum(np.square(np.abs(residual))) / len(self.eps_data))

        # set initial guess
        num_coeffs = num_poles * 4
        coeffs0 = 2 * (np.random.random(num_coeffs) - 0.5)

        # set bounds
        bounds_upper = np.zeros(num_coeffs, dtype=float)
        bounds_lower = np.zeros(num_coeffs, dtype=float)

        if self.lossy:
            # if lossy, the real parts can take on values
            bounds_lower[0::4] = -np.inf
            bounds_upper[2::4] = np.inf
            coeffs0[0::4] = -np.abs(coeffs0[0::4])
            coeffs0[2::4] = +np.abs(coeffs0[2::4])
        else:
            # otherwise, they need to be 0
            coeffs0[0::2] = 0

        bounds_lower[1::2] = -np.inf
        bounds_upper[1::2] = np.inf

        bounds = list(zip(bounds_lower, bounds_upper))

        # TODO: set up constraint properly
        scipy_constraint = opt.NonlinearConstraint(constraint, lb=0, ub=np.inf)

        # TODO: set options properly
        res = opt.minimize(
            objective,
            coeffs0,
            args=(),
            method="SLSQP",
            bounds=bounds,
            constraints=(scipy_constraint,),
            tol=1e-7,
            callback=None,
            options=dict(maxiter=10000),
        )

        coeffs = res.x
        rms_error = objective(coeffs)

        # set the latest fit
        medium = self._make_medium(coeffs)
        return medium, rms_error

    @add_ax_if_none
    def plot(
        self,
        medium: PoleResidue = None,
        wvl_um: ArrayFloat1D = None,
        ax: Ax = None,
    ) -> Ax:
        """Make plot of model vs data, at a set of wavelengths (if supplied).

        Parameters
        ----------
        medium : :class:`.PoleResidue` = None
            medium containing model to plot against data
        wvl_um : ArrayFloat1D = None
            Wavelengths to evaluate model at for plot in micrometers.
        ax : matplotlib.axes._subplots.Axes = None
            Axes to plot the data on, if None, a new one is created.

        Returns
        -------
        matplotlib.axis.Axes
            Matplotlib axis corresponding to plot.
        """

        ax.plot(self.wvl_um, self.n_data, "x", label="n (data)")
        if self.lossy:
            ax.plot(self.wvl_um, self.k_data, "+", label="k (data)")

        if medium:
            if wvl_um is None:
                wvl_um = C_0 / self.freqs
            eps_model = medium.eps_model(C_0 / wvl_um)
            n_model, k_model = AbstractMedium.eps_complex_to_nk(eps_model)
            ax.plot(wvl_um, n_model, label="n (model)")
            if self.lossy:
                ax.plot(wvl_um, k_model, label="k (model)")

        ax.set_ylabel("n, k")
        ax.set_xlabel("Wavelength ($\\mu m$)")
        ax.legend()

        return ax

    @staticmethod
    def _validate_url_load(data_load: List):
        """Validate if the loaded data from URL is valid
            The data list should be in this format:
                [["wl",     "n"],
                 [float,  float],
                  .        .
                  .        .
                  .        .
            (if lossy)
                 ["wl",     "k"],
                 [float,  float],
                  .        .
                  .        .
                  .        .]]

        Parameters
        ----------
        data_load : List
            Loaded data from URL

        Raises
        ------
        ValidationError
            Or other exceptions
        """
        has_k = 0

        if data_load[0][0] != "wl" or data_load[0][1] != "n":
            raise ValidationError(
                "Invalid URL. The file should begin with ['wl','n']. "
                "Or make sure that you have supplied an appropriate delimiter."
            )

        for row in data_load[1:]:
            if row[0] == "wl":
                if row[1] == "k":
                    has_k += 1
                else:
                    raise ValidationError(
                        "Invalid URL. The file is not well formatted for ['wl', 'k'] data."
                    )
            else:
                # make sure the rest is float type
                try:
                    _ = [float(x) for x in row]
                except Exception as e:
                    raise ValidationError("Invalid URL. Float data cannot be recognized.") from e

        if has_k > 1:
            raise ValidationError("Invalid URL. Too many k labels.")

    @classmethod
    def from_url(cls, url_file: str, delimiter: str = ",", ignore_k: bool = False, **kwargs):
        """loads :class:`DispersionFitter` from url linked to a csv/txt file that
        contains wavelength (micron), n, and optionally k data. Preferred from
        refractiveindex.info.

        Hint
        ----
        The data file from url should be in this format (delimiter not displayed
        here, and note that the strings such as "wl", "n" need to be included
        in the file):

        * For lossless media::

            wl       n
            [float] [float]
            .       .
            .       .
            .       .

        * For lossy media::

            wl       n
            [float] [float]
            .       .
            .       .
            .       .
            wl       k
            [float] [float]
            .       .
            .       .
            .       .

        Parameters
        ----------
        url_file : str
            Url link to the data file.
            e.g. "https://refractiveindex.info/data_csv.php?datafile=data/main/Ag/Johnson.yml"
        delimiter : str = ","
            E.g. in refractiveindex.info, it'll be "," for csv file, and "\\\\t" for txt file.
        ignore_k : bool = False
            Ignore the k data if they are present, so the fitted material is lossless.

        Returns
        -------
        :class:`DispersionFitter`
            A :class:`DispersionFitter` instance.
        """
        resp = requests.get(url_file, verify=Env.current.ssl_verify)

        try:
            resp.raise_for_status()
        except Exception as e:  # pylint:disable=broad-except
            raise WebError("Connection to the website failed. Please provide a valid URL.") from e

        data_url = list(
            csv.reader(codecs.iterdecode(resp.iter_lines(), "utf-8"), delimiter=delimiter)
        )
        data_url = list(data_url)

        # first validate data
        cls._validate_url_load(data_url)

        # parsing the data
        n_lam = []
        k_lam = []  # the two variables contain [wvl_um, n(k)]
        has_k = 0  # whether k is in the data

        for row in data_url[1:]:
            if has_k == 1:
                k_lam.append([float(x) for x in row])
            elif row[0] == "wl":
                has_k += 1
            else:
                n_lam.append([float(x) for x in row])

        n_lam = np.array(n_lam)
        k_lam = np.array(k_lam)

        if has_k == 1 and not ignore_k:
            if n_lam.shape == k_lam.shape and np.allclose(n_lam[:, 0], k_lam[:, 0]):
                return cls(wvl_um=n_lam[:, 0], n_data=n_lam[:, 1], k_data=k_lam[:, 1], **kwargs)
            raise ValidationError(
                "Invalid URL. Both n and k should be provided at each wavelength."
            )

        return cls(wvl_um=n_lam[:, 0], n_data=n_lam[:, 1], **kwargs)

    @classmethod
    def from_file(cls, fname: str, **loadtxt_kwargs):  # pylint:disable=arguments-differ
        """Loads :class:`DispersionFitter` from file containing wavelength, n, k data.

        Parameters
        ----------
        fname : str
            Path to file containing wavelength (um), n, k (optional) data in columns.
        **loadtxt_kwargs
            Kwargs passed to ``np.loadtxt``, such as ``skiprows``, ``delimiter``.

        Hint
        ----
        The data file should be in this format (``delimiter`` and ``skiprows`` can be
        customized in ``**loadtxt_kwargs``):

        * For lossless media::

            wl       n
            [float] [float]
            .       .
            .       .
            .       .

        * For lossy media::

            wl       n       k
            [float] [float] [float]
            .       .       .
            .       .       .
            .       .       .

        Returns
        -------
        :class:`DispersionFitter`
            A :class:`DispersionFitter` instance.
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
