import numpy as np
import matplotlib.pyplot as plt
import nlopt
from tqdm import tqdm

import sys
sys.path.append('../..')

from tidy3d.components.medium import nk_to_eps, eps_to_nk, eps_pole_residue
from tidy3d.components.medium import PoleResidue

"""
NK data fitting tool

All frequency units are in eV currently to keep typical values on the order of unity.
This simplifies operator validation of the fit parameters, and also random initialization
of model parameters.
"""

""" Helper functions """


def unpack_complex(z):
    return z.real, z.imag


def pack_complex(r, i):
    return r + 1j * i


def unpack_coeffs(coeffs):
    assert len(coeffs) % 4 == 0
    num_poles = len(coeffs) // 4
    indices = 4 * np.arange(num_poles)

    ar = coeffs[indices + 0]
    ai = coeffs[indices + 1]
    cr = coeffs[indices + 2]
    ci = coeffs[indices + 3]

    a = pack_complex(ar, ai)
    c = pack_complex(cr, ci)
    return a, c


def pack_coeffs(a, c):
    ar, ai = unpack_complex(a)
    cr, ci = unpack_complex(c)

    stacked_coeffs = np.stack((ar, ai, cr, ci), axis=1)
    return stacked_coeffs.flatten()


def make_poles(coeffs):
    coeffs_scaled = coeffs / td.constants.HBAR
    avec, cvec = unpack_coeffs(coeffs_scaled)
    poles = [(complex(a), complex(c)) for (a, c) in zip(avec, cvec)]
    return poles


""" tests (pytest -qs fit.py) """


def test_nk_eps():
    n, k = np.random.random(2)
    eps = nk_to_eps(n, k)
    n_, k_ = eps_to_nk(eps)
    eps_ = nk_to_eps(n_, k_)
    assert eps == eps_
    assert n == n_
    assert k == k_


def test_coeffs():
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    a, c = unpack_coeffs(coeffs)
    coeffs_ = pack_coeffs(a, c)
    a_, c_ = unpack_coeffs(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(a, a_)
    assert np.allclose(c, c_)


""" Dispersion Fit Class """


class DispersionFit:
    def __init__(self, wavelengths, n_data, k_data=None):
        """Creates a DispersionFit object from measured lambda, n, k data.
        
        Parameters
        ----------
        wavelengths : array-like
            (micron) wavelengths of data.
        n_data : array-like
            Real part of the refractive index at wavelengths.
        k_data : array-like, optional
            Imaginary part of the refractive index at wavelengths 
            (positive = lossy).
        """

        self.wavelengths = wavelengths
        self.n_data = n_data

        # handle k data
        if k_data is None:
            self.k_data = np.zeros_like(n_data)
        else:
            self.k_data = k_data

        # record if lossy or not
        if np.allclose(self.k_data, 0.0):
            self.lossy = False
        else:
            self.lossy = True

        # perform some validations on shape
        assert len(self.wavelengths.shape) == 1
        assert n_data.shape == wavelengths.shape
        assert k_data.shape == wavelengths.shape

        # Convert from wavelength to eV and index to permittivity
        self.eps = nk_to_eps(self.n_data, self.k_data)
        self.num_data_points = len(self.eps)

        # set the state and initalize fields set by fit
        self.has_fit = False
        self.coeffs = None
        self.rms_error = None
        self.poles = None
        self.num_poles = None

    def model(self, wavelengths, coeffs=None):
        """Returns the nk data at a set of wavelengths; if ``coeffs`` is not
        supplied, they are taken from last fit.
        
        Parameters
        ----------
        wavelengths : array-like
            (micron) array of wavelengths to evaluate model at.
        coeffs : None, optional
            If supplied, use for pole coefficients in model, otherwise use 
            fit's last result. These are in eV, while ``eps_pole_residue``
            assumes Hz. The conversion happens in ``make_poles``.
        
        Returns
        -------
        tuple
            (n, k) arrays of n, k model evaluated at wavelengths.
        """

        # convert wavelenths to freqs
        freqs = td.constants.C_0 / wavelengths

        # select and unpack coefficients of model
        if coeffs is None:
            coeffs = self.coeffs

        # Contribution from poles
        eps_poles = eps_pole_residue(make_poles(coeffs), freqs)

        # eps_infinity always set to 1.0
        complex_epsilon = 1.0 + eps_poles

        # convert back to n,k
        n, k = eps_to_nk(complex_epsilon)
        return n, k

    def fit(
        self, num_poles=3, num_tries=100, tolerance_rms=0.0, plot=True, globalopt=True, bound=np.inf
    ):
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
        globalopt : bool, optional
            Perform global optimization (if False, does local optimization).
        bound : float, optional
            Free parameters are bound between [-bound, +bound].
        
        Returns
        -------
        tuple
            (coeffs, rms_error) optimization result array and RMS error.
        """

        # Run it a number of times.
        best_coeffs = None
        best_rms = np.inf
        pbar = tqdm(range(num_tries))
        for _ in pbar:
            coeffs, rms_error = self.fit_single(
                num_poles=num_poles, globalopt=globalopt, bound=bound
            )

            # if improvement, set the best RMS and coeffs
            if rms_error < best_rms:
                best_rms = rms_error
                best_coeffs = coeffs

            # update status
            pbar.set_description(f"best RMS error so far: {best_rms:.2e}")

            # if below tolerance, return
            if best_rms < tolerance_rms:
                print(f"\tfound optimal fit with RMS error = {best_rms:.2e}, returning")
                self.coeffs = best_coeffs
                self.rms_error = best_rms
                self.poles = make_poles(coeffs)
                return best_coeffs, best_rms

        # if exited loop, did not reach tolerance (warn)
        print(
            f"\twarning: did not find fit with RMS error under tolerance_rms of {tolerance_rms:.2e}"
        )
        print(f"\treturning best fit with RMS error {best_rms:.2e}")
        self.coeffs = best_coeffs
        self.rms_error = best_rms
        self.poles = make_poles(coeffs)
        return best_coeffs, best_rms

    def fit_single(self, num_poles=3, globalopt=True, bound=np.inf):
        """Perform a single fit to the data and return optimization result.
        """

        def constraint(coeffs, _grad):
            """Evaluates the nonlinear stability criterion of
            Hongjin Choi, Jae-Woo Baek, and Kyung-Young Jung,
            "Comprehensive Study on Numerical Aspects of Modified
            Lorentz Model Based Dispersive FDTD Formulations,"
            IEEE TAP 2019. """

            a, c = unpack_coeffs(coeffs)
            ar, ai = unpack_complex(a)
            cr, ci = unpack_complex(c)
            prstar = ar * cr + ai * ci
            res = 2 * prstar * ar - cr * (ar * ar + ai * ai)
            res[res >= 0] == 0
            return np.sum(res)

        def obj(coeffs, _grad):
            """objective function for fit """

            model_n, model_k = self.model(self.wavelengths, coeffs)
            model_eps = nk_to_eps(model_n, model_k)

            residual = self.eps - model_eps
            rms_error = np.sqrt(np.sum(np.square(np.abs(residual))) / self.num_data_points)
            cons = constraint(coeffs, _grad)
            return rms_error  # + cons    # note: not sure about 1e2 factor

        # set initial guess
        num_DoF = num_poles * 4
        coeffs0 = 2 * (np.random.random(num_DoF) - 0.5)

        # set method and objective
        method = nlopt.LN_NELDERMEAD if globalopt else nlopt.LN_SBPLX
        opt = nlopt.opt(method, num_DoF)
        opt.set_min_objective(obj)

        # set bounds
        ub = np.zeros(num_DoF, dtype=float)
        lb = np.zeros(num_DoF, dtype=float)
        indices = 4 * np.arange(num_poles)

        if self.lossy:
            # if lossy, the real parts can take on values
            lb[indices] = -bound
            ub[indices + 2] = bound
            coeffs0[indices] = -np.abs(coeffs0[indices])
            coeffs0[indices + 2] = +np.abs(coeffs0[indices + 2])
        else:
            # otherwise, they need to be 0
            coeffs0[indices] = 0
            coeffs0[indices + 2] = 0

        lb[indices + 1] = -bound
        ub[indices + 1] = bound
        lb[indices + 3] = -bound
        ub[indices + 3] = bound

        opt.set_lower_bounds(lb.tolist())
        opt.set_upper_bounds(ub.tolist())

        # opt.add_inequality_constraint(constraint)
        opt.set_xtol_rel(1e-5)
        opt.set_ftol_rel(1e-5)

        if globalopt:
            # run global optimization with opt as inner loop
            optglob = nlopt.opt(nlopt.LN_AUGLAG, num_DoF)
            optglob.set_min_objective(obj)
            optglob.set_lower_bounds(lb.tolist())
            optglob.set_upper_bounds(ub.tolist())
            optglob.add_inequality_constraint(constraint)
            optglob.set_xtol_rel(1e-5)
            optglob.set_maxeval(10000)
            optglob.set_ftol_rel(1e-7)
            optglob.set_local_optimizer(opt)
            coeffs = optglob.optimize(coeffs0)
            rms_error = optglob.last_optimum_value()
        else:
            # run single optimization with opt
            coeffs = opt.optimize(coeffs0)
            rms_error = opt.last_optimum_value()

        # set the latest fit
        self.coeffs = coeffs
        self.rms_error = rms_error
        self.poles = make_poles(coeffs)
        self.has_fit = True
        self.num_poles = num_poles
        return coeffs, rms_error

    def plot(
        self,
        wavelengths=None,
        ax=None,
        dot_sizes=25,
        linewidth=3,
        n_data_color="black",
        k_data_color="grey",
        n_model_color="firebrick",
        k_model_color="dodgerblue",
    ):
        """Make plot of model vs data, at a set of wavelengths (if supplied).
        
        Parameters
        ----------
        wavelengths : array-like, optional
            (micron) wavelengths to evaluate model at.
        ax : matplotlib.axes.Axes, optional
            axis to plot the data on.
        dot_sizes : int, optional
            Size of input data scatter plots.
        linewidth : int, optional
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

        if wavelengths is None:
            wavelengths = self.wavelengths

        model_n, model_k = self.model(wavelengths, self.coeffs)

        if ax is None:
            f, ax = plt.subplots(1, 1)

        im = ax.scatter(self.wavelengths, self.n_data, s=dot_sizes, c=n_data_color, label="n (data)")
        ax.plot(wavelengths, model_n, linewidth=linewidth, color=n_model_color, label="n (model)")
        if self.lossy:
            ax.scatter(self.wavelengths, self.k_data, s=dot_sizes, c=k_data_color, label="k (data)")
            ax.plot(
                wavelengths, model_k, linewidth=linewidth, color=k_model_color, label="k (model)"
            )
        ax.set_ylabel("value")
        ax.set_xlabel("Wavelength ($\\mu m$)")
        ax.set_title(f"{self.num_poles} pole fit")
        ax.legend()

        return im

    def as_medium(self, name=None):
        """Returns a :class:`.Medium` representation of the fit, which can be used directly in 
        a simulation.
        
        Parameters
        ----------
        name : str, optional
            Custom name of the material.
        
        Returns
        -------
        :class:`.DispersionModel`
            Material model from the fit.
        """

        return td.DispersionModel(poles=self.poles, name=name)

    def print_medium(self, name=None):
        """Prints a string representation of the fit so it can be copied and pasted into script.
        
        Parameters
        ----------
        name : str, optional
            Custom name of the material.
        """
        poles_strings = [f"        ({a}, {c}),\n" for (a, c) in self.poles]
        pole_str = ""
        for ps in poles_strings:
            pole_str += ps
        poles_str = f"[\n{(pole_str)}    ]"
        medium_str = f"td.DispersionModel(\n    poles={poles_str},\n    name='{name}'\n)"
        print("\n\nCOPY AND PASTE BELOW TO CREATE MEDIUM IN TIDY3D SCRIPT")
        print(12 * "==")
        print(medium_str)
        print(12 * "==")

    def save_poles(self, fname="poles.txt"):
        """Saves poles as a txt file containing (num_poles, 2) array.
        
        Parameters
        ----------
        fname : str, optional
            Path to file where poles are saved.
        """
        np.savetxt(fname, self.poles)


def load_poles(fname="poles.txt", name=None):
    """Loads txt file with (num_poles, 2) complex array data into
    :class:`.Medium`.
    
    Parameters
    ----------
    fname : str, optional
        Path to file containing the pole data.
    name : str, optional
        Custom name of the material.
    
    Returns
    -------
    :class:`.Medium`
        Medium using the poles
    """
    A = np.loadtxt(fname, dtype=complex)

    # handle single pole edge case
    if len(A.shape) == 1:
        A = A[None, :]

    poles = [(a, c) for (a, c) in A]
    return td.DispersionModel(poles=poles, name=name)


def load_nk_file(fname, **load_txt_kwargs):
    """Loads nk data from file, performs validation on input. Wavelengths
    must be in micron.
    
    Parameters
    ----------
    fname : str
        Path to file containing wavelength (um), n, k (optional) data in columns.
    **load_txt_kwargs
        Kwargs passed to ``np.loadtxt``.
    
    Returns
    -------
    tuple
        (wavelength, n_data, k_data) arrays containing data from file.
    """
    data = np.loadtxt(fname, **load_txt_kwargs)
    assert len(data.shape) == 2, "data must contain [wavelength, ndata, kdata] in columns"
    assert data.shape[-1] in (2, 3), "data must have either 2 or 3 rows (if k data)"
    num_data = len(data)
    if data.shape[-1] == 2:
        data = np.concatenate((data, np.zeros((num_data, 1))), axis=-1)
    wavelengths, n_data, k_data = data.T
    return wavelengths, n_data, k_data


if __name__ == "__main__":

    """ initializing fitter """

    fname1 = "n_data.csv"
    fname2 = "waveguide-material.csv"
    fname3 = "cladding-material.csv"
    fname4 = "VO2-Povinelli_metal.csv"
    fnames = [fname1, fname2, fname3, fname4]
    npoles = [1, 1, 2, 4]
    f, axes = plt.subplots(1, len(fnames))

    for fname, ax, num_poles in zip(fnames, axes, npoles):

        wavelengths, n_data, k_data = load_nk_file(fname, skiprows=1, delimiter=",")

        # change units
        if "cladding" in fname:
            wavelengths /= 1000
        if "waveguide" in fname:
            wavelengths *= 1e6

        dispFit = DispersionFit(wavelengths, n_data, k_data)

        """ performing fit """

        print(f"\nfitting with {num_poles} poles...")

        poles, rms_error = dispFit.fit(
            num_poles=num_poles, tolerance_rms=1e-4, num_tries=1000, globalopt=True, bound=np.inf
        )

        """ visualizing fit """

        dispFit.plot(ax=ax, show=False)

        """ making mediums """

        medium_si = dispFit.as_medium(name="my_medium")
        dispFit.print_medium(name="my_medium")

        """ save and load for later """

        fname_poles = f"poles_{fname[:-3]}.txt"
        dispFit.save_poles(fname=fname_poles)
        medium_si = load_poles(fname=fname_poles)

    plt.show()
