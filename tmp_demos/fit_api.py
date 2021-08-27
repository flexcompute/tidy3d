import numpy as np
import matplotlib.pyplot as plt
import nlopt
import tidy3d as td
from tqdm import tqdm

"""
NK data fitting tool

All frequency units are in eV currently to keep typical values on the order of unity.
This simplifies operator validation of the fit parameters, and also random initialization
of model parameters.
"""

""" constants """

EPSILON_0 = np.float32(8.85418782e-18)  # vacuum permittivity [F/um]
MU_0 = np.float32(1.25663706e-12)  # vacuum permeability [H/um]
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum [um/s]
ETA_0 = np.sqrt(MU_0 / EPSILON_0)  # vacuum impedance
HBAR = 6.582119569e-16  # reduced Planck constant [eV*s]

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
    """ turns array of coeffs into list of (a, c) poles """
    coeffs_scaled = coeffs / td.constants.HBAR
    avec, cvec = unpack_coeffs(coeffs_scaled)
    poles = [(complex(a), complex(c)) for (a, c) in zip(avec, cvec)]
    return poles

def nk_to_eps(n, k):
    return (n - 1j * k) ** 2

def eps_to_nk(eps):
    nk_complex = np.sqrt(np.conj(eps))
    n, k = unpack_complex(nk_complex)
    return n, k

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
        """ Creates a DispersionFit object """

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
        """ Returns the nk data at wavelengths, if coeffs supplied use those otherwise use coeffs from last fit"""

        # convert wavelenths to eV and reshape
        eV = C_0 / wavelengths * 2 * np.pi * HBAR
        eV_ = eV[:, None]

        # select and unpack coefficients of model
        if coeffs is None:
            coeffs = self.coeffs
        a, c = unpack_coeffs(coeffs)

        # get pole contributions (regular and C.C.)
        pole_contrib1 = c / (1j * eV_ - a)
        pole_contrib2 = np.conj(c) / (1j * eV_ - np.conj(a))
        pole_contribs = pole_contrib1 + pole_contrib2

        # sum pole contributions to get complex permittiviy
        complex_epsilon = 1.0 + np.sum(pole_contribs, axis=-1)

        # convert back to n,k
        n, k = eps_to_nk(complex_epsilon)
        return n, k

    def fit(self, num_poles=3, num_tries=100, tolerance_rms=1e-2, plot=True, globalopt=True, bound=np.inf):
        """ Fits data a number of times and returns best results """

        # Run it a number of times.
        best_coeffs = None
        best_rms = np.inf
        pbar = tqdm(range(num_tries))
        for _ in pbar:
            coeffs, rms_error = self.fit_single(num_poles=num_poles, globalopt=globalopt, bound=bound)

            # if improvement, set the best RMS and coeffs
            if rms_error < best_rms:
                best_rms = rms_error
                best_coeffs = coeffs

            # update status
            pbar.set_description(f'best RMS error so far: {best_rms:.2e}')

            # if below tolerance, return
            if best_rms < tolerance_rms:
                print(f"\tfound optimal fit with RMS error = {best_rms:.2e}, returning")
                self.coeffs = best_coeffs
                self.rms_error = best_rms
                self.poles = make_poles(coeffs)
                return best_coeffs, best_rms

        # if exited loop, did not reach tolerance (warn)
        print(f"\twarning: did not find fit with RMS error under tolerance_rms of {tolerance_rms:.2e}")
        print(f"\treturning best fit with RMS error {best_rms:.2e}")
        self.coeffs = best_coeffs
        self.rms_error = best_rms
        self.poles = make_poles(coeffs)
        return best_coeffs, best_rms

    def fit_single(self, num_poles=3, globalopt=True, bound=np.inf):
        """ Fits the data a single time and return optimization result """

        def constraint(coeffs, _grad):
            """  Evaluates the nonlinear stability criterion of
                    Hongjin Choi, Jae-Woo Baek, and Kyung-Young Jung,
                    "Comprehensive Study on Numerical Aspects of Modified
                    Lorentz Model Based Dispersive FDTD Formulations,"
                    IEEE TAP 2019.
            """

            a, c = unpack_coeffs(coeffs)
            ar, ai = unpack_complex(a)
            cr, ci = unpack_complex(c)
            prstar = ar * cr + ai * ci
            res = 2 * prstar * ar - cr * (ar * ar + ai * ai)
            res[res >= 0] == 0
            return np.sum(res)

        def obj(coeffs, _grad):
            """ objective function for fit """

            model_n, model_k = self.model(self.wavelengths, coeffs)
            model_eps = nk_to_eps(model_n, model_k)

            residual = self.eps - model_eps
            rms_error = np.sqrt(np.sum(np.square(np.abs(residual))) / self.num_data_points)
            cons = constraint(coeffs, _grad)
            return rms_error #+ cons    # note: not sure about 1e2 factor

        # set initial guess
        num_DoF = num_poles * 4
        # coeffs0 = 10 * np.random.random(num_DoF)
        coeffs0 = 2 * (np.random.random(num_DoF) - 0.5)
        # coeffs0 = 1 * np.random.random(num_DoF)

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

    def plot(self,
        wavelengths=None,
        ax=None,
        show=True,
        dot_sizes=25,
        linewidth=3,
        n_data_color='black',
        k_data_color='grey',
        n_model_color='firebrick',
        k_model_color='dodgerblue'):
        """ make plot of model vs data, using wavelengths (if supplied) """

        if wavelengths is None:
            wavelengths = self.wavelengths

        model_n, model_k = self.model(wavelengths, self.coeffs)

        if ax is None:
            f, ax = plt.subplots(1, 1)
        ax.scatter(self.wavelengths, self.n_data,
            s=dot_sizes,
            c=n_data_color,
            label="n (data)")
        ax.plot(wavelengths, model_n,
            linewidth=linewidth,
            color=n_model_color,
            label="n (model)")
        if self.lossy:
            ax.scatter(self.wavelengths, self.k_data,
                s=dot_sizes,
                c=k_data_color,
                label="k (data)")
            ax.plot(wavelengths, model_k,
                linewidth=linewidth,
                color=k_model_color,
                label="k (model)")
        ax.set_ylabel("value")
        ax.set_xlabel("Wavelength ($\mu m$)")
        ax.set_title(f"{self.num_poles} pole fit")
        ax.legend()

        if show and ax is not None:
            plt.show()
        else:
            return ax

    def as_medium(self, name=None):
        """ Returns a td.Medium representation of the fit so it can be copied and pasted into script """
        model = td.DispersionModel(poles=self.poles)
        return td.Medium(epsilon=model, name=name)

    def print_medium(self, name=None):
        """ Prints a string representation of the fit so it can be copied and pasted into script """
        poles_strings = [f'\t\t({a}, {c}),\n' for (a, c) in self.poles]
        pole_str = ''
        for ps in poles_strings:
            pole_str += ps
        poles_str = f'[\n{(pole_str)}]'
        model_str = f'tidy3d.DispersionModel(poles={poles_str})'
        medium_str = f'my_medium = tidy3d.Medium(\n\tepsilon={model_str}, name="{name}")'
        print('\n\nCOPY AND PASTE BELOW TO CREATE MEDIUM IN TIDY3D SCRIPT')
        print(12*'==')
        print(medium_str)
        print(12*'==')

    def save_poles(self, fname='poles.txt'):
        """ saves poles as a txt file containing (num_poles, 2) array """
        np.savetxt(fname, self.poles)

def load_poles(fname='poles.txt'):
    """ loads txt file with (num_poles, 2) complex array data into Medium """
    A = np.loadtxt(fname, dtype=complex)
    if len(A.shape) == 1:
        A = A[None, :]
    poles = [(a, c) for (a, c) in A]
    return td.Medium(epsilon=td.DispersionModel(poles=poles))

def load_nk_file(fname, **load_txt_kwargs):
    """ Wrapper for np.loadtxt
        https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
    """
    data = np.loadtxt(fname, **load_txt_kwargs)
    assert (
        len(data.shape) == 2
    ), "data must contain [wavelength, ndata, kdata] in columns"
    assert data.shape[-1] in (2, 3), "data must have either 2 or 3 rows (if k data)"
    num_data = len(data)
    if data.shape[-1] == 2:
        data = np.concatenate((data, np.zeros((num_data, 1))), axis=-1)
    wavelengths, n_data, k_data = data.T
    return wavelengths, n_data, k_data

if __name__ == '__main__':

    """ initializing fitter """

    fname1 = 'n_data.csv'
    fname2 = 'waveguide-material.csv'
    fname3 = 'cladding-material.csv'
    fname4 = 'VO2-Povinelli_metal.csv'
    fnames = [fname1, fname2, fname3, fname4]
    npoles = [1, 1, 2, 4]
    f, axes = plt.subplots(1, len(fnames))

    for fname, ax, num_poles in zip(fnames, axes, npoles):

        wavelengths, n_data, k_data = load_nk_file(fname, skiprows=1, delimiter=',')

        # change units
        if 'cladding' in fname:
            wavelengths /= 1000
        if 'waveguide' in fname:
            wavelengths *= 1e6

        dispFit = DispersionFit(wavelengths, n_data, k_data)

        """ performing fit """

        print(f'\nfitting with {num_poles} poles...')

        poles, rms_error = dispFit.fit(
                                num_poles=num_poles,
                                tolerance_rms=1e-4,
                                num_tries=1000,
                                globalopt=True,
                                bound=np.inf)

        """ visualizing fit """

        dispFit.plot(ax=ax, show=False)

        """ making mediums """

        medium_si = dispFit.as_medium(name='my_medium')
        dispFit.print_medium(name='my_medium')

        """ save and load for later """

        fname_poles = f'poles_{fname[:-3]}.txt'
        dispFit.save_poles(fname=fname_poles)
        medium_si = load_poles(fname=fname_poles)

    plt.show()